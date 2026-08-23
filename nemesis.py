#!/usr/bin/env python3
"""napkin-nemesis harness: exploitability measurement, orchestrated.

Thin driver over the vendored engine (napkin_c4.py). It spawns `train-gpu` in
best-response mode per seed, evals every snapshot against the frozen target
with `eval-net`, and emits what the protocol registers: the mean curve across
seeds, the value at the mean-curve argmax snapshot, the mean over the last 5
snapshots, and confirmation readings (2048 games x 3 eval seeds) at the argmax.
Nothing here plays games itself -- the engine is the single implementation.

  nemesis.py exploit --target CKPT --tag NAME [--bootstrap warm|scratch]
  nemesis.py panel   --nets a.pt,b.pt,...  [--refs greedy,x.pt,...]
"""
import argparse
import json
import os
import re
import subprocess
import sys

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "napkin_c4.py")


def engine(args, log=None):
    r = subprocess.run([PY, ENGINE] + [str(a) for a in args],
                       capture_output=True, text=True, cwd=HERE)
    if log:
        with open(log, "a") as f:
            f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"engine {args[0]} failed: {r.stderr[-800:]}")
    return r.stdout


EVAL_RE = re.compile(r"score (\d\.\d+) \(95% Wilson (\d\.\d+)\.\.(\d\.\d+)\).*?"
                     r"(\d+) distinct final positions", re.S)


def _eval_args(net, vs, games, seed):
    sel = ["--vs", vs] if vs in ("greedy", "random") else ["--vs-net", vs]
    # k=4 opening plies is the registered standard (k=8 compresses differences,
    # measured in the source repo); eval-net's own default is 8
    return (["eval-net", "--net", net] + sel +
            ["--games", games, "--seed", seed, "--open-plies", 4])


def eval_net(net, vs, games, seed):
    """One eval-net reading. `vs` is a checkpoint path or 'greedy'/'random'."""
    out = engine(_eval_args(net, vs, games, seed))
    m = EVAL_RE.search(out)
    if not m:
        raise RuntimeError(f"unparseable eval-net output: {out[-300:]}")
    return {"score": float(m.group(1)), "lo": float(m.group(2)),
            "hi": float(m.group(3)), "distinct": int(m.group(4))}


def cmd_exploit(a):
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    snaps = list(range(a.snapshot_every, a.iters + 1, a.snapshot_every))
    curves = {}
    for seed in range(a.seeds):
        stem = f"out/{a.tag}_s{seed}"
        train = ["train-gpu", "--opponent", a.target, "--lr", a.lr,
                 "--tau", a.tau, "--iters", a.iters, "--seed", seed,
                 "--eval-every", a.snapshot_every,
                 "--snapshot-every", a.snapshot_every, "--out", f"{stem}.pt"]
        if a.bootstrap == "warm":
            train += ["--init", a.target]
        elif a.bootstrap.startswith("ckpt:"):
            train += ["--init", a.bootstrap[5:]]
        print(f"[{a.tag}] seed {seed}: training {a.iters} iters "
              f"({a.bootstrap} start)", flush=True)
        engine(train, log=f"{HERE}/{stem}.log")
        pts = {}
        for it in snaps:
            r = eval_net(f"{stem}_it{it:05d}.pt", a.target, a.curve_games, 1)
            pts[it] = r
            print(f"[{a.tag}] s{seed} it{it:05d} {r['score']:.3f} "
                  f"({r['lo']:.3f}..{r['hi']:.3f}) [{r['distinct']} distinct]",
                  flush=True)
        curves[seed] = pts

    mean_curve = {it: sum(curves[s][it]["score"] for s in curves) / len(curves)
                  for it in snaps}
    arg = max(mean_curve, key=mean_curve.get)
    confirm = [eval_net(f"out/{a.tag}_s{seed}_it{arg:05d}.pt", a.target,
                        a.confirm_games, es)
               for seed in range(a.seeds) for es in (2, 3, 4)]
    peak = sum(c["score"] for c in confirm) / len(confirm)
    last5 = sum(mean_curve[it] for it in snaps[-5:]) / len(snaps[-5:])
    summary = {"tag": a.tag, "target": a.target, "bootstrap": a.bootstrap,
               "iters": a.iters, "seeds": a.seeds,
               "mean_curve": mean_curve, "argmax_iter": arg,
               "peak_confirmed": round(peak, 4),
               "peak_confirm_readings": confirm,
               "last5_mean": round(last5, 4),
               "curves": {s: {it: p["score"] for it, p in pts.items()}
                          for s, pts in curves.items()}}
    path = f"{HERE}/out/{a.tag}_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[{a.tag}] exploit(target={a.target}, {a.bootstrap}): "
          f"peak {peak:.4f} at it{arg} (3x2048 confirmed), "
          f"last5 {last5:.4f} -> {path}", flush=True)
    return 0


def cmd_panel(a):
    """Score each net against each reference, seeds 2/3/4, 2048 games."""
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    rows = []
    for net in a.nets.split(","):
        for ref in a.refs.split(","):
            if os.path.abspath(net) == os.path.abspath(ref):
                continue
            rs = [eval_net(net, ref, a.games, es) for es in (2, 3, 4)]
            sc = sum(r["score"] for r in rs) / len(rs)
            rows.append({"net": net, "ref": ref, "score": round(sc, 4),
                         "readings": [r["score"] for r in rs],
                         "distinct": [r["distinct"] for r in rs]})
            print(f"{net} vs {ref}: {sc:.4f} "
                  f"{[r['score'] for r in rs]}", flush=True)
    with open(f"{HERE}/out/panel.json", "w") as f:
        json.dump(rows, f, indent=1)
    return 0


def cmd_perturb(a):
    """Write a copy of a checkpoint with per-tensor gaussian noise added:
    sigma = --scale x that tensor's own weight std (E2's A4 bootstrap)."""
    import torch
    torch.manual_seed(a.seed)
    ck = torch.load(a.net, map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    for k, w in sd.items():
        if w.dtype.is_floating_point and w.numel() > 1:
            sd[k] = w + torch.randn_like(w) * (a.scale * w.std())
    torch.save(ck, a.out)
    print(f"perturbed {a.net} -> {a.out} (scale {a.scale}, seed {a.seed})")
    return 0


def cmd_disagree(a):
    """Pairwise argmax disagreement between nets, on states drawn from each
    net's OWN games vs the frozen target (the states where exploits manifest;
    ordinary target-vs-target states are the wrong distribution)."""
    import sys as _sys
    _sys.path.insert(0, HERE)
    import torch
    from napkin_c4 import TensorC4, improved_policy, load_aznet, random_openings

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = a.nets.split(",")
    nets = [load_aznet(pp, dev)[0] for pp in paths]
    tgt = load_aznet(a.target, dev)[0]
    B = 512
    counts = {}
    with torch.no_grad():
        for di, driver in enumerate(nets):
            torch.manual_seed(a.seed + di)
            t = TensorC4(B, dev)
            random_openings(t, a.open_plies)
            seat = torch.arange(B, device=dev) % 2
            drv = torch.zeros(B, dtype=torch.long, device=dev)
            for _ in range(a.max_plies):
                mine = (t.side == seat) & ~t.done
                if bool(mine.any()):
                    moves = []
                    for net in nets:
                        _, pi, _ = improved_policy(t, net, tau=0.2, depth=2)
                        moves.append(pi.argmax(dim=1))
                    n_states = int(mine.sum())
                    for i in range(len(nets)):
                        for j in range(i + 1, len(nets)):
                            key = (i, j)
                            d0, n0 = counts.get(key, (0, 0))
                            dd = int((moves[i][mine] != moves[j][mine]).sum())
                            counts[key] = (d0 + dd, n0 + n_states)
                    drv = moves[di]
                _, tpi, _ = improved_policy(t, tgt, tau=0.2, depth=2)
                t.step(torch.where(mine, drv, tpi.argmax(dim=1)))
                if bool(t.done.all()):
                    break
    for (i, j), (d, n) in sorted(counts.items()):
        print(f"disagree {paths[i].split('/')[-1]} vs {paths[j].split('/')[-1]}: "
              f"{d / n:.4f} (n={n})")
    return 0


def cmd_selfcheck(a):
    # the fragile part is parsing the engine's human-oriented output; pin it
    canned = ("out/x.pt (trunk 160-112) vs greedy: score 0.875 "
              "(95% Wilson 0.807..0.922), win 0.850 draw 0.050, "
              "128 games from 8 random opening plies, 116 distinct final positions")
    m = EVAL_RE.search(canned)
    assert m and m.groups() == ("0.875", "0.807", "0.922", "116"), m
    # the registered protocol: k=4 openings on every reading, both vs modes
    a1 = _eval_args("a.pt", "b.pt", 1024, 1)
    a2 = _eval_args("a.pt", "greedy", 2048, 3)
    assert a1[-2:] == ["--open-plies", 4] and "--vs-net" in a1, a1
    assert a2[-2:] == ["--open-plies", 4] and "--vs" in a2, a2
    print("nemesis selfcheck: eval-net parse + k=4 invocation contracts OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("exploit")
    ex.add_argument("--target", required=True)
    ex.add_argument("--tag", required=True)
    ex.add_argument("--bootstrap", default="warm",
                    help="warm | scratch | ckpt:<path>")
    ex.add_argument("--iters", type=int, default=2500)
    ex.add_argument("--snapshot-every", type=int, default=100)
    ex.add_argument("--seeds", type=int, default=3)
    ex.add_argument("--lr", default="2e-4")
    ex.add_argument("--tau", default="0.2")
    ex.add_argument("--curve-games", type=int, default=1024)
    ex.add_argument("--confirm-games", type=int, default=2048)
    pa = sub.add_parser("panel")
    pa.add_argument("--nets", required=True)
    pa.add_argument("--refs", default="greedy")
    pa.add_argument("--games", type=int, default=2048)
    pt = sub.add_parser("perturb")
    pt.add_argument("--net", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--scale", type=float, default=0.05)
    pt.add_argument("--seed", type=int, default=7)
    dg = sub.add_parser("disagree")
    dg.add_argument("--nets", required=True)
    dg.add_argument("--target", required=True)
    dg.add_argument("--open-plies", type=int, default=4)
    dg.add_argument("--max-plies", type=int, default=70)
    dg.add_argument("--seed", type=int, default=11)
    sub.add_parser("selfcheck")
    a = ap.parse_args()
    return {"exploit": cmd_exploit, "panel": cmd_panel, "perturb": cmd_perturb,
            "disagree": cmd_disagree, "selfcheck": cmd_selfcheck}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
