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


def eval_net(net, vs, games, seed):
    """One eval-net reading. `vs` is a checkpoint path or 'greedy'/'random'."""
    sel = ["--vs", vs] if vs in ("greedy", "random") else ["--vs-net", vs]
    out = engine(["eval-net", "--net", net] + sel +
                 ["--games", games, "--seed", seed])
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("exploit")
    ex.add_argument("--target", required=True)
    ex.add_argument("--tag", required=True)
    ex.add_argument("--bootstrap", default="warm", choices=("warm", "scratch"))
    ex.add_argument("--iters", type=int, default=2500)
    ex.add_argument("--snapshot-every", type=int, default=100)
    ex.add_argument("--seeds", type=int, default=3)
    ex.add_argument("--lr", default="5e-4")
    ex.add_argument("--tau", default="0.2")
    ex.add_argument("--curve-games", type=int, default=1024)
    ex.add_argument("--confirm-games", type=int, default=2048)
    pa = sub.add_parser("panel")
    pa.add_argument("--nets", required=True)
    pa.add_argument("--refs", default="greedy")
    pa.add_argument("--games", type=int, default=2048)
    a = ap.parse_args()
    return {"exploit": cmd_exploit, "panel": cmd_panel}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
