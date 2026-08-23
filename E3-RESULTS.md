# E3 results — scored against SPEC-E3 (registered 2026-08-23, pre-run)

## Primary: held-out panel (mean vs {v3peak, soup, v3b}, 2048 x 3 eval seeds per pair)

| arm | s0 | s1 | s2 | 3-seed mean |
|---|---|---|---|---|
| C1 pure self-play | 0.4689 | 0.4813 | 0.4812 | **0.4771** |
| C2 pool | 0.4634 | 0.4628 | 0.4663 | **0.4642** |

Gap C2 - C1 = **-0.013**, and the separation is clean: every C2 seed scores below
every C1 seed. Secondary (vs v2): C1 0.521, C2 0.506 — both arms edge past their
init, C1 by more.

**E3-H1 (C2 > C1 on panel, 55%; gap interval [-0.01, +0.06]): directional MISS,
interval MISS** (observed -0.013, marginally outside).

## Exploitability screens (warm attack, E1 recipe, confirmed peaks = 3 x 2048)

- C1 best final (e3_c1_s1): exploit peak **0.494**
- C2 best final (e3_c2_s2): exploit peak **0.536**

**E3-H2 (exploit(C1) - exploit(C2) in [0.00, +0.10], 65%): directional MISS, interval
MISS.** Observed gap **-0.042** — the pool-trained net is MORE exploitable, not less,
and at confirmed-peak precision (~+/-0.013 per reading) the gap is significant, so the
registered near-boundary upgrade to 3-seed screens is not triggered. Both screens ran
on the same local device.

## The registered null gate FIRES

C1 >= C2 on BOTH the held-out panel and exploitability. Per SPEC-E3: **pool training
at napkin scale does not help** — on this pool (3 frozen members: v1 + two verified-
distinct nemeses of v2), this budget (2,500 iters, ~810k games/run), this game. It
cost a little strength AND a little robustness relative to plain self-play.

**E3-H3 (ladder) is not scored: no submission.** The submission was registered for the
world where pool training produced a plausibly better net. Submitting C1's final would
repeat the known-failing move — its profile (beats v2 offline 0.52, beats v3b, loses
to the 8000-iter lineage nets) is exactly the v3peak/soup pattern that lost ladder
points twice. v2 stays the incumbent.

## Series picture after E1+E2+E3

1. E1: exploitability tracks ladder order POSITIVELY (late strong nets are LESS
   exploitable); the self-play-overfitting story inverted.
2. E2: soup's robustness is real (unattackable from twin and non-twin starts);
   weight averaging is a genuine robustness intervention here; best-response training
   adds little over a good starting net.
3. E3: opponent-diversity training (uniform pool with verified-distinct members)
   HURT both strength and robustness slightly at this budget, against the classic
   population-training expectation.

The motivating anomaly stands unexplained by everything measured offline: soup beats
v2 head-to-head, resists every attack we can mount, and still scored worst on the
ladder. The population rewards something none of our offline instruments capture.
Per the registered gate, the next move (napkin-mirror: measure what the POPULATION
plays, instead of training against ourselves harder) is the user's decision.

## Housekeeping

- PFSP (C3) does not run: it was conditional on C2 > C1.
- E4 (UTTT replication) does not run: conditional on E3 producing a winner.
- Ladder incumbent remains v2 (30.93 / rank 49 in the current field).
