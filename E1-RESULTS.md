# E1 results — scored against SPEC-E1 (registered 2026-08-22, pre-measurement)

All numbers: k=4 paired-opening evals; peaks are 9x2048-game confirmations (pooled CI
~+/-0.007) at the 3-seed mean-curve argmax; curves 1024 games/snapshot, seeds 0,1,2.

## Measurements

| target | confirmed peak | last-5 mean | argmax | ladder (historical) |
|---|---|---|---|---|
| v1 | **0.612** | 0.608 | it2400 | rank 92 era |
| v2 | **0.514** | 0.519 | it2300 | 32.24 |
| v3peak | **0.477** | 0.464 | it2400 | 31.65 |
| soup | **0.379** | 0.380 | it1800 | 31.46 |

Panel (2048 x 3 eval seeds per pair): greedy saturates (0.97-0.98 for all four nets, no
resolution). Informative refs: v1 0.685, v2 0.789, v3peak 0.718, soup 0.755 (panel
means); head-to-head: soup beats v2 0.601, beats v1 0.685; v3peak beats v2 0.543,
beats v1 0.629; v2 beats v1 0.603.

## Hypothesis scores

**H1 (magnitude, joint "every peak > 0.52" at 65%): MISS.** Only v1 clears 0.52.
Per-target 80% intervals: v1 [0.53,0.68] -> 0.612 **hit**; v2 [0.50,0.60] -> 0.514
**hit**; v3peak [0.51,0.63] -> 0.477 **miss** (below); soup [0.49,0.58] -> 0.379
**miss** (far below). Calibration lesson: the gate net (v3b, weak) anchored the scale
too high, and below-parity outcomes were never seriously entertained.

**H2 (ordering v1 > v3peak > v2 > soup): exact order MISS** (observed v1 > v2 >
v3peak > soup — v2/v3peak swapped). Sub-predictions: "soup least exploitable" (60%)
**HIT**; "v1 most exploitable" (75%) **HIT**.

**H3 (panel matches ladder strictly AND inverse-exploitability does not; 55%): MISS**
as a conjunction. Clause 1 miss: panel says v2 > soup > v3peak vs ladder v2 > v3peak >
soup; the v3peak/soup swap is robust (identical ref sets, soup wins on each ref
separately). Clause 2 hit: inverse-exploitability (soup > v3peak > v2) is the exact
REVERSE of the ladder order.

## The finding the hypotheses didn't predict

Exploitability is **positively** ordered with ladder score in this lineage:
v2 (0.514 / 32.24) > v3peak (0.477 / 31.65) > soup (0.379 / 31.46) — a perfect rank
match, in the OPPOSITE direction from the self-play-overfitting story. The motivating
hypothesis for this series predicted the late, offline-strong nets would be MORE
exploitable; they are dramatically less. With n=3 an exact rank match has p ~ 1/6 by
chance, so this is suggestive, not established.

The anomaly that motivated the series now looks stranger, not explained: soup beats v2
head-to-head (0.601), beats every panel ref more convincingly than v2's own numbers,
and is by far the hardest net for a best response to attack — yet it scored 0.8 ladder
points WORSE than v2. Both of our offline instruments now favour the late nets; the
926-bot population disagrees. Whatever the population rewards, neither head-to-head
strength nor worst-case robustness (as measured here) captures it.

## Caveat that limits interpretation

For v3peak and soup the "peak" is BELOW 0.5: warm-start best-response training left
the learner weaker than the twin it started as for the entire budget. For those
targets E1 did not measure exploitability at all — it measured that THIS procedure
(warm start from the target, lr 2e-4, 2.5k iters) fails, and how badly. Two readings
compatible with the data:
1. the late nets are genuinely robust (no cheap exploits within reach), or
2. fine-tuning FROM those particular weight points is unstable — soup especially is a
   weight-average, plausibly a flat/special point whose basin any SGD step exits.
E2's bootstrap axis separates these: a non-twin attacker (e.g. bootstrapped from v2 or
from scratch) attacking soup distinguishes "soup is robust" from "soup is a bad place
to START an attacker".

## Validity notes

- The below-parity results are not a harness offset: the same procedure reads 0.612 on
  v1, ~0.54 on v3b (gate), and the lr=0 twin null reads 0.50 exactly.
- Confirmed peaks vs their own curves agree (no winner's-curse gap).
- Distinct final positions ~1550/2048 on confirms, ~860/1024 on curve points.
- The v3peak/soup ladder gap (0.19 points, different days, drifting field) was thin
  ground truth for H3; scored as registered anyway.
