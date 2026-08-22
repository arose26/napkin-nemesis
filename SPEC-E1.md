# SPEC-E1 — is exploitability real, and does it track our anomaly?

Registered 2026-08-22, BEFORE any E1 measurement. Engine: napkin_c4.py vendored at
napkin-100k-connect4 @ 162110e (post collection-fix). Harness: nemesis.py @ this commit.

## Question

Measure exploit() — the peak score a warm-started best-response net reaches against a
frozen target — for the four connect-4 lineage nets, and test whether exploitability or
a fixed-panel score better tracks their known ladder ordering.

## Targets (pinned)

| name | file | md5 | ladder |
|---|---|---|---|
| v1 | gpunet.pt | 2cee80627d4be5d133710a8f49324bad | rank 92 (no score recorded) |
| v2 | gpunet_v2.pt | c15f9851a0f05fb063bdb9bd317e9de8 | 32.24 (rank 34) |
| v3peak | gpunet_v3peak.pt | bafe63b1762407fbabe12447bb5e0cda | 31.65 |
| soup | soup_v3full.pt | 263347cd3481d089b7d4a53c741ebd54 | 31.46 |

## Procedure (fixed)

- `nemesis.py exploit`: bootstrap **warm** (init = target), **lr 2e-4**, tau 0.2,
  eps 0.08, opening 10 turns, depth 2, batch/buffer engine defaults, **2,500 iters**,
  snapshot every 100. Training seeds 0,1,2. The lr was gate-tuned on v3b (outside the
  target set): at lr 5e-4 a warm start dips to 0.358 and never recovers parity even at
  2,500 iters (0.486); at lr 2e-4 it barely dips (0.478), crosses parity by it400, and
  plateaus at ~0.537 pooled (see Disclosures).
- Curve: every snapshot vs the target, 1024 games, eval seed 1, paired openings, k=4
  random opening plies (the series standard; eval-net's default 8 compresses
  differences and is overridden), distinct-position counts recorded.
- Headline **peak** = mean of 9 readings (3 training seeds x eval seeds 2,3,4, 2048
  games) at the argmax of the 3-seed mean curve. Also reported: mean over the last 5
  snapshots of the mean curve. No max-over-snapshots anywhere.
- **Panel** (fixed yardsticks, disjoint from nemesis machinery): each target vs
  {scripted greedy, v1, v2}, self-pairings excluded, 2048 games x eval seeds 2,3,4.
  Panel score = mean over its references.
- Order: this spec commits BEFORE the panel or any exploit run. Validity gates already
  passed today on the fixed engine: twin-null 0.492 (selfcheck, 8,627 z-labels checked),
  harness-level lr=0 null 0.481..0.521, positive control (scratch vs scripted greedy)
  climbed to ~0.80 explore-mode within 300 iters. Single-reading noise floor ~0.03.

## Hypotheses

**E1-H1 (magnitude).** Every lineage net is exploitable: each target's confirmed peak
is **> 0.52** (above parity beyond pooled noise; the 9x2048 confirmation has CI
~+/-0.007). Confidence 65%. Per-target 80% intervals for the confirmed peak, calibrated
from the v3b gate plateau of ~0.537 (a WEAK target; stronger targets expected lower,
the weakest lineage net expected higher):

| target | 80% interval | point |
|---|---|---|
| v1 | [0.53, 0.68] | 0.59 |
| v2 | [0.50, 0.60] | 0.545 |
| v3peak | [0.51, 0.63] | 0.56 |
| soup | [0.49, 0.58] | 0.53 |

**E1-H2 (ordering).** Peak exploitability, most to least: **v1 > v3peak > v2 > soup**.
Reasoning registered: v1 is the weakest/earliest net (most holes); v3peak is a
late-lineage single checkpoint (self-play-specialised); soup's weight averaging smooths
decision boundaries (least exploitable). Scored three ways: exact order (main),
"soup least exploitable" (binary), "v1 most exploitable" (binary). Confidence: exact
order 35%, soup-least 60%, v1-most 75%.

**E1-H3 (which offline measure tracks the ladder).** Known ladder order: v2 (32.24) >
v3peak (31.65) > soup (31.46). Prediction: the panel ordering over these three nets
matches the ladder order strictly, and the inverse-exploitability ordering does NOT.
Confidence 55%. If BOTH match, that is a hit for exploitability as a predictor and is
the more interesting world; if NEITHER matches, our offline instruments do not explain
the ladder and napkin-mirror rises.

## Disclosures

- The E0 gate measured exploit-style curves against **v3b** (md5
  8ddffc258679778f94b70708fed8b657), deliberately outside the target set, in four arms
  seen before H1's intervals were written: lr 5e-4 x800 iters (k=4: 0.358/0.394/0.450/
  0.440/0.447), its +1700 continuation (0.422/0.468/0.486 at eff. 1200/1800/2500),
  lr 2e-4 x800 (0.478/0.501/0.514), and the recipe-final continuous lr 2e-4 x2500
  (0.480/0.536/0.533/0.521/0.552/0.528, final 0.551; pooled last-6 ~0.537 +/- 0.013).
  The H2 ordering and H3 predictions were written before ANY exploit or panel number
  against the four targets existed.
- The broken-harness run's four curve points (0.008-0.020) are void: root cause was a
  degenerate z distribution (unmirrored exploration), fixed and controlled at 162110e.
- The old NEMESIS.md N1 was contaminated and is superseded by this spec.

## Interpretation gates

If all four peaks are <= 0.53 (within ~2 noise floors of 0.5), exploitability at this
budget cannot explain the anomaly; report as such and proceed to the E3 decision with
that knowledge. E1 makes no claim about POPULATION robustness either way — that
construct is only touched by the panel (weak proxy) and E3's ladder submission.
