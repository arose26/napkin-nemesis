# SPEC-E3 — does pool training beat pure self-play? (the payoff phase)

Registered 2026-08-23, after E1/E2 (results known, disclosed), BEFORE any E3 run.
Engine: vendored @ 35e7171 (adds pool training; pool twin-null 0.481 in selfcheck,
real-loop lr=0 pool null 0.479..0.512, seat decorrelated from pool slot).

## Arms (both from init = v2, lr 2e-4, tau 0.2, 2,500 iters, seeds 0,1,2)

- **C1** pure self-play continuation (the lineage that produced v3peak/soup, but at
  lr 2e-4 rather than v3full's 5e-4 — disclosed departure, shared by both arms).
- **C2** pool: `self,self,self,v1,A1NEM,A4NEM` — 50% self-play slots, 16.7% each of
  v1 (2cee8062...), the E1 warm nemesis vs v2 (e1_v2_s0_it02300), and the E2 perturbed
  nemesis vs v2 (e2_a4_s0_it02500). Distinctness measured: pairwise argmax
  disagreement 0.33-0.45; v1 vs v2 head-to-head 0.397.
- C3 (PFSP) deferred: runs only if C2 beats C1 on the primary outcome (registered as
  conditional follow-up, not part of this spec's scoring).

## Outcomes, in order of authority

1. **Held-out panel** (primary): mean score vs {v3peak, soup, v3b} — none in any pool,
   none the init — 2048 games x eval seeds 2,3,4, k=4, on each arm's final net
   (3-seed mean).
2. vs v2 (the deployment-relevant number; semi-held-out, flagged).
3. Exploitability screen of each arm's best-seed final: warm-start attack, E1 recipe,
   1 seed, 2,500 iters (screening precision ~+/-0.03).
4. **Ladder** (the series' point): ONE submission of the winning arm's best net,
   scored against a SAME-DAY reading of the standing v2 baseline.

## Hypotheses

- **E3-H1 (panel).** C2 > C1 on the held-out panel mean. Directional confidence 55%
  (E1's reversal weakened the mechanism story; honesty over bravado). 80% interval on
  the gap C2-C1: [-0.01, +0.06], pt +0.02.
- **E3-H2 (exploitability).** C2's final is harder to attack than C1's:
  exploit(C1) - exploit(C2) in [0.00, +0.10], pt +0.04, directional 65%.
- **E3-H3 (ladder).** The submitted winner lands within [-0.8, +0.4] of the same-day
  v2 baseline (pt -0.2); the sub-claim "does not lose more than 0.4" at 50%.
  Context disclosed: the previous self-play continuation lost ~0.6-0.8 ladder points.
- **Null gate.** If C1 >= C2 on BOTH (1) and (3): pool training at napkin scale does
  not help — reportable, and the series' next move (napkin-mirror: measuring what the
  POPULATION plays) goes to the user as a decision, since E1 already showed our
  offline instruments and the ladder disagree about what "better" means.

## Validity preconditions (from PLAN's N5 lessons, all already measured)

- pool twin null ~0.5 through selfcheck AND the real loop: PASSED (0.481; 0.479-0.512)
- best-response procedure can climb: PASSED (v1 0.612, greedy 0.80)
- pool members genuinely distinct: PASSED (disagreement 0.33-0.45)
- noise floor: single k=4 1024-game reading ~+/-0.03; panel readings pooled 6x2048.
