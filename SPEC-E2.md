# SPEC-E2 — bootstrapping bake-off, and resolving E1's confound

Registered 2026-08-22, after E1 (results known and disclosed), BEFORE any E2 run.
Engine unchanged (vendored @ 162110e); recipe identical to E1: lr 2e-4, tau 0.2,
eps 0.08, opening 10 turns, depth 2, 2,500 iters, snapshots every 100, seeds 0,1,2;
curves 1024 games @ k=4, eval seed 1; peaks = 9x2048 confirmations at the mean-curve
argmax.

## Questions

1. Which antagonist bootstrap finds exploits cheapest at equal budget? (the "direct
   copy of the main agent or something else" axis)
2. E1's confound: v3peak/soup "peaks" below 0.5 mean warm-start-FROM-target training
   FAILED there — is soup genuinely robust, or is the soup weight-point just a bad
   place to START an attacker?

## Arms

On frozen target **v2** (c15f9851a0f05fb063bdb9bd317e9de8):
- **A1 warm** = REUSE of e1_v2 (peak 0.514, KNOWN before this registration; all other
  arms' predictions are made knowing it).
- **A2 scratch** (random init).
- **A3 ckpt:v1** (the target's ancestor; starts at its measured head-to-head 0.397).
- **A4 perturbed twin**: v2 + per-tensor gaussian noise, sigma = 0.05 x that tensor's
  weight std, seed 7 (`nemesis.py perturb`; the artifact scores ~0.445 vs v2 at init,
  measured in a 256-game smoke before registration).

On frozen target **soup** (263347cd3481d089b7d4a53c741ebd54):
- **B1 ckpt:v2**: a strong NON-twin attacker (starts at its measured 0.399 vs soup).

## Hypotheses

**E2-H1 (bootstrap ranking on v2).** Predicted peak order: A1 >= A4 > A3 > A2.
80% intervals: A4 [0.46, 0.55] pt 0.51; A3 [0.40, 0.56] pt 0.48 (wide: transfer from a
weaker ancestor is genuinely unknown); A2 [0.30, 0.48] pt 0.40. Exact-order confidence
30%; "A2 last" 65%.

**E2-H2 (the confound resolver).** B1 peak 80% interval [0.40, 0.54], pt 0.46.
Decision rule, registered: B1 >= 0.52 -> E1's soup number reflected warm-start-
from-target INSTABILITY (a non-twin attacker exploits soup fine); B1 <= 0.48 -> soup
is GENUINELY hard to attack at this budget (E1's reading stands, strengthened);
0.48-0.52 -> inconclusive on the confound.

**E2-H3 (exploit diversity, secondary).** Pairwise argmax disagreement
(`nemesis.py disagree`: states drawn from each net's OWN games vs the target, k=4
openings) between the argmax-snapshot nets of A1/A2/A3/A4 vs v2.
Prediction: disagree(A2, A1) > disagree(A4, A1). Confidence 70%.

## Notes

- A2 scratch previously read as hopeless (100% losses) under the BROKEN harness; with
  mirrored exploration it gets a fair opening game. The prediction stays pessimistic
  because most of a scratch net's budget goes to learning the game at all.
- Budget: 12 new runs (~8.5 h GPU) + the disagreement measurement (~minutes).
