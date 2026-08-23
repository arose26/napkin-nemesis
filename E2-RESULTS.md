# E2 results — scored against SPEC-E2 (registered 2026-08-22, pre-run)

Same instrument as E1 (k=4 curves, 9x2048 confirmed peaks, CI ~+/-0.007).

## Bake-off on v2 (equal budget, 2,500 iters, 3 seeds)

| arm | bootstrap | init vs v2 | confirmed peak | gain over start |
|---|---|---|---|---|
| A4 | perturbed twin (sigma 0.05) | 0.445 | **0.518** | +0.073 |
| A1 | warm (= e1_v2, reused) | 0.500 | **0.514** | +0.014 |
| A3 | ckpt:v1 (ancestor) | 0.397 | **0.468** | +0.071 |
| A2 | scratch | ~0 | **0.322** | +0.32 |

A4's curve was still rising at the horizon (last point 0.535) — mild truncation noted.

## Confound resolver on soup

B1 (ckpt:v2 attacking soup): confirmed peak **0.378** (curve plateaus 0.36-0.37,
BELOW v2's raw 0.399 head-to-head vs soup). Registered decision rule: <= 0.48 =>
**soup is genuinely hard to attack at this budget.** Convergent evidence: the twin
attack (E1) and the strong non-twin attack (E2) plateau at the SAME value (0.379 vs
0.378). E1's caveat resolves: soup's number was robustness, not a bad-start artifact.

## Exploit diversity (E2-H3)

Pairwise argmax disagreement on exploit-path states (37,027 states):
A1-A2 0.449, A1-A3 0.376, A1-A4 0.329, A2-A3 0.449, A2-A4 0.445, A3-A4 0.372.
Different bootstraps find measurably different policies.

## Hypothesis scores

- **E2-H1** exact order (A1 >= A4 > A3 > A2, 30%): **miss on point estimates** — A4
  nominally edges A1 (0.518 vs 0.514), though the gap is inside the confirmation CIs
  (statistical tie). A3 third and A2 last as predicted; "A2 last" (65%): **HIT**.
  Intervals: A4 0.518 in [0.46,0.55] **hit**; A3 0.468 in [0.40,0.56] **hit**;
  A2 0.322 in [0.30,0.48] **hit**. 3/3 after E1's 2/4 — the recalibration held.
- **E2-H2** (B1 interval [0.40,0.54] pt 0.46): observed 0.378 — interval **miss**
  (again too optimistic about attacking soup), but the registered DECISION resolves
  cleanly: soup genuinely robust.
- **E2-H3** (disagree(A2,A1) > disagree(A4,A1), 70%): 0.449 > 0.329 — **HIT**.

## What E1+E2 establish together

1. Best-response attacks on competent targets gain little at this budget: +0.07 for
   good non-twin starts, +0.014 for the twin. The best "attacker" is mostly the best
   STARTING net. Scratch climbs furthest (+0.32) but from the floor.
2. Soup resists attack from every tried start; its E1 number is real robustness.
   Weight-averaging looks like a genuine robustness intervention, yet soup LOST ladder
   score — robustness and population performance are different axes (E1's reversal).
3. The nemeses are genuinely distinct policies (0.33-0.45 disagreement), which
   satisfies E3's pool-diversity precondition with measured evidence.
