# napkin-nemesis — series plan

Context: napkin-100k series. Self-play NNs in ≤100KB CodinGame bots on live ladders, with
calibration-scored preregistered hypotheses. Measured anomaly motivating this series: two
connect-4 nets with real offline edges over the deployed v2 (0.537 and 0.594, both above
the ~0.03 noise floor) each scored WORSE on the live ladder (32.24 → 31.65 → 31.46).
Self-play trains against an opponent distribution of exactly one policy; the hypothesis is
that this produces nets specialized to beat their own lineage and fragile against a
926-bot population.

## The series question

What opponent distribution should the main agent train against to be strong against a
population it has never seen — and what is the cheapest antagonist setup that beats pure
self-play on ladder translation?

## Prior art and the actual novel claim

This is well-trodden ground: fictitious play, double oracle / PSRO (Lanctot et al. 2017),
AlphaStar's league with main/league exploiters (Vinyals et al. 2019), exploitability as
NashConv. Our angle is NOT a new algorithm. It is:
1. Ground truth: population-training papers score themselves by offline exploitability;
   we have a live ladder of ~926 independently-authored bots as an out-of-distribution
   test set, and a measured case where offline improvement anti-predicted ladder outcome.
2. Regime: single 6GB GPU, whole training runs ~1 GPU-hour. Which population-training
   trade-offs survive at napkin scale is genuinely open — leagues were built on
   thousands of TPUs.
3. Method: calibration-scored preregistration per experiment.

## Testbed

Connect-4 (7×9 + steal arena) primary:
- fastest engine we have (~840 games/s full GPU loop), 10-action space
- the anomaly was measured here, so a fix is testable here
- rank-34 baseline in the arena's top league; score-based tracking; submissions resolve
  in ~2h
UTTT is the held-out replication testbed for whatever wins (guards against overfitting
conclusions to one game — the c4→UTTT hyperparameter transfer already taught us recipes
don't port blindly; target entropy differs by 17×). A brand-new game is out of scope:
venue-parity verification is a real cost we don't need to respend.

## Repo

New repo `napkin-nemesis`, distinct from both game repos. Vendors the c4
engine/instrument from napkin-100k-connect4 at a pinned commit (byte-copy with provenance
note) so results stay reproducible if the source repo moves. The 100k packer and CDP
submission path stay in the c4 repo: nemesis produces checkpoints, the c4 repo packs and
submits them.

## Design space (what "antagonist setup" decomposes into)

Three orthogonal axes:

**A — antagonist bootstrapping** (the "direct copy or something else" question):
- A1 warm start from the target (direct copy)
- A2 from scratch (random init)
- A3 from a past checkpoint of the target's lineage
- A4 perturbed copy (target weights + noise) — cheapest possible diversity

**B — antagonist framework:**
- B1 best response to a frozen target (asymmetric, stable)
- B2 alternating best response (target and nemesis take turns; cycling risk)
- B3 simultaneous adversarial (GAN-style; cycling in zero-sum simultaneous training is a
  documented failure mode — include in discussion, likely cut, napkin budget won't fix it)

**C — feedback into the main agent:**
- C1 none (nemesis as pure measurement instrument)
- C2 pool training: opponent sampled uniformly per game from {current self, past selves,
  nemeses}
- C3 PFSP: same pool, opponents sampled proportional to how often they beat the learner
- C4 data-only: nemesis games added to the replay buffer, self-play opponent unchanged

## Experiments

Each experiment gets its own preregistered spec IN THE REPO, written immediately before
it runs — not all now, because later experiments' predictions depend on earlier results
(that's a feature of the design, not an escape hatch; hypothesis SHAPES are fixed here,
numbers and intervals registered at start-of-experiment).

### E0 — instrument + debt (engineering gate, no hypotheses)
- z-sign assert for opponent mode. The prior attempt's curve read 0.008 against its own
  warm-start target — impossible, must start ~0.500 — and the suspected cause is a value
  sign flip that selfcheck doesn't cover. The series does not start until a warm-started
  nemesis reads ~0.500 at iter 0 THROUGH the training harness.
- Null controls through the harness: target-vs-itself ≈ 0.500.
- Positive control: the nemesis procedure vs a scripted deterministic minimax-d2 target
  must climb far above 0.500 — proof the procedure can find exploits at all (this makes
  the old N5 validity check concrete). Note: exploitability of a deterministic policy is
  measured over the paired-random-opening distribution with distinct-position counts,
  which is what makes the number meaningful rather than a single repeated trap line.
- Exploitability summary command: exploit(target, bootstrap, budget) → 3-seed mean at
  argmax snapshot AND mean over last 5 snapshots, never max-over-snapshots.
- Ladder housekeeping: the owed v2 rollback (pre-committed when the soup failed), so the
  ladder baseline reflects our best known net before this series reads it.
  **Done 2026-08-22**: resubmitted the archived rank-34 artifact `c4_bot_v2.cpp`
  (verified against gpunet_v2.pt: check-pack value drift 1e-6, policy argmax 32/32;
  sandbox TestSession/play passed), agent id 41177999. Post-placement snapshots at
  15:25 and 16:56 UTC read identically: score 30.93, rank 49 — settled. The SAME bytes
  scored 32.24 two days earlier, so the field drifts ~1.3 points/2 days and historical
  ladder scores are not comparable across days. Consequences: (1) E3's ladder outcome
  is scored against a same-day reading of this standing v2 baseline; (2) whether v2
  still beats soup in TODAY'S field is unknown and not worth a submission to find out;
  the pre-committed rollback stands and v2 remains the incumbent until E3's winner.

### E1 — is exploitability real, and does it explain our anomaly? (measurement)
Measure exploit() for the c4 lineage: v1, v2, v3peak, soup, + minimax-d2 anchor.
Hypothesis shapes to register:
- magnitude: every self-play net is exploitable ≥ X at budget B (X, B registered then)
- ordering: predicted rank order of exploitability across the four nets, registered
  before measurement. Honest framing: four points cannot establish a correlation; what
  E1 can do is score a registered ordering prediction and tell us whether
  exploitability(net) even MOVES in the direction that would explain the ladder anomaly.
- construct caveat, stated in the spec: exploitability vs one trained adversary is a
  WORST-CASE measure; the ladder anomaly is about robustness to a diverse, mostly
  suboptimal population. These are different constructs and E1 does not conflate them —
  it asks whether the cheap worst-case number happens to track the lineage's ladder
  ordering. To give it competition, E1 also scores each lineage net against the fixed
  yardstick panel (a mini-population proxy, eval-only, cheap) so we learn which offline
  measure — panel score or exploitability — better predicts ladder order. The population
  construct itself is only tested directly in E3's ladder submission.
- disclosure: the prior broken run emitted four garbage points (0.008–0.020) before
  being killed; treated as uninformative (they were a harness bug, not a measurement)
  but disclosed in the spec.

### E2 — bootstrapping bake-off (which antagonist finds exploits cheapest)
Arms A1–A4, equal game budget, same frozen target (v2), 3 seeds.
Primary outcome: exploit-at-budget curve; peak per E0 protocol.
Secondary outcome: do different bootstraps find DIFFERENT exploits? Measured as action
disagreement between the resulting nemeses on states drawn from the NEMESES' OWN
nemesis-vs-target game trajectories (pooled across arms) — the states where exploits
actually manifest. States from ordinary target-vs-target games are the wrong
distribution: exploits by definition steer play off it.
Prior evidence to disclose: a from-scratch nemesis previously hit z=-1 on 100% of 16,384
plies (zero learning signal), so "warm start dominates early" is a weakly-held prior,
not a clean prediction.

### E3 — feedback frameworks (the payoff)
Arms C1 (pure self-play continuation — the control), C2 uniform pool, C3 PFSP, C4
data-only. Equal compute, all starting from v2. Pool composition fixed in the spec:
{v2, v1, best E2 nemeses, k past snapshots}, with ≥4 members verified pairwise-distinct
(pairwise scores away from 0.500).
Outcomes in order of authority:
1. offline vs a FIXED yardstick panel (minimax-d2, minimax-d4 if affordable, v1, v2) —
   a panel disjoint from the training pool, so the metric isn't the objective
2. exploitability of each final net (E0 instrument)
3. ONE ladder submission, for the arm that wins on (1)+(2) — submissions replace the
   incumbent, so the winner is chosen offline first and the translation prediction is
   registered before submitting.
Hypothesis shapes: pool arms beat C1 on the panel; pool arms are less exploitable than
C1; ladder delta within a registered interval. The interesting fight is C3 vs C2
(does prioritized sampling matter at pool sizes this small?).

### E4 — replication on UTTT (conditional on E3 finding a winner)
Port the winning recipe; register the transfer prediction with the target-entropy lesson
in mind (last recipe transfer measured ~0.000 across these two games).

## Decision gates
- E1 finds exploitability ≤ noise floor for all nets → the self-play-overfitting story
  is wrong or unmeasurable at this budget → series pivots rather than proceeds (the
  leading alternative: the population plays strategies absent from self-play
  distributions entirely — that's a different experiment, napkin-mirror).
- E2's winner determines the nemeses that populate E3's pools.
- E3: C1 beating all pool arms is the registered null; the old N5 validity checks
  (null control passed, procedure can climb, pool genuinely diverse, effect above noise
  floor) are preconditions for interpreting it as "diversity doesn't help" rather than
  "test inconclusive".

## Budget
- E0: ~half a day engineering, minutes of GPU.
- E1: 5 targets × 3 seeds × ~35 min ≈ 9 GPU-h.
- E2: 4 arms × 3 seeds × ~35 min ≈ 7 GPU-h.
- E3: 4 arms × ~2 h: screen with 1 seed, confirm top-2 with 3 seeds ≈ ~20 GPU-h, or rent
  parallel GPUs and finish in an afternoon (standing rule: rent freely).
- Ladder: 2 submissions total (E0 rollback + E3 winner), ~2 h placement each.

## Explicitly not doing
- Full AlphaStar league (three concurrent populations): doesn't fit the budget, and the
  informative ablations ARE E2/E3.
- Simultaneous adversarial training (B3): known cycling failure; alternating frozen
  phases capture the idea with stability.
- Full PSRO meta-solvers: at pool sizes ≤8, uniform vs PFSP already spans the
  interesting sampling policies; a Nash meta-solver is a follow-up only if C2 vs C3
  shows sampling policy matters.

## Decisions locked (2026-08-22)
1. Connect-4 confirmed as primary testbed.
2. Engine vendored: `napkin_c4.py` is a byte-copy from napkin-100k-connect4 @ 162110e
   (md5 f73f518dfc75e8019e64262d7de2fd5b). Do not edit the vendored file; the nemesis
   harness lives in its own file and imports it.
3. v2 ladder rollback approved as part of E0.

Still open (decide before E3): serial on the local GPU (~3 days wall) vs renting
parallel GPUs (hours).

## Provenance
A partial sketch of this experiment (NEMESIS.md in the napkin-100k-connect4 repo)
predates this plan; it was written after a premature, bug-broken run. Its protocol
lessons and disclosures are folded in above; its hypotheses are superseded — each
experiment here registers fresh ones at start-of-experiment.
