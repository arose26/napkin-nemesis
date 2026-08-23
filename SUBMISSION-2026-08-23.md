# Ladder submission — e3_c1_s2 (user-directed, outside E3's registered gate)

**Status of this submission in the series:** E3's null gate fired and the registered
protocol said *no submission*. The user directed one anyway. This is a **user-directed
deviation**, recorded as such — the gate is not retroactively reinterpreted. Because a
submission is a scarce, irreversible act (it replaces the incumbent; leagues never
demote), the prediction below is registered BEFORE placement so the run still buys
information.

## What was submitted and why that net

Candidates were the six E3 finals. The two best were tied on the registered primary
metric (held-out panel: s1 0.4813, s2 0.4812) and tied head-to-head
(**0.497**, CI 0.476..0.519, 2048 games — spans 0.5). Tiebreak went to performance
against the net it must displace: **e3_c1_s2 beats v2 at 0.540** vs s1's 0.517
(6,144 games each, panel protocol).

Artifact: `napkin-100k-connect4/out/c4_bot_e3c1s2.cpp`, 98,359 bytes (1,641 under cap).
Verified before submitting: check-pack max |value drift| 1e-6, policy argmax 35/35;
check-bot 51 moves legal, 8/8 forced wins taken; sandbox TestSession/play compiled and
won 10-0. Agent id **41183978**.

## Same-day baseline (the number to compare against)

v2 incumbent, read 2026-08-23 17:51 UTC, immediately before submitting:
**score 31.21, rank 48** of 928. (Cross-day comparison is invalid — the same v2 bytes
read 32.24 on 08-20 and 30.93 on 08-22.)

## Registered prediction (before placement)

Point estimate **30.8**. 80% interval on the final settled score: **[30.0, 31.7]**.
P(beats the 31.21 baseline) = **30%**.

Reasoning, stated so it can be scored: s2's offline profile is the same shape that
anti-predicted the ladder twice — it beats the incumbent head-to-head (0.540) while
losing to v3peak (0.477) and badly to soup (0.405), and both of those nets scored
BELOW v2 on the ladder. Under the anti-correlation pattern E1-E3 measured, a net whose
offline strength sits between v2 and v3peak should land between their ladder scores,
i.e. modestly below today's v2 reading. Under the naive offline-transfer story it
should gain. The interval admits both, and 30% is a real chance of being wrong.

**This is the cleanest test of the series' central finding to date**: an out-of-sample
prediction, registered before the measurement, on the exact question the offline
instruments keep getting wrong.

## Rollback rule, pre-committed

If the settled score is below 31.21 by more than 0.2, roll back to v2
(`out/c4_bot_v2.cpp`, verified artifact) and record another same-day baseline. If it
lands above, v2's rank-34-era artifact is retired as the incumbent and s2 stands.
