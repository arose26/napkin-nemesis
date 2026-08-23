# napkin-nemesis

Series-3 experiment: what opponent distribution should a self-play agent train
against to be strong against a population it has never seen? Testbed: the 7x9
pie-rule Connect 4 arena (napkin-100k-connect4, rank 34/926 baseline), with UTTT
held out for replication.

Status: **planning — nothing has run.** See [PLAN.md](PLAN.md) for the full
design: the measured self-play-overfitting anomaly, the three-axis design space
(antagonist bootstrapping / framework / feedback), experiments E0–E4, and
decision gates. Each experiment preregisters its own hypotheses in this repo
immediately before it runs.

`napkin_c4.py` is a **vendored byte-copy** of the engine + instrument from
napkin-100k-connect4 @ 35e7171 (md5 20a134f0d520e2f2d535561bdc7f3c99). Do not
edit it here; the nemesis harness will be a separate file importing it. The
100KB packer and ladder submission path stay in the source repo.
