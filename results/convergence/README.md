# convergence

**From:** `qsub experiments/bl/hpc/convergence.pbs` (cx3/PBS) or `sbatch experiments/bl/hpc/convergence.slr`
(Aero/Slurm) → `experiments/bl/scripts/convergence_study.py`, one subfolder per dataset tag.

**Purpose:** the latent-dimension convergence study on clean *simulation*
snapshots. Every reduction family — POD, dense AE, conv CAE, in both torch and
JAX — at latent 2, 4, 8 … 256, so the curves are directly comparable. It answers
how far a nonlinear encoder beats a linear one at fixed latent size, and what
that costs in parameters and fit time.

Not to be confused with the sparse-sensor convergence study in `../sparse_sweep/`,
which shares nothing with this but the word.

| subfolder | dataset | split | status |
|---|---|---|---|
| `bl/` | boundary layer (Challenge1.1) | 1798/600/600, gap 1 | current |
| `bl_old_block/` | boundary layer | 598/200/200, gap 1 | superseded — smaller split, before the split diagnostics were added |
| `circle_old/` | flow past a cylinder | 464/160/160, gap 8 | superseded |
| `circle_old_leaky/` | flow past a cylinder | 481/159/160, **gap 0** | kept as the counter-example — see below |
| `circle/` | — | — | empty; the cylinder rerun on the current pipeline was never done |

`circle_old_leaky` was run before the decorrelation gap existed. With gap 0 the
held-out block starts on the frame after the training block ends, and
consecutive frames are near-identical, so the test set is effectively a copy of
the training set. Compare it against `circle_old` (same study, gap 8) to see how
much a leaky split flatters the same models.
