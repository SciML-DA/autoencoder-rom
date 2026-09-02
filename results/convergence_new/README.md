# convergence_new

**From:** `scripts/convergence_study.py`, same job as `../convergence/`, rerun
after the autoencoders were replaced by the optimised JAX implementations
(commit `e42c924`).

**Purpose:** confirm the rewrite changed the speed and not the answer. Same
dataset, same split, same latent sweep as `../convergence/bl/` — so the two
CSVs are row-for-row comparable. Test relative error agrees to ~3 decimals
throughout; `CAEJax` at latent 64 fits ~14x faster, `AEJax` ~9x.

Only the two summary figures were kept; the per-latent loss curves were not
regenerated.
