# sparse_sweep

**From:** `experiments/april_wake/scripts/sparse_sensor_sweep.py`, one subfolder per `--tag`, submitted
by `experiments/april_wake/hpc/sparse_sensor_sweep.pbs` (the sweep proper) and `experiments/april_wake/hpc/lowrank.pbs`
(stage R). Both are chained by `./experiments/april_wake/hpc/run_study.sh phase2`.

**Purpose:** the convergence study for sparse-sensor reconstruction — the
analogue of `../convergence/` for this task. It sweeps one axis at a time
against a fixed baseline rather than taking the full cross product, which would
be thousands of fits in which every curve is confounded with every other axis.

| stage | axis | question |
|---|---|---|
| A | latent, `r_field` 4…128 | the convergence curve |
| B | delays, `n_delays` 1…100 | where history stops paying |
| C | sensors, channel ablation | how few channels you need |
| D | seeds, best configs repeated | is the gap real or noise |
| R | rank x `r_sensor` x ridge | tuning the *linear* estimator at low rank |

The job is resumable: the CSV is appended row by row and read back on startup,
and fitted autoencoders are cached to `$EPHEMERAL`, so a link killed at walltime
is picked up by the next one.

| subfolder | run |
|---|---|
| `main/` | stages A–D, GPU — the sweep itself |
| `lowrank/` | stage R, CPU — the low-rank linear follow-up |
