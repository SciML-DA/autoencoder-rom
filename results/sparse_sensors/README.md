# sparse_sensors

**From:** `scripts/sparse_sensor_study.py`, one subfolder per `--tag`. Three
jobs write here:

| job | tag | what it runs |
|---|---|---|
| `hpc/sensors.pbs` | `linear_baseline` | linear baselines only, CPU, minutes |
| `hpc/sparse_sensors.pbs` | the run name | the two-branch autoencoders, GPU, hours |
| `hpc/sparse_sensors_crossyaw.pbs` | `xyaw_<run>` | held-out-yaw array job, one index per run |

**Purpose:** reconstruct the PIV velocity field around disc 2 from the twelve
load-cell channels alone, with every method in the repo, on one contiguous
held-out block, at matched latent dimension. The grid is the 2x2 of {POD, AE/CAE}
encoder x {linear, MLP/GRU} sensor map, crossed with the delay-embedding length,
so an improvement can be *attributed* rather than just observed — a win the
nonlinear sensor map already had means the manifold was never the limitation.

Every score is quoted against two lines: 1.0 (predicting the temporal mean) and
the projection floor (the best anything could do inside the given field basis).

Only `linear_baseline/` has been run so far; the GPU and cross-yaw tags are not
present yet.
