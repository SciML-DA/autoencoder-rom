# sparse_sensors / linear_baseline

**From:** `qsub hpc/sensors.pbs` → `scripts/sparse_sensor_study.py` with
`--branches` empty, i.e. **POD-LSE and extended POD only, no networks**. CPU,
`--r-field 64`, `--delays 1 5 10 25 50 100`, `--ridge-cv`, `--force-lag 0`, on
run `4p5d_10ms_yaw_0_0_0`. Full arguments in `config.json`.

**Purpose:** the number every later claim is relative to. It is one SVD and one
small solve, so it takes minutes rather than hours — and if the PIV/force
pairing or the drift correction is wrong, it shows up here for the price of a
short CPU job rather than after an eight-hour GPU one. Run it before the GPU
job.
