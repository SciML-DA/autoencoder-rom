# sparse_sweep / main

**From:** `qsub experiments/april_wake/hpc/sparse_sensor_sweep.pbs` (defaults) → stages **A B C D** on
run `4p5d_10ms_yaw_0_0_0`, `--latents pod ae`, `--branches linear mlp cnn gru`,
latents 4…128, delays 1…100, 5 seeds, on GPU at `FORCE_LAG=0`. Full arguments in
`config.json`.

The main convergence result for sparse-sensor reconstruction — and the run whose
flat NMSE ~0.83 across every axis prompted `../../diagnosis/`, `../../spectra/`
and `../lowrank/`. Read it together with those; taken alone it looks like a
model problem rather than an information ceiling.

Note `FORCE_LAG=0`: if phase 1 found a non-zero PIV/force offset, these numbers
measure a misalignment. See `experiments/april_wake/hpc/run_study.sh`.
