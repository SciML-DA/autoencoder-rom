# sparse_sweep / lowrank

**From:** `qsub hpc/lowrank.pbs` → `scripts/sparse_sensor_sweep.py --stages R`,
CPU only, `--latents pod`, `--branches linear mlp gru`, on run
`4p5d_10ms_yaw_0_0_0`. Sweeps field rank 2…8 against `r_sensor` (12…200) and
ridge (1e-6…0.1). Full arguments in `config.json`.

**Purpose:** tune the linear estimator at the ranks the sensors actually
resolve. `../../diagnosis/` put the linear ceiling at NMSE 0.757 against 0.828
achieved, with only 3 of 64 field modes observable, and stage A in `../main/`
swept `r_field` up to 128 where everything above ~4 is fitting noise. So this
sweeps the bottom of that range properly. Every fit is a closed-form solve on a
rank <= 8 basis, hence CPU.

No `--notch` here, deliberately — see `../../diagnosis/notch_rejected.log`.
