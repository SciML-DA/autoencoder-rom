# results

One folder per study. Each has its own README saying which job produced it and
what question it was asked. Only `results.csv` and these READMEs are tracked by
git (plus the `performance/` figures) — everything else regenerates.

| folder | produced by | question |
|---|---|---|
| `convergence/` | `hpc/convergence.pbs` → `scripts/convergence_study.py` | How much does a nonlinear encoder beat POD at fixed latent size, on clean simulation data? |
| `convergence_new/` | same, after the JAX rewrite | Do the optimised JAX autoencoders reproduce the torch curves, faster? |
| `sweeps/` | `perf/sweep.py` | What does each training hyperparameter actually do to the JAX autoencoders? |
| `performance/` | `perf/baseline.slr`, `perf/regress.py` | Reference timings and golden loss curves for the unoptimised models. |
| `spectra/` | `hpc/spectra.pbs` → `scripts/spectra.py` | At which frequencies are the load cells linearly related to the flow? |
| `diagnosis/` | `hpc/diagnose.pbs` → `scripts/diagnose_sensors.py` | Why does sparse-sensor reconstruction plateau at NMSE ~0.83? |
| `sparse_sensors/` | `hpc/sensors.pbs`, `hpc/sparse_sensors.pbs` | Reconstruct the PIV field from twelve load cells — every method, one held-out block. |
| `sparse_sweep/` | `hpc/sparse_sensor_sweep.pbs`, `hpc/lowrank.pbs` | The convergence study for that reconstruction: one axis at a time. |

The sparse-sensor folders are ordered: `spectra` and `diagnosis` are phase 1 of
`hpc/run_study.sh` and decide what phase 2 (`sparse_sweep`) is allowed to claim.
