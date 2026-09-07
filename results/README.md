# results

One folder per study. Each has its own README saying which job produced it and
what question it was asked. Only `results.csv` and these READMEs are tracked by
git (plus the `performance/` figures) — everything else regenerates.

| folder | produced by | question |
|---|---|---|
| `convergence/` | `experiments/bl/hpc/convergence.pbs` → `experiments/bl/scripts/convergence_study.py` | How much does a nonlinear encoder beat POD at fixed latent size, on clean simulation data? |
| `convergence_new/` | same, after the JAX rewrite | Do the optimised JAX autoencoders reproduce the torch curves, faster? |
| `sweeps/` | `perf/sweep.py` | What does each training hyperparameter actually do to the JAX autoencoders? |
| `performance/` | `perf/baseline.slr`, `perf/regress.py` | Reference timings and golden loss curves for the unoptimised models. |
| `spectra/` | `experiments/april_wake/hpc/spectra.pbs` → `experiments/april_wake/scripts/spectra.py` | At which frequencies are the load cells linearly related to the flow? |
| `diagnosis/` | `experiments/april_wake/hpc/diagnose.pbs` → `experiments/april_wake/scripts/diagnose_sensors.py` | Why does sparse-sensor reconstruction plateau at NMSE ~0.83? |
| `sparse_sensors/` | `experiments/april_wake/hpc/sensors.pbs`, `experiments/april_wake/hpc/sparse_sensors.pbs` | Reconstruct the PIV field from twelve load cells — every method, one held-out block. |
| `sparse_sweep/` | `experiments/april_wake/hpc/sparse_sensor_sweep.pbs`, `experiments/april_wake/hpc/lowrank.pbs` | The convergence study for that reconstruction: one axis at a time. |

The sparse-sensor folders are ordered: `spectra` and `diagnosis` are phase 1 of
`experiments/april_wake/hpc/run_study.sh` and decide what phase 2 (`sparse_sweep`) is allowed to claim.
