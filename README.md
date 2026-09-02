# Autoencoder ROM

Dimensionality reduction for fluid snapshot data, and reduced-order models built
on top of it. The question the repository is set up to answer is how far a
nonlinear encoder beats a linear one at a fixed latent size, and what it costs.

Three reduction families share one `Projector` interface — POD/SPOD (linear),
dense and convolutional autoencoders in PyTorch, and the same two autoencoders
reimplemented from scratch in JAX. An Echo State Network provides the temporal
forecaster, so a POD or autoencoder latent space can be rolled forward in time.

On top of that sits **reconstruction from sparse sensors**: recovering a full
PIV velocity field from twelve load-cell channels, by extended POD and linear
stochastic estimation (the baseline) or by a two-branch autoencoder with a
learned sensor map (the thing that has to beat it). See
[docs/sparse_sensors/](docs/sparse_sensors/).

---

## 🚀 Getting started

1. **Clone and install**

```bash
git clone <this-repo> && cd autoencoder-rom
pip install -e . --use-pep517
```

The layout is `src`-based: `package-dir = {"" = "src"}` puts `tools`, `models`,
`datasets`, `plotting` and `config` on the path as top-level packages, so imports
read `from tools import POD`, not `from src.tools import POD`.

2. **Point it at data**

Snapshot files are resolved through `datasets.snapshots.data_path`, which reads
`$DATA_ROOT` and falls back to `./data`. Nothing in `data/` is tracked.

3. **Run the convergence study**

```bash
python scripts/convergence_study.py
```

This sweeps latent dimension across every model on one dataset and writes
`results/convergence/<dataset>/` — loss curves per latent size, summary plots,
and a `results.csv`. Only the CSV is tracked; the plots regenerate.

4. **Or the sparse-sensor study**

```bash
VERIFY_SLOW=1 pytest tests/ -q
```

66 checks with no data and no network, ending in end-to-end runs of every job
script's command against a synthetic RDS-shaped directory. Then, with
`RDS_ROOT` pointing at the real thing:

```bash
python scripts/sparse_sensor_study.py --quick     # one split, every method
python scripts/sparse_sensor_sweep.py  --quick    # the convergence study
```

The sweep is resumable and caches fitted autoencoders, so a cluster job killed
at walltime continues where it stopped. It also writes a `video_pack.npz` that
[make_reconstruction_video.py](scripts/make_reconstruction_video.py) turns into
truth/reconstruction/error footage — that script imports nothing from `src/`, so
it runs wherever you have ffmpeg rather than wherever you have torch.

---

## 🌟 What is available?

**Snapshot data** [`datasets`](src/datasets/)
* `load_snapshots` — reads `.h5` and `.mat` into the `(Nu, Nt, Nx, Ny)` layout
  every `Projector` expects, with NaN marking solid-body points. Decimation is
  antialiased (boxcar), because plain striding folds energy above the new
  Nyquist back in as noise.
* `SPECS` — resolution and file layout per dataset: `bl`, `circle`, `triangle`.
* `prepare_split`, `split_diagnostics`, `linear_span_ceiling` — gap-separated
  train/val/test splits plus the diagnostics that say whether a split is
  measuring generalisation or interpolation.

**Dimensionality reduction** [`tools`](src/tools/)
* `POD`, `SPOD` — snapshot POD, exact (Sirovich 1987) or randomized
  (Halko 2011), and spectral POD (Sieber 2016); `spod_towne` for the Welch CSD
  variant (Towne 2018).
* `AE`, `CAE` — dense and convolutional autoencoders, PyTorch.
* `AEJax`, `CAEJax` — the same two in JAX, with hand-written Adam, conv and
  transposed-conv kernels rather than a NN framework.
* `EchoStateNetwork` — reservoir building block.
* `LSTM` — single-layer LSTM trained by truncated BPTT, an alternative
  forecaster.

All projectors share one API:

```python
p.fit(X)          # learn from X (N_x, N_t)
p.encode(X)       # -> Z (N_latent, N_t)
p.decode(Z)       # -> X_hat (N_x, N_t)
p.reconstruct(X)  # round-trip
p.score(X)        # mean squared reconstruction error
```

**Reconstruction from sparse sensors** [`tools`](src/tools/)
* `PODLSE`, `ExtendedPOD` — POD + linear stochastic estimation, and Borée's
  extended POD. The same estimator by two routes; the second skips the field
  SVD and is what you sweep hyperparameters with.
* `delay_embed` — causal lag stacking. A linear map from twelve instantaneous
  channels reaches a twelve-dimensional subspace of the flow *whatever*
  `r_field` says; lags raise that ceiling, and they also absorb the fact that a
  load cell integrates the pressure field and so lags it.
* `mode_observability`, `projection_floor` — the two ceilings that say whether
  the basis, the sensors or the fit is the limit. Quote them next to every score.
* `BranchedAE` — the two-branch autoencoder: any `Projector` as the encoder /
  decoder plus a learned sensor branch (`linear` | `mlp` | `cnn` | `gru`),
  trained against a frozen decoder on the latent loss.
* `LatentForecaster` — a GRU on latent trajectories with a multi-step unrolled
  loss, for rolling a nowcast forward.

**Reduced-order models** [`models/data_driven`](src/models/data_driven/)
* `ESN_model` — Echo State Network as a forecasting model.
* `POD_ESN` — POD reduction + ESN forecaster.

**The experiment** [`datasets`](src/datasets/)
* `build_case` — one entry point for the April porous-disc experiment on RDS:
  snapshots, PIV/force pairing, baseline drift correction and masking all
  decided in one place, because getting any of the three subtly wrong produces a
  model that trains happily and reconstructs noise.

**ESN configuration** [`config`](src/config/)
* `ESNConfig`, `auto_load_or_create`, `find_matching_config` — content-hashed
  save/load so a trained ESN is not refitted when its hyperparameters match one
  already on disk.

---

## 📂 Structure

```
.
├── data/                        # Snapshot files (untracked)
├── docs/                        # ESN walkthroughs, sparse-sensor briefing
├── hpc/                         # PBS job scripts and cluster setup
│   ├── lib.sh                   # threads, live log, env -- sourced by every job
│   ├── preflight.sh             # run before every qsub
│   └── run_study.sh             # submits the study as a dependency chain
├── results/                     # Generated results (only results.csv tracked)
├── scripts/
│   ├── convergence_study.py     # Latent-dimension sweep across all models
│   ├── sparse_sensor_study.py   # Field-from-sensors: every method, one split
│   ├── sparse_sensor_sweep.py   # Convergence study: latent/delay/sensor sweeps
│   ├── make_reconstruction_video.py  # Renders footage from a video pack
│   ├── spectra.py               # FFT/coherence survey: what the sensors see
│   ├── diagnose_sensors.py      # Why the reconstruction plateaus
│   ├── check_threads.py         # What the job's BLAS actually got
│   └── (checks live in tests/, run with pytest)
│   └── inspect_experiment.py    # One-shot RDS survey -- run first on cx3
├── src/
│   ├── utils.py
│   ├── config/
│   │   └── esn_config.py        # ESNConfig, hashed save/load
│   ├── datasets/
│   │   ├── snapshots.py         # Readers, decimation, SPECS
│   │   ├── wake_experiment.py   # April experiment: readers, Case, build_case
│   │   └── splitting.py         # Gap-separated splits + diagnostics
│   ├── models/
│   │   ├── model.py             # Model base class
│   │   ├── history.py           # HistoryTracker mixin
│   │   ├── integrator.py        # IVPIntegrator, DiscreteIntegrator, ...
│   │   └── data_driven/
│   │       ├── esn.py           # ESN_model
│   │       └── pod_esn.py       # POD_ESN
│   ├── plotting/
│   │   ├── pod.py               # Modes, coefficients, spectrum, RMS
│   │   └── reconstruction.py    # Observability, extended modes, comparisons
│   └── tools/
│       ├── autoencoders.py      # Projector, POD, SPOD, AE, CAE  (PyTorch)
│       ├── autoencoders_jax.py  # AEJax, CAEJax                   (JAX)
│       ├── pod_spod.py          # POD/SPOD algorithms
│       ├── epod.py              # PODLSE, ExtendedPOD, delays, observability
│       ├── branched_ae.py       # BranchedAE, SensorBranch, LatentForecaster
│       ├── esn_core.py          # EchoStateNetwork reservoir
│       └── lstm_core.py         # LSTM forecaster
├── pyproject.toml
└── README.md
```

---

## 🖥 Running on the cluster

Every PBS job sources [hpc/lib.sh](hpc/lib.sh), which decides the BLAS thread
count, the live log and the environment in one place. The thread count is the
scheduler's allocation (`$NCPUS`) clamped by the affinity mask -- not the
affinity mask itself, because PBS Pro only cpuset-confines a job when the
cgroup hook is enabled, and where it does not `nproc` reports the whole node
for a job that asked for eight cores.

Check before you submit, not after:

```bash
./hpc/preflight.sh
```

It parses every job script, resolves the thread block by actually sourcing it,
extracts the python command each job builds and checks every flag against that
script's `--help`, and confirms the BLAS runs on the threads it was given. Add
`--full` to include the synthetic end-to-end runs.

Then submit the study as a chain rather than by hand:

```bash
./hpc/run_study.sh diagnose
```

Phase 1 is verify → spectra → diagnose. It measures the PIV/force offset two
independent ways; pass the result to phase 2 as `FORCE_LAG`, because
`delay_embed` is strictly causal and cannot reach an offset in the other
direction no matter how long the window.

```bash
FORCE_LAG=-25 ./hpc/run_study.sh sweep
```

---

## 📝 Attribution

The model/integrator layer, the ESN reservoir (`tools/esn_core.py`) and parts of
`utils.py` originate from A. Nóvoa's
[real-time bias-aware DA](https://github.com/andreanovoa/real-time-bias-aware-DA)
repository, which is where the ESN and POD-ESN implementations come from. The
data-assimilation, bias-estimation and physical-model layers of that repository
are not carried here — see the original for those.
