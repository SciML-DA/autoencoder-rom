# Autoencoder ROM

Dimensionality reduction for fluid snapshot data, and reduced-order models built
on top of it. The question the repository is set up to answer is how far a
nonlinear encoder beats a linear one at a fixed latent size, and what it costs.

Three reduction families share one `Projector` interface — POD/SPOD (linear),
dense and convolutional autoencoders in PyTorch, and the same two autoencoders
reimplemented from scratch in JAX. An Echo State Network provides the temporal
forecaster, so a POD or autoencoder latent space can be rolled forward in time.

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

**Reduced-order models** [`models/data_driven`](src/models/data_driven/)
* `ESN_model` — Echo State Network as a forecasting model.
* `POD_ESN` — POD reduction + ESN forecaster.

**ESN configuration** [`config`](src/config/)
* `ESNConfig`, `auto_load_or_create`, `find_matching_config` — content-hashed
  save/load so a trained ESN is not refitted when its hyperparameters match one
  already on disk.

---

## 📂 Structure

```
.
├── data/                        # Snapshot files (untracked)
├── docs/                        # ESN walkthroughs, JAX port plan
├── hpc/                         # Slurm/PBS job scripts and cluster setup
├── results/                     # Generated results (only results.csv tracked)
├── scripts/
│   └── convergence_study.py     # Latent-dimension sweep across all models
├── src/
│   ├── utils.py
│   ├── config/
│   │   └── esn_config.py        # ESNConfig, hashed save/load
│   ├── datasets/
│   │   ├── snapshots.py         # Readers, decimation, SPECS
│   │   └── splitting.py         # Gap-separated splits + diagnostics
│   ├── models/
│   │   ├── model.py             # Model base class
│   │   ├── history.py           # HistoryTracker mixin
│   │   ├── integrator.py        # IVPIntegrator, DiscreteIntegrator, ...
│   │   └── data_driven/
│   │       ├── esn.py           # ESN_model
│   │       └── pod_esn.py       # POD_ESN
│   ├── plotting/
│   │   └── pod.py               # Modes, coefficients, spectrum, RMS
│   └── tools/
│       ├── autoencoders.py      # Projector, POD, SPOD, AE, CAE  (PyTorch)
│       ├── autoencoders_jax.py  # AEJax, CAEJax                   (JAX)
│       ├── pod_spod.py          # POD/SPOD algorithms
│       ├── esn_core.py          # EchoStateNetwork reservoir
│       └── lstm_core.py         # LSTM forecaster
├── pyproject.toml
└── README.md
```

---

## 📝 Attribution

The model/integrator layer, the ESN reservoir (`tools/esn_core.py`) and parts of
`utils.py` originate from A. Nóvoa's
[real-time bias-aware DA](https://github.com/andreanovoa/real-time-bias-aware-DA)
repository, which is where the ESN and POD-ESN implementations come from. The
data-assimilation, bias-estimation and physical-model layers of that repository
are not carried here — see the original for those.
