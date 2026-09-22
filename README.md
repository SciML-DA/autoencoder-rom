# autoencoder-rom

A toolkit for **reducing fluid snapshot data to a low-dimensional latent space**,
**forecasting that latent space in time**, and **estimating a full field from a
handful of sensors**.

Everything is built around two small interfaces. A `Projector` reduces snapshots
and reconstructs them (`fit` / `encode` / `decode` / `score`); POD, SPOD and four
autoencoders all implement it interchangeably. A field estimator maps a sensor
record to a field (`fit` / `predict` / `score`); the closed-form linear estimator
and the neural one implement the same calls. 
| you want to | use | page |
|---|---|---|
| read snapshots into a standard array layout | `datasets.load_snapshots` | [Loading data](#loading-data) |
| split a time series without leaking between blocks | `datasets.prepare_split` | [Splitting](#splitting-data) |
| reduce a field to `r` coefficients | `POD`, `SPOD`, `AE`, `CAE`, `AEJax`, `CAEJax` | [Projectors](#projectors) |
| forecast those coefficients | `ESN_model`, `LSTM`, `LSTMJax` | [Forecasters](#forecasters) |
| do both as one object | `POD_ESN`, `AE_LSTM`, … | [ROMs](#roms) |
| reconstruct a field from sensors, in closed form | `PODLSE` | [Linear estimation](#linear-estimation-podlse) |
| …with a learned sensor map | `BranchedAE` | [Neural estimation](#neural-estimation-branchedae) |
| know whether sensors, basis or fit is the limit | `mode_observability`, `projection_floor` | [Diagnostics](#diagnosing-a-result) |
| avoid refitting a model you already trained | `config.auto_load_or_create` | [Saving models](#saving-and-reloading-models) |

---

## Install

```bash
git clone <this-repo> && cd autoencoder-rom
pip install -e . --use-pep517
```

Python ≥ 3.12. The layout is `src`-based with `package-dir = {"" = "src"}`, so
the packages are top level — `from datasets import ...`, not
`from src.datasets import ...`.

Torch and JAX are both installed. The JAX pin is platform-split on purpose:
torch's Linux wheel bundles CUDA and plain `jax` does not, so an unqualified
`jax` silently gives you a CPU build and any torch-vs-JAX timing becomes
meaningless. Read the comments in [pyproject.toml](pyproject.toml) before
changing those lines.

### Pointing at data

Nothing under `data/` is tracked.

| variable | read by | default |
|---|---|---|
| `DATA_ROOT` | `datasets.snapshots.data_path` | `./data` |
| `RDS_ROOT` | `experiments.april_wake.case_reader` | the campaign path in `case_reader.py` |
| `<NAME>_FILE` **and** `<NAME>_DOWNSAMPLE` | `datasets.SPECS`, at import — point a dataset at a pre-decimated copy | the values in `snapshots.py` |

The `SPECS` overrides must be set as a pair (`BL_FILE=sub.h5 BL_DOWNSAMPLE=4,4`),
because the copy is already decimated and reusing the original factor would
silently change the resolution. They are read at import, so export them before
starting the process; a malformed value fails `import datasets`.

### Checking the install

```bash
pytest tests/ -q
```

140 checks, no data files and no network: POD against analytically known
answers, finite-difference gradient checks for the JAX kernels,
the force-lag estimator, config hashing and the split diagnostics.

A further 25 are skipped by default because they train models or drive whole
scripts. `VERIFY_SLOW=1 pytest tests/ -q` runs those too, including each study
script end to end against a synthetic data directory written into a temp dir.

---

## Array conventions

Three layouts, used consistently throughout. Getting these wrong is the most
common source of confusion, so they are worth reading once.

| name | shape | meaning |
|---|---|---|
| **grid** | `(Nu, Nt, Nx, Ny)` | what loaders return. `Nu` velocity components, NaN at solid points |
| **snapshot** | `(Nu, Nx, Ny)` | a single grid snapshot |
| **flat** | `(N_x, N_t)` | one snapshot per **column**, fluid points only, `N_x = Nu * N_fluid` |
| sensors | `(N_s, N_t)` | one sample per column, aligned with the flat field |

The flat layout groups rows by component: row `u * N_fluid + f` holds component
`u` at the `f`-th fluid point. Every `Projector` accepts grid, snapshot or flat
input and converts internally; `field_estimation` works in flat and sensor
layouts only.

Latent coefficients are `(N_latent, N_t)` — also one snapshot per column.

---

## Loading data

```python
from datasets import SPECS, load_snapshots

X = load_snapshots(SPECS["bl"])      # (Nu, Nt, Nx, Ny), float32, NaN at solid points
```

`SPECS` ships three datasets — `bl` (three-component boundary-layer DNS,
`.h5`), `circle` and `triangle` (two-component wakes, `.mat`) — each a
`SnapshotSpec` giving the filename, the field names, and the resolution to read
at. 
```python
spec = SPECS["circle"].replace(downsample=(4, 4), max_snapshots=1000)
X = load_snapshots(spec)
```

| `SnapshotSpec` field | default | effect |
|---|---|---|
| `filename` | — | resolved against `$DATA_ROOT` by `data_path` |
| `fields` | — | variable names, one per component; their order fixes the leading axis |
| `downsample` | `(1, 1)` | spatial decimation as `(x, y)`, in *your* axis order — the reader maps it onto the file's native order |
| `stride` | `1` | take every `stride`-th snapshot |
| `max_snapshots` | `None` | cap on snapshots read |
| `antialias` | `True` | boxcar-mean decimation instead of plain striding |
| `chunk` | `200` | snapshots decimated at a time by the streaming HDF5 reader |

**Adding a format.** Write a function matching the `Reader` protocol, returning
`(Nt, Nx, Ny)` per field in the file's own axis order, and add it to `READERS`
under its extension. `load_snapshots` only stacks; the reader owns the transpose.

## Splitting data



```python
from datasets import prepare_split, LEAK_THRESHOLD

X_train, X_val, X_test, meta = prepare_split(X, val_frac=0.2, test_frac=0.2)
print(meta["gap"], meta["span_ceiling"], meta["test_nn_dist_median"])

if meta["test_nn_dist_median"] < LEAK_THRESHOLD:
    print("the test block repeats the training set — the split is not measuring generalisation")
```

The layout is `[warmup]--[train]--gap--[val]--gap--[test]`, with `val_frac=0`
collapsing to a single gap and no validation block.

The pieces are usable separately:

* `decorrelation_lag(X, max_lag=200, thresh=1/e)` — how many samples apart two
  snapshots become independent. This is the gap width.
* `split_indices(n_t, val_frac, test_frac, gap, warmup)` — index arrays only, for
  when you have your own data matrices. `warmup` discards leading samples
* `split_diagnostics(X_a, X_b)` — nearest-neighbour distance and distribution
  shift between two blocks
* `linear_span_ceiling(X_train, X_test)` — the fraction of test variance lying
  outside the training block's column space

---

## Projectors

Every projector implements the same five calls, so they are drop-in
interchangeable:

```python
p.fit(X)          # grid (Nu, Nt, Nx, Ny) or flat (N_x, N_t)
p.encode(X)       # -> (N_latent, N_t)
p.decode(Z)       # -> (N_x, N_t)
p.reconstruct(X)  # round trip
p.score(X)        # mean squared error over fluid points
```

Plus, on all of them: `decode_at(Z, idx)` to decode only selected rows (sensor
locations, say), `spatial_basis()`, and `n_params` (a property, not a call).

```python
from models.data_driven.autoencoders import POD, SPOD, AE, CAE, AEJax, CAEJax

pod = POD(n_modes=20).fit(X_train)
ae  = AE(n_latent=20, hidden=(512, 128), epochs=500).fit(X_train)

pod.score(X_test), ae.score(X_test)       # same metric, same data, comparable
```


### Autoencoders

Four total, two architectures in two frameworks:

| | PyTorch | JAX |
|---|---|---|
| dense | `AE(hidden=(512, 128))` | `AEJax(hidden=(512, 128))` |
| convolutional | `CAE(channels=(16, 32, 64), kernel_size=3, stride=2, pad=1)` | `CAEJax(...)` |


Training options are shared by all four (from `Autoencoder`):

| option | default | |
|---|---|---|
| `n_latent` | `10` | latent dimension |
| `learning_rate` | `1e-3` | initial Adam step |
| `epochs` | `500` | maximum epochs |
| `batch_size` | `32` | |
| `val_fraction` | `0.2` | held out **from the end** of the record for early stopping; `0` trains on everything and disables it |
| `weight_decay` | `0.0` | L2 penalty |
| `patience` | `50` | epochs without improvement before stopping |
| `threshold` | `1e-4` | relative improvement that counts as progress |
| `lr_factor`, `lr_patience`, `min_lr` | `0.5`, `10`, `1e-6` | plateau LR schedule |
| `seed` | `0` | |

`fit` scales each component by its standard deviation, trains on masked MSE,
decays the learning rate on a plateau, stops early, and restores the parameters
from the best validation epoch. Solid points enter the convolutional models as
zeros and are excluded from the loss. Per-epoch train/validation loss and
learning rate are on `.training_history`.


### Lazy imports

Importing `models.data_driven` or `models.data_driven.autoencoders` pulls in
**neither torch nor JAX**. Each projector, ROM and `LSTMJax` imports its own
module on first attribute access, so a POD or linear-estimator script never pays
for a framework it does not use.

---

## Forecasters

```python
from models.data_driven import ESN_model, LSTM, phi_to_esn_layout
```

* `ESN_model` — an echo state network wrapped in the `dynamodels.Model`
  interface (state history, discrete integrator, observation operator). The
  state is `[u; r]`: physical outputs plus reservoir. It trains at construction.
* `LSTM` / `LSTM_model` — a single-layer LSTM with forward and
  backward passes, trained by truncated BPTT over windows of `seq_len` after an
  `N_wash`-step washout, keeping the epoch with the lowest **closed-loop**
  validation error. Key options: `N_units`, `N_wash=100`, `seq_len=200`,
  `epochs=40`, `lr=3e-3`, `clip=5.0`, `val_frac=0.1`, `norm_method="range"`,
  `recurrent_init="orthogonal"`. `openLoop` and `closedLoop` run it.
* `LSTMJax` — the same model trained with JAX.

`ESN_model` wants `(L, N_t, N_latent)` segments, while projectors emit
`(N_latent, N_t)`. `phi_to_esn_layout` converts:

```python
Z = pod.encode(X_train)                 # (N_latent, N_t)
esn = ESN_model(data=phi_to_esn_layout(Z), dt=0.01, N_units=200)
```

## ROMs

A ROM is a projector and a forecaster in one object. All ten combinations exist:

| | `ESN` | `LSTM` |
|---|---|---|
| `POD` | `POD_ESN` | `POD_LSTM` |
| `AE` | `AE_ESN` | `AE_LSTM` |
| `CAE` | `CAE_ESN` | `CAE_LSTM` |
| `AEJax` | `AEJax_ESN` | `AEJax_LSTM` |
| `CAEJax` | `CAEJax_ESN` | `CAEJax_LSTM` |

```python
from models.data_driven import POD_ESN

rom = POD_ESN(data=X, dt=0.01, n_modes=10, Nq=8)   # fits POD, trains the ESN, places sensors
psi, t = rom.time_integrate(Nt=200)
rom.update_history(psi, t)
y = rom.get_observables(Nt=200)                     # sensor readings off the forecast
```

Each class is `class AE_ESN(LatentROM, ESN_model, AE)` — constructor options
from the projector and the forecaster both apply, so
`CAE_ESN(data=X, dt=0.01, n_latent=8, channels=(16, 32), Nq=8, N_units=100)`
passes `channels` to the CAE and `N_units` to the ESN.

`LatentROM` adds the sensor layer: `Nq` sensor points placed by QR pivoting
(`qr_selection=True`) or at random, restricted to `domain_of_measurement` and
thinned by `down_sample_measurement`; `measure_modes=True` observes the latent
coefficients instead. `define_sensors`, `reset_case` and `refit_projector`
re-place sensors or refit one half without rebuilding the object.

`Nq` changes meaning at construction: you pass
the number of sensor *points*, and afterwards it holds the number of
*observables*, which is one per point per component — so `Nq=8` on a
two-component field reports `Nq == 16` once built, with `N_sensors == 8`.
And locations are raw grid indices `u * Nx * Ny + g`, so `sensor_rows` is what
maps them onto rows of the flat field; use it rather than indexing the flat
field directly.

---

## Field estimation from sparse sensors

Given a field `Q` `(N_x, N_t)` and a synchronised sensor record `S` `(N_s, N_t)`,
estimate `Q` from `S` alone. Two families, one interface.

### Linear estimation: `PODLSE`

POD of the field, POD of the sensors, and a ridge map between the two sets of
coefficients:

```python
from datasets import split_indices
from field_estimation import PODLSE, delay_embed, nmse

Sd = delay_embed(S, n_delays=25)                     # causal lag stacking
tr, _, te = split_indices(Q.shape[1], val_frac=0, test_frac=0.25,
                          gap=100, warmup=24)        # warmup drops the padded columns

model = PODLSE(r_field=64, r_sensor=60, ridge=1e-3).fit(Q[:, tr], Sd[:, tr])
model.score(Q[:, te], Sd[:, te])                     # NMSE
Q_hat = model.predict(Sd[:, te])
```

| option | default | |
|---|---|---|
| `r_field` | `None` | field modes kept. `None` skips the field POD entirely and computes **extended POD** (Borée 2003), which gives the same estimate|
| `r_sensor` | `None` | sensor directions kept; `None` keeps all `N_s` |
| `sensor_basis` | `"pod"` | `"pod"` ranks sensor directions by variance, `"pls"` by cross-covariance with the field (SVD of `B @ C.T`) |
| `ridge` | `0.0` | Tikhonov parameter, relative to mean sensor coefficient energy |
| `standardise_sensors` | `True` | per-channel standardisation before the sensor decomposition |
| `pod_method` | `"svd"` | `"svd"`, `"snapshot"`, `"randomized"` or `"auto"` |

After `fit`: `Psi`, `Sigma`, `B` (field side), `Phi`, `C` (sensor side), `M` (the
map), `q_mean`, `s_mean`, `s_scale` — or `Psi_ext` in extended-POD mode. Methods:
`predict`, `coefficients`, `encode`, `score`, `project`, `floor` and
`observability`, plus the `extended` and `n_params` properties. The extended-POD route skips the field SVD; `encode`, `project`,
`floor` and `observability` need the field POD and raise there.


```python
from field_estimation import ridge_cv
best_ridge, scores = ridge_cv(Q, Sd, tr, ridges=(0, 1e-6, 1e-4, 1e-2, 1.0), k=5, gap=100)
```

`epod.blocked_folds(idx, k, gap)` yields those folds directly if you want to
cross-validate something else.

### Neural estimation: `BranchedAE`

Adds a trained sensor branch `G` to a **frozen** encoder/decoder pair:
`ŷ = G(s)`, `φ̂ = D(G(s))`.

```python
from field_estimation import BranchedAE, LinearLatent, TorchLatent

latent = LinearLatent(pod.Psi, pod.Q_mean)        # or TorchLatent(fitted_ae)
model = BranchedAE(latent, branch="mlp", n_delays=25, hidden=(128, 128)).fit(Q, S, tr)
model.score(Q, S, te)
```

Note the calling convention: because the windows are causal, `fit`, `encode`,
`predict` and `score` take the **whole** record plus an index array, not a
pre-sliced block.

The latent space is either `LinearLatent(Psi, q_mean)` (a POD basis) or
`TorchLatent(ae_or_cae)` (a fitted `AE`/`CAE`), and the branch is one of four
architectures, which spans the design space in a controlled way:

| encoder/decoder | branch | equivalent to |
|---|---|---|
| POD | `linear` | POD-LSE, up to the optimiser |
| POD | `mlp` | nonlinear sensor map, linear manifold |
| AE | `linear` | linear sensor map, nonlinear manifold |
| AE | `mlp` / `cnn` / `gru` | fully nonlinear |

Branch options: `branch` (`"linear"`, `"mlp"`, `"cnn"`, `"gru"`), `n_delays=25`,
`delay_stride=1`, `hidden=(128, 128)`, `activation="tanh"`, `dropout`,
`cnn_channels=(32, 64)`, `kernel_size=5`, `gru_hidden=64`, `gru_layers=1`.
Training options: `lambda_field=0.0` (weight on a field-space loss term in
addition to the latent term; the loss normalises each term by its own target's
variance, so this is dimensionless), `latent_weight` (`"energy"` keeps modes
weighted by energy, `"unit"` whitens them), `sensor_noise` (Gaussian noise added
to training batches only), `finetune_decoder=False`, plus the usual
`learning_rate` / `epochs` / `batch_size` / `val_fraction` / `patience` /
`grad_clip` / `seed` / `device`. `track_sets` records extra scores during
training; `loss_history` and `val_loss_history` hold the curves.

`LatentForecaster` rolls the latent state forward with a GRU trained on a
multi-step unrolled loss, completing the pipeline
`s → G → y → F → y' → D → φ`:

```python
from field_estimation import LatentForecaster
f = LatentForecaster(n_latent=64, n_delays=25, n_unroll=5).fit(Z, tr)
Z_future = f.rollout(Z[:, -25:], horizon=50)
```

### JAX implementations

`PODLSEJax`, `BranchedAEJax`, `pod_jax` and `ridge_cv_jax` at package level,
`extended_pod_jax` and `lse_map_jax` from `field_estimation.epod_jax`, with
`LinearLatentJax` / `AutoencoderLatentJax` as latent spaces.
Same interfaces as above. `ridge_cv_jax` forms the Gram matrix once and maps only
the solve across candidates.
Importing `field_estimation` imports torch and JAX and enables 64-bit JAX.


### Diagnosing a result

```python
from field_estimation import mode_observability, projection_floor, nmse

obs   = mode_observability(B, C)             # rho2 per field mode: what ANY linear map could reach
floor = projection_floor(Q_test, Psi, q_mean)  # error of projecting truth onto the basis, no estimation
```

* `mode_observability(B, C)` returns `rho2` (per-mode explainable fraction), `R`
  (field-mode × sensor-mode correlations) and `cum` (energy-weighted cumulative
  unexplained fraction). It orthonormalises `C` first.
* `projection_floor` is the floor from the basis alone. A score far above the
  floor while the floor keeps falling means more modes will not help you.
* Metrics: `nmse` (top level), and `nmse_per_snapshot`, `fluctuation_variance`
  (pass its result as `var` to score several windows against one normaliser),
  `cosine`, `energy_ratio` from `field_estimation.epod`.

---

## Saving and reloading models

Models are stored under a hash of their configuration, so a matching model is
loaded instead of retrained:

```python
from config.model_config import auto_load_or_create, save_model_to_config, load_model_from_config

ae = auto_load_or_create(AE, data=X, training_data_filename="wake.h5", n_latent=8)
config, path = save_model_to_config(lstm, training_data_filename="wake.h5")
lstm = load_model_from_config(q=config.to_hash())
```

Each entry is `<store>/<hash>/model_config.yaml` (class path and options) plus
`trained_matrices.npz` (the arrays), under `results/model_configs/` by default
(`results/esn_configs/` for ESNs); pass `config_dir` / `save_dir` / `load_dir`
for somewhere else. `ESNConfig` and
`auto_load_or_create_esn` do the same for `ESN_model`, using `esn_config.yaml`.
Also available: `find_matching_config`, `list_saved_configs`, and `**overrides`
on load for options that should not participate in the hash, such as `device`.

Any model implementing `Configurable` — a dataclass reporting
`config_options()` and `trained_arrays()` and rebuilding via `from_trained()` —
can be stored this way. This is what makes the sweep scripts resumable.

## Plotting

`plotting` holds the ROM-side figures: `plot_modes`, `plot_time_coefficients`,
`plot_spectrum`, `plot_flows_rms`, all taking a fitted `POD`.

`field_estimation.plots` holds the estimation figures: `plot_spectrum`,
`plot_observability`, `plot_extended_modes`, `plot_comparison`,
`plot_error_history`, `plot_horizon`, and `animate_reconstruction`. These take
**plain arrays** rather than estimator objects
Neither module selects a matplotlib backend; set one yourself
(`matplotlib.use("Agg")`) before importing.

---

## Layout

```
.
├── data/                      snapshot files (untracked)
├── results/                   one folder per study (only results.csv tracked)
├── tests/                     165 checks (25 need VERIFY_SLOW=1)
├── experiments/
│   ├── april_wake/            case reader, preprocessing, 14 scripts
│   ├── bl/                    convergence study, subsetting
│   └── perf/                  torch-vs-JAX benchmark
└── src/
    ├── datasets/
    │   ├── snapshots.py       SnapshotSpec, SPECS, load_snapshots, readers
    │   └── splitting.py       split_indices, prepare_split, diagnostics
    ├── models/
    │   └── data_driven/
    │       ├── autoencoders/  Projector, POD, SPOD, AE, CAE, AEJax, CAEJax
    │       ├── forecasters/   ESN_model, LSTM, LSTM_model, LSTMJax
    │       ├── roms/          the ten projector × forecaster combinations
    │       ├── latent_rom.py  LatentROM: fitting, sensors, observables
    │       ├── configurable.py  the save/restore protocol
    │       └── training.py    TrainingHistory
    ├── field_estimation/
    │   ├── epod.py            PODLSE, extended POD, delays, metrics, ridge CV
    │   ├── branched_ae.py     BranchedAE, SensorBranch, LatentForecaster
    │   ├── epod_jax.py        PODLSEJax and the JAX primitives
    │   ├── branched_ae_jax.py BranchedAEJax and its latent spaces
    │   └── plots.py           figures for both families
    ├── config/
    │   └── model_config.py    ModelConfig, ESNConfig, hashed save/load
    └── plotting/
        ├── pod.py             modes, coefficients, spectrum, RMS
        └── figures.py         shared figure helpers
```

## Attribution

The modelling core (`Model`, `HistoryTracker`, the integrators, the physical
models) and the ESN reservoir come from A. Nóvoa's
[real-time bias-aware DA](https://github.com/andreanovoa/real-time-bias-aware-DA)
repository, now released as the standalone packages
[`dynamodels`](https://github.com/andreanovoa/dynamodels),
[`echostatenetwork`](https://github.com/andreanovoa/EchoStateNetwork) and
[`romda`](https://github.com/andreanovoa/real-time-bias-aware-DA), which this
repository depends on and `src/models/__init__.py` re-exports. `add_pdf_page` and
`get_figsize_based_on_domain` in `plotting/figures.py` come from the same source.
The data-assimilation and bias-estimation layers are not carried here — see the
original for those.
