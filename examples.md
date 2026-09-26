# Usage guide and worked examples

This is a hands-on companion to the [README](README.md). 

It covers the library under `src/`.

**Contents**

1. [Setup](#1-setup)
2. [The mental model](#2-the-mental-model)
3. [Loading and splitting data](#3-loading-and-splitting-data) — `datasets`
4. [Projectors: reducing a field](#4-projectors-reducing-a-field) — POD, SPOD, AE, CAE, AEJax, CAEJax
5. [Forecasters: advancing latent coefficients](#5-forecasters-advancing-latent-coefficients) — LSTM, LSTMJax, ESN
6. [ROMs: projector + forecaster + sensors](#6-roms-projector--forecaster--sensors) — POD_ESN, AE_LSTM, …
7. [Field estimation from sparse sensors](#7-field-estimation-from-sparse-sensors) — PODLSE, BranchedAE, LatentForecaster
8. [Saving and reloading trained models](#8-saving-and-reloading-trained-models) — `config.model_config`
9. [Two complete workflows](#10-two-complete-workflows)
10. [Pitfalls and FAQ](#11-pitfalls-and-faq)

---

## 1. Setup

### Install

```bash
git clone <this-repo> && cd autoencoder-rom
pip install -e . --use-pep517
pytest tests/ -q          # sanity check, no data needed
```

The layout is `src`-based, so the packages are imported at top level:
`from datasets import ...`, **not** `from src.datasets import ...`. If you have
not installed the package, running from inside `src/` (or adding it to
`sys.path`) works too.

### A synthetic dataset for every example

Every example below runs on the same small synthetic field.
It is two velocity components of travelling waves on a 32 × 16 grid, with a square "solid body"
marked by NaN.

```python
import matplotlib
matplotlib.use("Agg")  # no GUI windows; see "Pitfalls"

import numpy as np

DOMAIN = [0.0, 2 * np.pi, 0.0, np.pi]      # [x0, x1, y0, y1]

def synthetic_field(N_t=1200, Nx=32, Ny=16, seed=0):
    """(Nu=2, N_t, Nx, Ny) float32 field, NaN inside a solid body."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 32 * np.pi, N_t)
    x = np.linspace(*DOMAIN[:2], Nx)
    y = np.linspace(*DOMAIN[2:], Ny)
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    u = (np.sin(Xg)[None] * np.cos(t)[:, None, None]
         + 0.5 * np.sin(2 * Xg + Yg)[None] * np.sin(2 * t)[:, None, None])
    v = np.cos(Xg + Yg)[None] * np.sin(t)[:, None, None]
    X = np.array([u, v]) + 0.02 * rng.standard_normal((2, N_t, Nx, Ny))
    X[:, :, 10:14, 6:10] = np.nan           # the solid body
    return X.astype(np.float32)

X = synthetic_field()
print(X.shape)                               # (2, 1200, 32, 16)
```

To use real data instead, replace `X` with `load_snapshots(...)` from
[section 3](#3-loading-and-splitting-data).

---

## 2. The mental model

Each broader model type has an interface that
every implementation shares:

```
            ┌──────────── Projector ────────────┐
 field X ──►│ encode ──► latent Z ──► decode    │──► field X̂
 (Nu,Nt,    └───────────────────────────────────┘    (N_x, Nt)
  Nx,Ny)                    │  ▲
                            ▼  │
                  Forecaster: Z(t) ──► Z(t+1), Z(t+2), …

 sensors S ──► Field estimator (PODLSE / BranchedAE) ──► field X̂
```

* A **projector** (`POD`, `SPOD`, `AE`, `CAE`, `AEJax`, `CAEJax`). All of them have `fit / encode / decode /
  reconstruct / score`, so you can swap one for another.
* A **forecaster** (`ESN_model`, `LSTM`, `LSTMJax`) with open loop and closed loop forecasting 
* A **ROM** (`POD_ESN`, `AE_LSTM`, …) is one object that is both, plus a set of
  virtual sensors to read the forecast at.
    * A **field estimator** (`PODLSE`, `BranchedAE`), which can map one set of data to the latent space of another dataset
  

### Array layouts


| name | shape | where you see it |
|---|---|---|
| **grid** | `(Nu, Nt, Nx, Ny)` | what loaders return; NaN at solid points |
| **snapshot** | `(Nu, Nx, Ny)` | one grid snapshot |
| **flat** | `(N_x, Nt)` | one snapshot per **column**, fluid points only, `N_x = Nu * N_fluid` |
| **latent** | `(N_latent, Nt)` | what `encode` returns; one snapshot per column |
| **sensors** | `(N_s, Nt)` | one sample per column, aligned with the flat field |

Rows of the flat layout are grouped by component: all fluid points of `u_x`,
then all fluid points of `u_y`, and so on. Projectors accept grid, snapshot or
flat input; `decode` always returns flat. The field estimators in
`field_estimation` work in the flat and sensor layouts only.

Two helpers for moving between grid and flat:
```python
def grid_to_flat(X):
    """(Nu, Nt, Nx, Ny) -> flat (Nu * N_fluid, Nt), plus the fluid mask."""
    Nu, Nt = X.shape[:2]
    fluid = ~np.isnan(X[0, 0]).ravel()            # (Nx * Ny,) True at fluid points
    Q = X.reshape(Nu, Nt, -1)[:, :, fluid].transpose(0, 2, 1).reshape(-1, Nt)
    return Q, fluid

def flat_to_grid(Q, fluid, grid_shape):
    """flat (Nu * N_fluid, Nt) -> (Nu, Nt, Nx, Ny) with NaN at solid points."""
    Nu, Nx, Ny = grid_shape
    Nt = Q.shape[1]
    G = np.full((Nu, Nt, Nx * Ny), np.nan)
    G[:, :, fluid] = Q.reshape(Nu, -1, Nt).transpose(0, 2, 1)
    return G.reshape(Nu, Nt, Nx, Ny)

Q, fluid = grid_to_flat(X)
print(Q.shape)                                    # (992, 1200): 2 * 496 fluid points
assert np.array_equal(flat_to_grid(Q, fluid, (2, 32, 16)), X, equal_nan=True)
```

---

## 3. Loading and splitting data

`datasets` is pure NumPy: importing it never loads torch or JAX.

### Loading a known dataset

```python
from datasets import SPECS, load_snapshots

X = load_snapshots(SPECS["circle"])     # (Nu, Nt, Nx, Ny), float32, NaN at solid points
```

`SPECS` holds three datasets — `"bl"` (3-component boundary layer, HDF5),
`"circle"` and `"triangle"` (2-component wakes, `.mat`). Each is a
`SnapshotSpec`. Specs are frozen, so to read at a different resolution you make a
modified copy with `.replace`:

```python
spec = SPECS["circle"].replace(downsample=(4, 4), max_snapshots=200)
X_small = load_snapshots(spec)          # (2, 200, 128, 32)
```

| field | what it does |
|---|---|
| `filename` | path relative to `$DATA_ROOT` |
| `fields` | variable names, one per velocity component, in the order you want on axis 0 |
| `downsample=(xs, ys)` | spatial decimation, always in `(x, y)` order whatever the file stores |
| `stride` | keep every `stride`-th snapshot |
| `max_snapshots` | cap on how many snapshots to read |
| `antialias=True` | average blocks of `xs × ys` points rather than just striding |
| `chunk` | snapshots the HDF5 reader decimates at a time (memory control) |

**Where the files come from.** `data_path(filename)` joins `filename` onto
`$DATA_ROOT`, which defaults to `data` **relative to your current working
directory**, not to the repo. If you run a notebook from somewhere else, set it:

```bash
export DATA_ROOT=/path/to/autoencoder-rom/data
```

### Your own file

For a one-off file, make a spec directly:

```python
from datasets import SnapshotSpec

spec = SnapshotSpec(filename="wakes/my_run.mat", fields=("ux", "uy"), downsample=(2, 2))
```

`.mat`, `.h5` and `.hdf5` are supported. For another format, write a function
with the reader signature — it returns one `(Nt, Nx, Ny)` float32 array per field
— and register it:

```python
from datasets import snapshots

def read_npz(path, fields, *, stride, max_snapshots, xs, ys, antialias, chunk):
    f = np.load(path)
    return [f[name][:max_snapshots:stride, ::xs, ::ys].astype(np.float32) for name in fields]

snapshots.READERS[".npz"] = read_npz
```

### Splitting a time series honestly

Consecutive snapshots are highly correlated. If you split a time series at
random, or even in adjacent blocks, the test set contains near-copies of
training snapshots and your error estimate is optimistic. The splitter places
a **gap** of discarded samples between blocks, sized by the data's own
correlation time:

```
[warmup]--[train]--gap--[val]--gap--[test]
```

The one-line version measures the gap, splits, and runs diagnostics:

```python
from datasets import prepare_split, LEAK_THRESHOLD, SHIFT_THRESHOLD

X_train, X_val, X_test, meta = prepare_split(X, val_frac=0.2, test_frac=0.2)

print(meta["gap"], meta["n_train"], meta["n_val"], meta["n_test"])
if meta["test_nn_dist_median"] < LEAK_THRESHOLD:
    print("test block repeats the training set -- not measuring generalisation")
if meta["test_mean_shift"] > SHIFT_THRESHOLD:
    print("test block has a different mean from training -- expect a bias")
print("fraction of test variance outside the training span:", meta["span_ceiling"])
```

`meta` holds `gap`, the block sizes, `test_nn_dist_median`, `test_nn_dist_min`,
`test_mean_shift` (and the same three for `val_`), and `span_ceiling`.
`span_ceiling` is the best any *linear* model could ever do on the test block.
eg. if it is 0.1, even infinitely many POD modes leave 10 % of the test variance
unexplained.

When you need index arrays rather than sliced data — for example because you
are splitting a flat matrix and a sensor record together — use the pieces:

```python
from datasets import decorrelation_lag, split_indices, split_diagnostics, linear_span_ceiling

gap = decorrelation_lag(X)                      # samples until correlation < 1/e
tr, va, te = split_indices(X.shape[1], val_frac=0.2, test_frac=0.2, gap=gap)

Q_train, Q_test = Q[:, tr], Q[:, te]            # the same indices work on any array
print(split_diagnostics(X[:, tr], X[:, te]))
```

With `val_frac=0` you get an empty validation array and a single gap.
`warmup` drops leading samples; you need it after delay embedding
([section 7](#7-field-estimation-from-sparse-sensors)).

---

## 4. Projectors: reducing a field

```python
from models.data_driven.autoencoders import POD, SPOD, AE, CAE, AEJax, CAEJax
```

### The shared interface

```python
p.fit(X_train)            # grid (Nu, Nt, Nx, Ny) -- fitting needs the grid, to find the solid mask
Z = p.encode(X_test)      # (N_latent, Nt)
X_hat = p.decode(Z)       # (N_x, Nt) flat, temporal mean added back
X_hat = p.reconstruct(X_test)   # encode then decode
err = p.score(X_test)     # mean squared error over fluid points and snapshots
p.n_params                # a property, not a method
p.Q_mean                  # (N_x, 1) temporal mean of the training data
```

After fitting, `encode`, `reconstruct` and `score` also accept a snapshot
`(Nu, Nx, Ny)` or flat `(N_x, Nt)` array. The flat input must have the same
fluid mask as the training data.

### Autoencoders

Four autoencoders share one set of training options. You choose an architecture
(dense or convolutional) and a framework (PyTorch or JAX):

| | PyTorch | JAX |
|---|---|---|
| dense | `AE(hidden=(512, 128))` | `AEJax(hidden=(512, 128))` |
| convolutional | `CAE(channels=(16, 32, 64), kernel_size=3, stride=2, pad=1)` | `CAEJax(...)` |

```python
ae = AE(n_latent=6, hidden=(256, 64), epochs=300, patience=30, seed=0).fit(X_train)
print("AE test MSE:", ae.score(X_test), "params:", ae.n_params)

h = ae.training_history          # per-epoch losses and learning rate
print(h.n_epochs_run, h.train[-1], h.val[-1], h.lr[-1])
```

What `fit` does:
1. Subtracts the temporal mean, then divides each velocity component by its
   standard deviation.
2. Holds out the **last** `val_fraction` (default 0.2) of the snapshots for early
   stopping. Pass `val_fraction=0` to train on everything with no early stopping.
3. Trains with Adam on masked MSE. Solid points are excluded from the loss.
4. Halves the learning rate after `lr_patience` epochs without improvement
   (`lr_factor`, `min_lr`), stops after `patience` epochs without improvement, and
   **restores the best-validation weights**.

| option | default | notes |
|---|---|---|
| `n_latent` | 10 | latent dimension |
| `epochs` | 500 | maximum; early stopping usually ends sooner |
| `learning_rate`, `batch_size`, `weight_decay` | 1e-3, 32, 0 | |
| `val_fraction`, `patience`, `threshold` | 0.2, 50, 1e-4 | early stopping |
| `lr_factor`, `lr_patience`, `min_lr` | 0.5, 10, 1e-6 | plateau schedule |
| `activation` | `"tanh"` | |
| `device` | `"cpu"` | torch only; e.g. `"cuda"` |
| `dtype`, `layer_activations` | `"float32"`, `None` | JAX only |
| `seed` | 0 | |

The convolutional models treat each velocity component as an image channel.
Every entry of `channels` adds one strided convolution. 

```python
cae = CAE(n_latent=6, channels=(16, 32), epochs=200).fit(X_train)
```

The JAX classes have the same constructor and behaviour. They are usually
faster on a GPU once compiled, though the first epoch includes compile time:

```python
aej = AEJax(n_latent=6, hidden=(256, 64), epochs=300).fit(X_train)
```

### Comparing projectors

Because every projector has the same interface, a comparison is just a loop:

```python
models = {
    "POD": POD(n_modes=6),
    "AE":  AE(n_latent=6, hidden=(128, 32), epochs=200),
    "CAE": CAE(n_latent=6, channels=(8, 16), epochs=100),
}
for name, m in models.items():
    m.fit(X_train)
    print(f"{name:4s}  test MSE {m.score(X_test):.2e}   params {m.n_params}")
```

`score` is an absolute MSE. To compare across datasets, normalise it by the
test variance: `m.score(X_test) / np.nanvar(X_test)`.

### Getting fields back on the grid

`decode` returns the flat layout. Use `flat_to_grid` from
[section 2](#array-layouts) with the projector's mask to get an image.

```python
X_hat = pod.reconstruct(X_test)                             # (N_x, Nt)
G_hat = flat_to_grid(X_hat, pod.fluid_mask_flat, pod.grid_shape)   # (Nu, Nt, Nx, Ny)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].imshow(X_test[0, 0].T, origin="lower"); ax[0].set_title("truth $u_x$")
ax[1].imshow(G_hat[0, 0].T, origin="lower"); ax[1].set_title("POD reconstruction")
fig.savefig("reconstruction.png")
```

### Two methods used for sensors

* `decode_at(Z, idx)` decodes only the rows `idx` of the flat field. For POD it
  never forms the full field (`Psi[idx] @ Z`), so reading a few sensors off a
  forecast costs almost nothing.
* `spatial_basis(z0)` is the decoder Jacobian, `(N_x, N_latent)`, at latent
  state `z0`. For POD it is exactly `Psi`; for autoencoders it is computed by
  finite differences. The ROMs place sensors with QR pivoting on this matrix.

```python
Z = pod.encode(X_test)
idx = np.array([10, 200, 700])                 # three rows of the flat field
readings = pod.decode_at(Z, idx)               # (3, Nt)
J = ae.spatial_basis()                         # (N_x, 6), linearised about z = 0
```

---

## 5. Forecasters: advancing latent coefficients

You can use one on its own, as shown here, or inside a ROM ([section 6](#6-roms-projector--forecaster--sensors)),
which wires it up for you.

```python
from models.data_driven import LSTM, LSTMJax, ESN_model, phi_to_esn_layout

pod = POD(n_modes=6).fit(X_train)
Z_train = pod.encode(X_train)       # (6, Nt_train)
Z_test = pod.encode(X_test)
```

### LSTM

A single-layer LSTM written in NumPy. It is trained by truncated backpropagation
through time and keeps the epoch with the lowest **closed-loop** validation error.
It expects time on the first axis, so transpose the latent array:

```python
lstm = LSTM(N_dim_in=6, N_units=32, N_wash=50, seq_len=100, epochs=40, lr=3e-3)
lstm.train(Z_train.T, verbose=False)            # (Nt, N_dim_in), or (L, Nt, N_dim_in) for several runs

# Open loop ("teacher forcing"): feed the true inputs to warm up the hidden state
Y, state = lstm.openLoop(Z_test.T[:50])         # Y: (50, 6, 1) one-step predictions

# Closed loop: feed each prediction back in 
forecast, _ = lstm.closedLoop(Z_test.T[49], 200, state)    # (200, 6, 1)
field_forecast = pod.decode(forecast[:, :, 0].T)           # (N_x, 200)
```

Key options: `N_units`, `N_wash` (washout steps before the loss is counted),
`seq_len` (BPTT window), `epochs`, `lr`, `clip` (gradient norm), `val_frac`,
`norm_method` (`"range"`, `"std"`, `"max"`, `"mean"`, `None`), `recurrent_init`
(`"orthogonal"` or `"uniform"`). Each training segment must be at least
`N_wash + seq_len + 1` long. `lstm.training_history` records the loss per epoch.

`LSTMJax` has the same constructor and methods, with a `dtype` (datatype) option. It starts
from the same initial weights as `LSTM` for the same seed.

### Echo state network

`ESN_model` wraps the `echostatenetwork` reservoir in the `dynamodels.Model`
interface. Two things are different from the LSTM:

* **It trains at construction**, including a Bayesian search over the spectral
  radius `rho` and input scaling `sigma_in`. `N_func_evals` and `N_grid` set the
  search budget. Larger values search more thoroughly but train more slowly.
* **It wants `(L, Nt, N_dim)` data**, where `L` is the number of independent
  runs. `phi_to_esn_layout` converts a projector's `(N_latent, Nt)` output to
  this layout.

```python
esn = ESN_model(
    data=phi_to_esn_layout(Z_train),
    dt=0.01,                 # time between snapshots
    N_units=100,             # reservoir size
    N_wash=10,               # washout steps
    N_func_evals=10,         # hyperparameter search budget (default 40)
    N_grid=3,
    plot_training=False,     # default True opens figures
)
```

By default the ESN splits the data it is given into 80 % training time and
validation = 20 % of that, and the rest (if `perform_test`) is a test block. Pass
`N_train=`, `N_val=`, `N_test=` (in steps) to set the split yourself.

Quick evaluation — washout on the first `N_wash` true samples, then forecast in
closed loop:

```python
pred, target = esn.closed_loop(Z_test.T, n_steps=200)     # both (200, 6)
print("forecast NMSE:", np.mean((pred - target) ** 2) / np.var(target))
```

As a `Model` it also runs through the standard time-integration interface, which
is what the ROMs use:

```python
psi, t = esn.time_integrate(Nt=200)      # psi: (201, N_dim + N_units, m) = [u; r] per step
esn.update_history(psi, t)
u_hist = esn.get_observable_hist()       # (Nt, N_dim, m) -- the latent forecast
```

`m` is the ensemble size (1 unless you are doing data assimilation).
`esn.reset_forecaster(u0=...)` restarts from a new state without retraining.

---

## 6. ROMs: projector + forecaster + sensors

A ROM is a single object that fits a projector, encodes the training data,
trains a forecaster on those latent coefficients, and places virtual sensors.
All ten combinations exist:

| | ESN | LSTM |
|---|---|---|
| POD | `POD_ESN` | `POD_LSTM` |
| AE | `AE_ESN` | `AE_LSTM` |
| CAE | `CAE_ESN` | `CAE_LSTM` |
| AEJax | `AEJax_ESN` | `AEJax_LSTM` |
| CAEJax | `CAEJax_ESN` | `CAEJax_LSTM` |

### Building one

```python
from models.data_driven import POD_ESN, POD_LSTM, AE_LSTM

rom = POD_ESN(
    data=X_train, dt=0.01,
    n_modes=6,                      # -> POD
    domain=DOMAIN,                  # -> POD (needed if you restrict sensors by region)
    Nq=8,                           # number of sensor points to place
    N_units=100, N_wash=10,         # -> ESN
    N_func_evals=10, N_grid=3, plot_training=False,
)
```

Keyword arguments are routed automatically. Anything the projector defines goes
to the projector, `LatentROM`'s own options (`Nq`, `measure_modes`,
`qr_selection`, …) are consumed, and everything else goes to the forecaster. The
one exception is epochs: both halves might train in epochs, so a plain `epochs=`
is rejected. Use the explicit names instead:

```python
rom_ae = AE_LSTM(
    data=X_train, dt=0.01,
    n_latent=6, hidden=(128, 32), projector_epochs=200,     # AE
    N_units=32, N_wash=20, seq_len=100, forecaster_epochs=20, # LSTM
    Nq=8,
)
print(rom_ae.projector_history.n_epochs_run, rom_ae.forecaster_history.n_epochs_run)
```

POD takes `n_modes`; the autoencoders take `n_latent`. Passing
`projector_epochs` to a POD ROM, or `forecaster_epochs` to an ESN ROM, raises a
`TypeError` because they do not train in epochs.

### Forecasting and reading the result

```python
psi, t = rom.time_integrate(Nt=200)
rom.update_history(psi, t)

y = rom.get_observable_hist()                 # (Nt+1, Nq, m): the sensor readings
Z_fc = rom.get_latent_coefficients(Nt=200)    # (200, N_latent, m): the latent forecast
field_fc = rom.decode(Z_fc[:, :, 0].T)        # (N_x, 200): the full-field forecast
```

The ROM *is* its projector as well, so `rom.encode`, `rom.decode`,
`rom.score` and `rom.Psi` (for POD) all work. `rom.latent_training_trajectory`
holds the `(N_latent, Nt)` coefficients the forecaster was trained on.

To restart the forecast from the first training snapshot (or from `Z0=`)
without retraining:

```python
rom.reset_case(reset_forecaster=True)
```

### Sensors
By default `Nq` points are chosen by **QR pivoting** on the decoder's
spatial basis, which picks the most informative, least redundant locations.

**`Nq` changes meaning.** You pass the number of sensor *points*. Once the ROM
is built, `Nq` holds the number of *observables*, which is one per point per
velocity component:

```python
print(rom.N_sensors, rom.Nq)        # 8 points, 16 observables for a 2-component field
```

Sensor locations are raw grid indices `u * Nx * Ny + g`. To find where a sensor
is, or which flat-field row it reads, use:

```python
Nu, Nx, Ny = rom.grid_shape
g = rom.sensor_locations[: rom.N_sensors]           # the u_x copy of each point
ix, iy = np.unravel_index(g, (Nx, Ny))              # grid coordinates
rows = rom.sensor_rows                              # rows of the flat field -- use these, not the raw indices
```

Controlling placement:

```python
# at construction
rom = POD_ESN(data=X_train, dt=0.01, n_modes=6, domain=DOMAIN, Nq=6,
              domain_of_measurement=[3.0, 6.3, 0.0, 3.2],   # only in this box (needs domain=)
              down_sample_measurement=2,                     # only every 2nd grid point
              qr_selection=True,                              # False = random placement
              N_units=100, N_wash=10, N_func_evals=10, N_grid=3, plot_training=False)

# or afterwards, without retraining anything
rom.select_sensors(N_sensors=4, qr_selection=True, domain_of_measurement=[3.0, 6.3, 0.0, 3.2])

# or observe the latent coefficients themselves rather than any sensors
rom_modes = POD_ESN(data=X_train, dt=0.01, n_modes=6, measure_modes=True,
                    N_units=100, N_wash=10, N_func_evals=10, N_grid=3, plot_training=False)
print(rom_modes.Nq)       # 6 -- one observable per mode
```

You can also pass `sensor_locations=` to fix the sensors yourself (raw grid
indices, all components). If QR pivoting can find fewer distinct points than
you asked for, it warns and fills in the remainder.

---

## 7. Field estimation from sparse sensors

Imagine you have a real sensor record `S` `(N_s, Nt)` (load cells, pressure taps, probes) 
synchronised with a field record `Q` `(N_x, Nt)`, and you want to estimate `Q` from `S` alone.

```python
from field_estimation import (
    PODLSE, delay_embed, ridge_cv, nmse, mode_observability, projection_floor,
    BranchedAE, LinearLatent, TorchLatent, LatentForecaster,
)
```

Importing `field_estimation` loads both torch and JAX and switches JAX to 64-bit.

For the examples we need sensors. Real load cells measure forces, which are
roughly quadratic in velocity, so make five sensors that see the square of the
velocity at random points, plus noise:

```python
from datasets import split_indices

Q, fluid = grid_to_flat(X)
Q = Q.astype(np.float64)
rng = np.random.default_rng(1)
probe_rows = rng.choice(Q.shape[0], size=5, replace=False)
S = Q[probe_rows] ** 2 + 0.01 * rng.standard_normal((5, Q.shape[1]))   # (5, Nt)
```

### Delay embedding

One sensor sample is rarely enough to pin down the field, but a short history
of it often is. `delay_embed` stacks lagged copies so each column holds
`s_t, s_{t-1}, …, s_{t-n+1}`. It is causal: no future samples are used.

```python
n_delays = 10
Sd = delay_embed(S, n_delays=n_delays)                # (5 * 10, Nt)

# The first n_delays - 1 columns contain zero padding; skip them with warmup.
tr, _, te = split_indices(Q.shape[1], val_frac=0, test_frac=0.25,
                          gap=decorrelation_lag(X), warmup=n_delays - 1)
```

Always embed the **whole** record and then index it. If you embed each block
separately, the start of each block gets zero-padded windows.

### PODLSE: the linear baseline

POD of the field, a reduced basis for the sensors, and a ridge-regularised linear map between the two sets of coefficients.
It is a good baseline for what a neural model should try to beat.

```python
lse = PODLSE(r_field=8, r_sensor=20, ridge=1e-3).fit(Q[:, tr], Sd[:, tr])

print("test NMSE:", lse.score(Q[:, te], Sd[:, te]))     # 0 = perfect, 1 = no better than the mean
Q_hat = lse.predict(Sd[:, te])                          # (N_x, n_test)
```

`PODLSE` takes **pre-sliced blocks**. Contrast this with `BranchedAE` below.

| option | default | meaning |
|---|---|---|
| `r_field` | `None` | field POD modes; `None` computes extended POD (see below) |
| `r_sensor` | `None` | sensor directions kept; `None` keeps all |
| `sensor_basis` | `"pod"` | `"pod"` ranks sensor directions by their variance; `"pls"` by covariance with the field |
| `ridge` | 0 | Tikhonov regularisation, relative to the mean sensor coefficient energy |
| `standardise_sensors` | `True` | scale each channel to unit variance first (almost always keep this) |
| `pod_method` | `"svd"` | `"svd"`, `"snapshot"`, `"randomized"`, `"auto"` |

**Extended POD** (Borée 2003) is the `r_field=None` special case. It skips the
field decomposition entirely and gives one extended field mode per sensor mode,
stored in `Psi_ext`:

```python
epod = PODLSE(ridge=1e-3).fit(Q[:, tr], Sd[:, tr])
print(epod.extended, epod.Psi_ext.shape)          # True, (N_x, 50)
```

Choosing the ridge by blocked cross-validation inside the training block:
```python
best_ridge, cv_scores = ridge_cv(Q, Sd, tr, ridges=(0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                                 k=5, gap=20, r_field=8, r_sensor=20)
lse = PODLSE(r_field=8, r_sensor=20, ridge=best_ridge).fit(Q[:, tr], Sd[:, tr])
```

Extra keyword arguments to `ridge_cv` go to the `PODLSE` constructor.
`epod.blocked_folds(idx, k, gap)` yields the same folds if you want to
cross-validate something else.

### Diagnosing a result

```python
floor = lse.floor(Q[:, te])     # error of projecting the TRUTH onto the r_field modes
obs = lse.observability()       # per mode: how much any linear map from these sensors can explain

print("projection floor:", floor)                # the basis limit
print("observable fraction per mode:", np.round(obs["rho2"], 2))
print("test NMSE:", lse.score(Q[:, te], Sd[:, te]))
```

* Score ~= floor**:  the basis is the limit
* `rho2` is small for energetic modes

The same functions work on arrays you computed yourself:
`mode_observability(B, C)` and `projection_floor(Q_test, Psi, q_mean)`.

Other metrics are in `field_estimation.epod`:

```python
from field_estimation.epod import nmse_per_snapshot, fluctuation_variance, cosine, energy_ratio

var = fluctuation_variance(Q[:, te])                       # one normaliser for the whole block
e_t = nmse_per_snapshot(Q[:, te], Q_hat, var=var)          # (n_test,) error over time
```

### BranchedAE: a learned sensor map

`BranchedAE` keeps a fitted encoder/decoder **frozen** and trains a new network
`G` that maps a window of sensor history straight into the latent space:
`ẑ = G(s)`, `q̂ = D(ẑ)`.

First wrap a latent space. Either a POD basis:

```python
from field_estimation.epod import pod as pod_svd

Psi, Sigma, B, q_mean = pod_svd(Q[:, tr], r=8, subtract_mean=True)   # fit on TRAINING data only
latent = LinearLatent(Psi, q_mean)
```

or a fitted torch autoencoder:

```python
ae = AE(n_latent=8, hidden=(128, 32), epochs=200).fit(X[:, tr])
latent_ae = TorchLatent(ae)             # also accepts a fitted CAE
```

Then train the branch. Note the calling convention: because the windows are
causal, `fit`, `predict`, `encode` and `score` take the **whole** record `(Q, S)`
plus an **index array**, and `S` is the **raw** sensor record. The branch builds
its own delay windows.

```python
bae = BranchedAE(latent, branch="mlp", n_delays=n_delays, hidden=(128, 128),
                 epochs=300, patience=30).fit(Q, S, tr)

print("test NMSE:", bae.score(Q, S, te))
Q_hat = bae.predict(S, te)          # (N_x, n_test)
Z_hat = bae.encode(S, te)           # (8, n_test)
```

Every index you pass must be at least `bae.warmup = (n_delays - 1) * delay_stride`.
`split_indices(..., warmup=n_delays - 1)` guarantees this.

The four combinations of latent space and branch cover the design space in a
controlled way, so you can attribute an improvement to the right component:

| latent | `branch=` | what it is |
|---|---|---|
| POD | `"linear"` | POD-LSE again, trained by gradient descent — a sanity check |
| POD | `"mlp"` | nonlinear sensor map onto a linear manifold |
| AE | `"linear"` | linear sensor map onto a nonlinear manifold |
| AE | `"mlp"`, `"cnn"`, `"gru"` | fully nonlinear |

Useful options: `lambda_field` (add a field-space loss term; dimensionless),
`latent_weight` (`"energy"` or `"unit"`), `sensor_noise` (augmentation on
training batches only), `finetune_decoder=True` (unfreeze a `TorchLatent`
decoder), `dropout`, `cnn_channels`, `kernel_size`, `gru_hidden`, `gru_layers`,
plus the usual `learning_rate`, `epochs`, `batch_size`, `val_fraction`,
`patience`, `grad_clip`, `seed`, `device`, `verbose`. After training,
`loss_history` and `val_loss_history` hold the curves.

To watch the test error during training (for learning curves only), pass `track_sets`:

```python
bae = BranchedAE(latent, branch="mlp", n_delays=n_delays, epochs=100,
                 track_sets={"test": (Q, S, te)}, track_every=5).fit(Q, S, tr)
epochs, test_nmse = zip(*bae.track_history["test"])
```

### LatentForecaster: estimating *and* forecasting

`LatentForecaster` is a GRU that rolls a latent trajectory forward. Combined
with a sensor branch it gives the full chain
`sensors → G → z_t → F → z_{t+1…t+h} → D → field`:

```python
Z = latent.encode(Q)                                        # (8, Nt)
fc = LatentForecaster(n_latent=8, n_delays=10, n_unroll=5, epochs=200).fit(Z, tr)

window = bae.encode(S, te[:10])                             # last 10 estimated states
Z_future = fc.rollout(window, horizon=50)                   # (8, 50)
Q_future = latent.decode(Z_future)                          # (N_x, 50)
```

`train_idx` must be contiguous. `n_unroll` is the number of steps the loss
unrolls; larger values make closed-loop forecasts more stable.

### JAX versions

`PODLSEJax`, `BranchedAEJax`, `pod_jax` and `ridge_cv_jax` (plus
`extended_pod_jax` and `lse_map_jax` in `field_estimation.epod_jax`) have the
same interfaces. The JAX latent spaces are `LinearLatentJax(Psi, q_mean)` and
`AutoencoderLatentJax(fitted_AEJax_or_CAEJax)`:

```python
from field_estimation import PODLSEJax, BranchedAEJax, LinearLatentJax

lse_j = PODLSEJax(r_field=8, r_sensor=20, ridge=1e-3).fit(Q[:, tr], Sd[:, tr])
bae_j = BranchedAEJax(LinearLatentJax(Psi, q_mean), branch="mlp",
                      n_delays=n_delays, epochs=300).fit(Q, S, tr)
```

---

## 8. Saving and reloading trained models

The config store saves a model under a hash of its class
and options, so that the next time you ask for the same model it is loaded
instead of retrained. 

```python
from config.model_config import (
    auto_load_or_create, save_model_to_config, load_model_from_config, list_saved_configs,
)

# Trains and saves the first time; loads after that.
ae = auto_load_or_create(AE, data=X_train, training_data_filename="synthetic_v1",
                         n_latent=6, hidden=(128, 32), epochs=200)
```

On disk, each entry is `<store>/<hash>/model_config.yaml` (class and options)
plus `trained_matrices.npz` (the weights and statistics). The default store is
`results/model_configs/`; pass `config_dir=` to use somewhere else, and
`force_create=True` to retrain regardless.

> **The hash covers the class, the options and `training_data_filename` — not
> the contents of `data`.** If you change the data (different split, different
> downsampling) you must change `training_data_filename` too, or you will
> get the model trained on the old data. 

Saving and loading by hand:

```python
lstm = LSTM.from_data(Z_train.T, N_units=32, N_wash=50, seq_len=100, epochs=20)
config, path = save_model_to_config(lstm, training_data_filename="synthetic_v1")
print(path)                                       # results/model_configs/<hash>

lstm_again = load_model_from_config(q=config.to_hash())
ae_on_gpu = load_model_from_config(q="<hash>", device="cuda")   # overrides not part of the hash

for entry in list_saved_configs():
    print(entry["name"], entry["model_class"], entry["training_data_filename"])
```

The following can be stored: `AE`, `CAE`, `AEJax`, `CAEJax` and `LSTM`/`LSTMJax`.
The ESN has its own store with the same idea (`auto_load_or_create_esn`,
`save_esn_model_to_config`, `load_esn_model_from_config`, stored under
`results/esn_configs/`). 
ROMs are not stored as a whole. Save the pieces, or rebuild the ROM

To make your own model storable, implement `Configurable`
(`models/data_driven/configurable.py`). It must be a dataclass that reports
`config_options()` and `trained_arrays()` and rebuilds itself via
`from_trained()` and `from_data()`.

---

## 10. Two complete workflows

### A. Which projector, and how many modes?

Train each projector at several latent sizes and plot test error against `r`.

```python
from datasets import prepare_split
from models.data_driven.autoencoders import POD, AE, CAE
from config.model_config import auto_load_or_create

X = synthetic_field()
X_train, X_val, X_test, meta = prepare_split(X, val_frac=0.2, test_frac=0.2)
test_var = float(np.nanvar(X_test))
tag = f"synthetic_train{meta['n_train']}"           # identifies the data for the store

results = {"POD": [], "AE": []}
ranks = [2, 4, 6, 8]
for r in ranks:
    results["POD"].append(POD(n_modes=r).fit(X_train).score(X_test) / test_var)
    ae = auto_load_or_create(AE, data=X_train, training_data_filename=tag,
                             n_latent=r, hidden=(128, 32), epochs=200, seed=0)
    results["AE"].append(ae.score(X_test) / test_var)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for name, errs in results.items():
    ax.semilogy(ranks, errs, "o-", label=name)
ax.set(xlabel="latent dimension r", ylabel="test NMSE")
ax.legend(); fig.savefig("projector_vs_r.png")
```


### B. Field estimation from sensors, end to end

```python
from datasets import decorrelation_lag, split_indices
from field_estimation import PODLSE, BranchedAE, LinearLatent, delay_embed, ridge_cv
from field_estimation.epod import pod as pod_svd

X = synthetic_field()
Q, fluid = grid_to_flat(X); Q = Q.astype(np.float64)
rng = np.random.default_rng(1)
S = Q[rng.choice(Q.shape[0], 5, replace=False)] ** 2 + 0.01 * rng.standard_normal((5, Q.shape[1]))

# 1. split with a gap and a warmup for the delay windows
n_delays = 10
Sd = delay_embed(S, n_delays=n_delays)
tr, _, te = split_indices(Q.shape[1], val_frac=0, test_frac=0.25,
                          gap=decorrelation_lag(X), warmup=n_delays - 1)

# 2. linear baseline, ridge chosen inside the training block
ridge, _ = ridge_cv(Q, Sd, tr, ridges=(0, 1e-4, 1e-2, 1.0), k=5, gap=20, r_field=8, r_sensor=20)
lse = PODLSE(r_field=8, r_sensor=20, ridge=ridge).fit(Q[:, tr], Sd[:, tr])

# 3. diagnose: basis limit vs sensor limit
print("floor", lse.floor(Q[:, te]), " rho2", np.round(lse.observability()["rho2"], 2))

# 4. nonlinear sensor map on the same basis, so any gain is attributable to G
Psi, _, _, q_mean = pod_svd(Q[:, tr], r=8, subtract_mean=True)
bae = BranchedAE(LinearLatent(Psi, q_mean), branch="mlp", n_delays=n_delays,
                 epochs=300, patience=30).fit(Q, S, tr)

print(f"POD-LSE    {lse.score(Q[:, te], Sd[:, te]):.3f}")
print(f"BranchedAE {bae.score(Q, S, te):.3f}")
```

Because the sensors here are quadratic, the MLP branch should beat the linear
estimator, and the gap is the value of the nonlinearity. On real data, if the
two are close, `rho2` usually tells you why.

---

## 11. Pitfalls and FAQ

`ModuleNotFoundError: No module named 'datasets'` -> install with
`pip install -e .`, or run from `src/`. Do not write `from src.datasets …`.

`FileNotFoundError: data/...` -> `DATA_ROOT` defaults to `./data` relative to
your *current directory*. Export `DATA_ROOT` as an absolute path.

Call `matplotlib.use("Agg")` before importing anything that imports pyplot.

`fit` fails on flat data -> fitting a projector needs the grid layout so it
can find the solid mask. After fitting, flat input is fine.

`ValueError: snapshots hold non-finite values at points that are fluid in
the first snapshot` -> the solid mask must be the same in every snapshot, and
is taken from the first one. Fill or crop any NaNs that come and go.

**`domain_of_measurement needs the grid's domain to be set`** — pass
`domain=[x0, x1, y0, y1]` when you build the ROM.

**PODLSE vs BranchedAE calling conventions** — `PODLSE.fit(Q[:, tr], Sd[:, tr])`
takes sliced blocks and a delay-embedded `Sd`. `BranchedAE.fit(Q, S, tr)` takes
the whole record, the raw `S`, and indices.
