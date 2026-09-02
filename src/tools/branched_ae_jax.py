"""JAX implementation of the two-branch autoencoder's sensor branch.

Ports the operator `G` from `branched_ae.py`, which maps a causal window of
load-cell history to a latent code and trains against a frozen encoder and
decoder. This module relates to `branched_ae.py` as `autoencoders/ae_jax.py`
relates to `autoencoders.py`: parameters live in plain nested dicts, Adam is
written out explicitly, and `jax.jit` covers the whole training step, rather
than relying on a neural-network framework.

The API mirrors `BranchedAEJax.fit(Q, S, train_idx)`, `encode`, `predict`, and
`score`, so a JAX branch and a torch branch drop into the same comparison loop
and produce directly comparable numbers.

The sensor branch holds tens of thousands of parameters against the
autoencoder's tens of millions, so this module is not about the speed of a
single fit. It targets the sweep, which runs hundreds of small fits, and where
torch spends most of its time in Python launching kernels. Here each shape
compiles once and the epoch loop then runs on device.

Windows
-------
`sensor_windows` builds causal windows of shape (N_t, n_delays, n_channels),
oldest sample first, so window `t` covers samples
`t - (n_delays - 1) * stride` through `t`. Causality matters: a load cell
integrates the pressure field and therefore lags the flow, so a window reaching
forward in time would read the answer.

The first `(n_delays - 1) * stride` columns have partly zero-padded windows.
Exclude them from training and from scoring; `warmup` reports that count and
`fit` raises if `train_idx` includes them.

Losses
------
The default loss is `||G(s) - E(phi)||^2`, which for an orthonormal POD basis
equals the field MSE and costs much less for any other basis, because it needs
no decoder pass. Setting `lambda_field` above 0 adds the decoded field term,
which requires the decoder to be a JAX function; see `LinearLatentJax` and the
`decode_fn` argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .epod import apply_sensor_stats, sensor_stats, sensor_windows

__all__ = [
    "SensorBranchConfig",
    "BranchedAEJax",
    "LinearLatentJax",
    "sensor_windows",
    "init_branch",
    "branch_forward",
]

_ACT = {"tanh": jnp.tanh, "relu": jax.nn.relu, "elu": jax.nn.elu,
        "identity": lambda x: x, "gelu": jax.nn.gelu}


# ── windows ───────────────────────────────────────────────────────────────────




# ── parameters ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SensorBranchConfig:
    """Static architecture of the sensor branch `G`.

    Frozen, so instances hash and can serve as a `jit` static argument.

    Attributes:
        n_channels: Number of sensor channels.
        n_delays: Window length in samples.
        n_latent: Dimension of the latent space.
        kind: Architecture. One of:
            `"linear"`: A single affine map, which computes LSE on the
                delay-embedded record. Use it as the control.
            `"mlp"`: A multilayer perceptron over the flattened window.
            `"cnn"`: One-dimensional convolutions along time, sharing weights
                across lags.
            `"gru"`: A recurrent branch, and the only one that steps online a
                sample at a time.
        hidden: Hidden widths, for `kind="mlp"`.
        activation: Activation name. One of `tanh`, `relu`, `elu`, `gelu`,
            `identity`.
        cnn_channels: Channel widths, for `kind="cnn"`.
        kernel_size: Convolution kernel length, for `kind="cnn"`.
        gru_hidden: Hidden width, for `kind="gru"`.
        dtype: `"float32"` or `"float64"`.

    Raises:
        ValueError: If `kind` or `activation` is unknown.
    """

    n_channels: int
    n_delays: int
    n_latent: int
    kind: str = "mlp"
    hidden: tuple = (128, 128)
    activation: str = "tanh"
    cnn_channels: tuple = (32, 64)
    kernel_size: int = 5
    gru_hidden: int = 64
    dtype: str = "float32"

    def __post_init__(self):
        object.__setattr__(self, "hidden", tuple(self.hidden))
        object.__setattr__(self, "cnn_channels", tuple(self.cnn_channels))
        if self.kind not in ("linear", "mlp", "cnn", "gru"):
            raise ValueError(f"kind must be linear/mlp/cnn/gru, got {self.kind!r}")
        if self.activation not in _ACT:
            raise ValueError(f"activation must be one of {sorted(_ACT)}")

    @property
    def np_dtype(self):
        return jnp.float32 if self.dtype == "float32" else jnp.float64


def _dense(key, n_in, n_out, dtype):
    """Returns Glorot-uniform weights and a zero bias.

    Torch's `nn.Linear` default initialisation is close enough to this that both
    implementations start from comparable scales.
    """
    lim = jnp.sqrt(6.0 / (n_in + n_out))
    return {"W": jax.random.uniform(key, (n_in, n_out), dtype, -lim, lim),
            "b": jnp.zeros((n_out,), dtype)}


def init_branch(key, cfg: SensorBranchConfig) -> dict:
    """Builds the initial parameters for the sensor branch.

    Args:
        key: A `jax.random` PRNG key.
        cfg: Branch architecture.

    Returns:
        Parameters as a nested dict of arrays.
    """
    dt = cfg.np_dtype
    n_flat = cfg.n_channels * cfg.n_delays

    if cfg.kind == "linear":
        return {"head": [_dense(key, n_flat, cfg.n_latent, dt)]}

    if cfg.kind == "mlp":
        dims = [n_flat, *cfg.hidden, cfg.n_latent]
        keys = jax.random.split(key, len(dims) - 1)
        return {"head": [_dense(k, a, b, dt) for k, a, b in zip(keys, dims, dims[1:])]}

    if cfg.kind == "cnn":
        keys = jax.random.split(key, len(cfg.cnn_channels) + 1)
        conv, c = [], cfg.n_channels
        for k, c_out in zip(keys, cfg.cnn_channels):
            lim = jnp.sqrt(6.0 / (c * cfg.kernel_size + c_out * cfg.kernel_size))
            conv.append({"W": jax.random.uniform(
                k, (cfg.kernel_size, c, c_out), dt, -lim, lim),
                "b": jnp.zeros((c_out,), dt)})
            c = c_out
        return {"conv": conv,
                "head": [_dense(keys[-1], c * cfg.n_delays, cfg.n_latent, dt)]}

    # gru: one gate block (z, r, n) plus a linear readout
    k1, k2, k3 = jax.random.split(key, 3)
    h = cfg.gru_hidden
    lim = jnp.sqrt(1.0 / h)
    return {
        "gru": {
            "Wz": jax.random.uniform(k1, (cfg.n_channels, h), dt, -lim, lim),
            "Uz": jax.random.uniform(k1, (h, h), dt, -lim, lim),
            "bz": jnp.zeros((h,), dt),
            "bhz": jnp.zeros((h,), dt),
            "Wr": jax.random.uniform(k2, (cfg.n_channels, h), dt, -lim, lim),
            "Ur": jax.random.uniform(k2, (h, h), dt, -lim, lim),
            "br": jnp.zeros((h,), dt),
            "bhr": jnp.zeros((h,), dt),
            "Wn": jax.random.uniform(k3, (cfg.n_channels, h), dt, -lim, lim),
            "Un": jax.random.uniform(k3, (h, h), dt, -lim, lim),
            "bn": jnp.zeros((h,), dt),
            "bhn": jnp.zeros((h,), dt),
        },
        "head": [_dense(k3, h, cfg.n_latent, cfg.np_dtype)],
    }


# ── forward ───────────────────────────────────────────────────────────────────


def _mlp(head: list, x, act, last_linear=True):
    for i, layer in enumerate(head):
        x = x @ layer["W"] + layer["b"]
        if not (last_linear and i == len(head) - 1):
            x = act(x)
    return x


def _gru_scan(p, x):
    """Runs a GRU over a window and returns the final hidden state.

    Args:
        p: GRU parameters.
        x: Input windows, shape (B, L, C).

    Returns:
        The hidden state after the last step, shape (B, H).

    Scanning the window with `lax.scan` rather than a Python loop keeps the
    unrolled graph independent of `n_delays`, so a 100-step window compiles as
    quickly as a 5-step one.
    """
    # Reproduce torch's nn.GRU formulation, including its two bias vectors per
    # gate. For the z and r gates the pair is redundant, since their sum acts as
    # one bias, but for the candidate gate it is not: torch places the recurrent
    # bias inside the reset multiplication,
    #     n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    # so r gates b_hn and not b_in. Combining them into one bias outside the
    # product computes a different function rather than reparameterising this
    # one, and the two implementations would no longer be comparable.
    def step(h, x_t):
        z = jax.nn.sigmoid(x_t @ p["Wz"] + p["bz"] + h @ p["Uz"] + p["bhz"])
        r = jax.nn.sigmoid(x_t @ p["Wr"] + p["br"] + h @ p["Ur"] + p["bhr"])
        n = jnp.tanh(x_t @ p["Wn"] + p["bn"] + r * (h @ p["Un"] + p["bhn"]))
        return (1.0 - z) * n + z * h, None

    # Take the parameter dtype for h0 rather than the input's. lax.scan requires
    # the carry to hold one type across iterations, and the weights decide the
    # promotion inside `step`: float64 parameters lift a float32 h to float64 on
    # the first step, and the scan then fails to type-check. Seeding from
    # p["Wz"] matches the carry type to the body's output whatever precision
    # jax_enable_x64 has selected.
    h0 = jnp.zeros((x.shape[0], p["Wz"].shape[1]), p["Wz"].dtype)
    h, _ = jax.lax.scan(step, h0, jnp.swapaxes(x, 0, 1))  # (L, B, C)
    return h


def _param_dtype(params):
    return jax.tree_util.tree_leaves(params)[0].dtype


def branch_forward(params: dict, x: jax.Array, cfg: SensorBranchConfig) -> jax.Array:
    """Maps a batch of sensor windows to latent codes.

    No activation follows the output layer, so latent codes are unbounded and
    match the range of an AE bottleneck and of POD coefficients.

    This function casts the input to the parameter dtype first. Unlike NumPy and
    torch, JAX does not promote mixed precision: `lax.conv_general_dilated` and
    `lax.scan` reject a float32 and float64 mix rather than upcasting. Sensor
    windows arrive as float32, because they derive from float32 snapshot arrays,
    while `jax_enable_x64` makes the weights float64. Without the cast, every
    branch except the plain matmuls raises on its first step.

    Args:
        params: Branch parameters from `init_branch`.
        x: Sensor windows, shape (B, n_delays, n_channels).
        cfg: Branch architecture.

    Returns:
        Latent codes, shape (B, n_latent).
    """
    x = x.astype(_param_dtype(params))
    act = _ACT[cfg.activation]
    if cfg.kind in ("linear", "mlp"):
        return _mlp(params["head"], x.reshape(x.shape[0], -1), act)
    if cfg.kind == "cnn":
        h = x  # (B, L, C), conv along L with 'same' padding
        for layer in params["conv"]:
            h = jax.lax.conv_general_dilated(
                h, layer["W"], (1,), "SAME",
                dimension_numbers=("NWC", "WIO", "NWC")) + layer["b"]
            h = act(h)
        return _mlp(params["head"], h.reshape(h.shape[0], -1), act)
    return _mlp(params["head"], _gru_scan(params["gru"], x), act)


# ── latent space ──────────────────────────────────────────────────────────────


class LinearLatentJax:
    """Wraps a POD basis as an encoder/decoder pair, held on device.

    `decode_jax` is differentiable, so `lambda_field > 0` is supported. For an
    orthonormal basis the field and latent losses are the same objective up to a
    constant, so the field term adds cost without changing the result.

    Args:
        Psi: Spatial modes with orthonormal columns, shape (N_x, r).
        q_mean: Temporal mean, shape (N_x, 1).
        dtype: `"float32"` or `"float64"`.
    """

    def __init__(self, Psi: np.ndarray, q_mean: np.ndarray, dtype="float32"):
        dt = jnp.float32 if dtype == "float32" else jnp.float64
        self.Psi = jnp.asarray(Psi, dt)
        self.q_mean = jnp.asarray(q_mean, dt)
        self.n_latent = int(self.Psi.shape[1])

    def encode(self, Q):
        return np.asarray(self.Psi.T @ (jnp.asarray(Q, self.Psi.dtype) - self.q_mean))

    def decode(self, Z):
        return np.asarray(self.Psi @ jnp.asarray(Z, self.Psi.dtype) + self.q_mean)

    def decode_jax(self, Zt):
        """(B, r) -> (B, N_x), the differentiable path used by the field loss."""
        return Zt @ self.Psi.T + self.q_mean.T


# ── optimiser ─────────────────────────────────────────────────────────────────


class AdamState(NamedTuple):
    m: dict
    v: dict
    t: jax.Array


def adam_init(params) -> AdamState:
    z = jax.tree.map(jnp.zeros_like, params)
    return AdamState(m=z, v=jax.tree.map(jnp.zeros_like, params), t=jnp.zeros((), jnp.int32))


def adam_update(params, grads, st: AdamState, lr, wd=0.0, b1=0.9, b2=0.999, eps=1e-8):
    t = st.t + 1
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, st.m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, st.v, grads)
    mc = jax.tree.map(lambda x: x / (1 - b1**t), m)
    vc = jax.tree.map(lambda x: x / (1 - b2**t), v)
    new = jax.tree.map(lambda p, m_, v_: p - lr * (m_ / (jnp.sqrt(v_) + eps) + wd * p),
                       params, mc, vc)
    return new, AdamState(m=m, v=v, t=t)


def _global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(tree)))


def _clip(grads, max_norm):
    """Clips gradients by global norm, without a branch so `jit` can trace it.

    A GRU over a 100-step window occasionally produces a very large gradient on
    one batch. Unclipped, that step does not diverge visibly; it undoes the
    epoch's progress.
    """
    n = _global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (n + 1e-12))
    return jax.tree.map(lambda g: g * scale, grads)


# ── loss and step ─────────────────────────────────────────────────────────────


def _loss_fn(params, xb, zb, cfg, lam, decode_fn, fb):
    z_hat = branch_forward(params, xb, cfg)
    loss = jnp.mean((z_hat - zb) ** 2)
    if lam > 0.0 and decode_fn is not None:
        loss = loss + lam * jnp.mean((decode_fn(z_hat) - fb) ** 2)
    return loss


# Mark `lam` static: it decides whether the field term exists at all, which is a
# Python branch rather than an array operation. Traced instead, it would build
# the decoder pass even when lambda_field is 0, which is the default and the
# common case.
@partial(jax.jit, static_argnames=("cfg", "decode_fn", "lam", "wd", "clip"))
def train_step(params, opt_state, X, Z, F, batch, lr, cfg, lam, decode_fn,
               wd=0.0, clip=5.0):
    """Runs one Adam step on one batch.

    The gather stays inside `jit` by design. Indexing `X[batch]` outside `jit`
    repeats the gather on the host every step and copies the result to device;
    passing the index array instead lets XLA fuse it into the forward pass.
    Across the hundreds of small fits in a sweep, that difference dominates
    runtime.

    Args:
        params: Branch parameters.
        opt_state: Adam state.
        X: All sensor windows.
        Z: All latent targets.
        F: All field targets, or `None` when `lam` is 0.
        batch: Indices selecting this batch.
        lr: Learning rate.
        cfg: Branch architecture.
        lam: Field-loss weight. Static, so `lam=0` compiles without the decoder.
        decode_fn: Differentiable decoder, or `None`.
        wd: Weight decay.
        clip: Maximum global gradient norm.

    Returns:
        A tuple `(params, opt_state, loss)`.
    """
    xb, zb = X[batch], Z[batch]
    fb = F[batch] if F is not None else None
    loss, grads = jax.value_and_grad(_loss_fn)(params, xb, zb, cfg, lam, decode_fn, fb)
    grads = _clip(grads, clip)
    params, opt_state = adam_update(params, grads, opt_state, lr, wd)
    return params, opt_state, loss


@partial(jax.jit, static_argnames=("cfg", "decode_fn", "lam"))
def eval_loss(params, X, Z, F, cfg, lam, decode_fn):
    return _loss_fn(params, X, Z, cfg, lam, decode_fn, F)


def _epoch_batches(key, n: int, batch_size: int):
    """(n_batches, batch_size) index array. The remainder is dropped so every
    step sees one shape and jit compiles once, not once per ragged tail."""
    perm = jax.random.permutation(key, n)
    nb = n // batch_size
    return perm[: nb * batch_size].reshape(nb, batch_size)


# ── the estimator ─────────────────────────────────────────────────────────────


@dataclass
class BranchedAEJax:
    """Trains the sensor branch `G` against a frozen encoder/decoder pair, in JAX.

    Mirrors `branched_ae.BranchedAE`, including the `fit(Q, S, train_idx)`
    convention: pass the whole record and let the index array select targets.
    The windows are causal, so a pre-sliced block loses its history at the seam.

    Attributes:
        latent: A fitted `LinearLatentJax`, or any object exposing `encode`,
            `decode`, and `n_latent`, plus `decode_jax` when `lambda_field > 0`.
        branch: Branch architecture, one of `linear`, `mlp`, `cnn`, or `gru`.
        n_delays: Causal window length in samples.
        delay_stride: Sample spacing between consecutive lags.
        hidden: Hidden widths for an MLP branch.
        activation: Activation name.
        cnn_channels: Channel widths for a CNN branch.
        kernel_size: Convolution kernel length for a CNN branch.
        gru_hidden: Hidden width for a GRU branch.
        lambda_field: Weight on the field-space loss term.
        latent_weight: `"energy"` leaves latent targets unscaled; `"unit"`
            whitens them.
        learning_rate: Adam step size.
        weight_decay: L2 penalty applied by Adam.
        n_epochs: Maximum training epochs.
        batch_size: Minibatch size.
        val_fraction: Fraction of the training block held out for early
            stopping, taken contiguously from its end.
        patience: Epochs without improvement before stopping.
        threshold: Relative improvement required to reset `patience`.
        grad_clip: Maximum global gradient norm.
        seed: PRNG seed.
        dtype: `"float32"` or `"float64"`.
        verbose: Whether to print per-epoch losses.
    """

    latent: object
    branch: str = "mlp"
    n_delays: int = 25
    delay_stride: int = 1

    hidden: Sequence[int] = (128, 128)
    activation: str = "tanh"
    cnn_channels: Sequence[int] = (32, 64)
    kernel_size: int = 5
    gru_hidden: int = 64

    lambda_field: float = 0.0
    latent_weight: str = "energy"

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    n_epochs: int = 500
    batch_size: int = 128
    val_fraction: float = 0.2
    patience: int = 50
    threshold: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-6
    grad_clip: float = 5.0
    seed: int = 0
    dtype: str = "float32"
    verbose: bool = False

    params: Optional[dict] = field(default=None, repr=False)
    cfg: Optional[SensorBranchConfig] = field(default=None, repr=False)
    s_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    z_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    loss_history: list = field(default_factory=list, repr=False)
    val_loss_history: list = field(default_factory=list, repr=False)
    fitted: bool = False

    @property
    def warmup(self) -> int:
        """Number of leading columns whose window is partly zero padding.

        Exclude these indices from every index array passed to this class.
        """
        return (self.n_delays - 1) * self.delay_stride

    # -- api -------------------------------------------------------------------

    def fit(self, Q: np.ndarray, S: np.ndarray, train_idx: np.ndarray) -> BranchedAEJax:
        Q, S = np.asarray(Q, np.float64), np.asarray(S, np.float64)
        train_idx = np.asarray(train_idx)
        if Q.shape[1] != S.shape[1]:
            raise ValueError(f"field has {Q.shape[1]} snapshots, sensors {S.shape[1]}; "
                             "these must be synchronised sample-for-sample")
        if train_idx.min() < self.warmup:
            raise ValueError(
                f"train_idx starts at {train_idx.min()} but the first "
                f"{self.warmup} columns have partly-padded windows. Drop them.")

        dt = jnp.float32 if self.dtype == "float32" else jnp.float64
        # Standardise per channel, using the training block only. Forces are in
        # newtons and moments in newton-metres, so without this the branch
        # receives one set of inputs a thousand times larger than the other.
        self.s_mean, self.s_scale = sensor_stats(S[:, train_idx])

        W = sensor_windows(apply_sensor_stats(S, self.s_mean, self.s_scale),
                           self.n_delays, self.delay_stride)
        Z = self.latent.encode(Q).T  # (N_t, r)
        self.z_scale = (np.std(Z[train_idx], axis=0, keepdims=True)
                        if self.latent_weight == "unit"
                        else np.ones((1, Z.shape[1])))
        self.z_scale = np.where(self.z_scale > 0, self.z_scale, 1.0)

        Xj = jnp.asarray(W, dt)
        Zj = jnp.asarray(Z / self.z_scale, dt)
        Fj = jnp.asarray(Q.T, dt) if self.lambda_field > 0 else None
        decode_fn = getattr(self.latent, "decode_jax", None) if self.lambda_field > 0 else None

        # Take validation from the tail of the training block, contiguously. A
        # random validation split inside a 250 Hz record measures interpolation,
        # which produces a leaked early-stopping score.
        n_val = int(round(self.val_fraction * len(train_idx)))
        tr_i = jnp.asarray(train_idx[: len(train_idx) - n_val])
        va_i = jnp.asarray(train_idx[len(train_idx) - n_val:]) if n_val else None

        self.cfg = SensorBranchConfig(
            n_channels=S.shape[0], n_delays=self.n_delays,
            n_latent=int(Z.shape[1]), kind=self.branch, hidden=tuple(self.hidden),
            activation=self.activation, cnn_channels=tuple(self.cnn_channels),
            kernel_size=self.kernel_size, gru_hidden=self.gru_hidden, dtype=self.dtype)

        key = jax.random.PRNGKey(self.seed)
        key, k0 = jax.random.split(key)
        params = init_branch(k0, self.cfg)
        opt = adam_init(params)

        lr = self.learning_rate
        best, best_params, wait, lr_wait = jnp.inf, params, 0, 0
        self.loss_history, self.val_loss_history = [], []
        n_tr = int(tr_i.shape[0])
        bs = min(self.batch_size, n_tr)

        for ep in range(self.n_epochs):
            key, ks = jax.random.split(key)
            batches = tr_i[_epoch_batches(ks, n_tr, bs)]
            run = 0.0
            for i in range(batches.shape[0]):
                params, opt, loss = train_step(
                    params, opt, Xj, Zj, Fj, batches[i], lr, self.cfg,
                    self.lambda_field, decode_fn, self.weight_decay, self.grad_clip)
                run += float(loss)
            self.loss_history.append(run / max(batches.shape[0], 1))

            if va_i is not None:
                v = float(eval_loss(params, Xj[va_i], Zj[va_i],
                                    Fj[va_i] if Fj is not None else None,
                                    self.cfg, self.lambda_field, decode_fn))
                self.val_loss_history.append(v)
                if v < best * (1.0 - self.threshold):
                    best, best_params, wait, lr_wait = v, params, 0, 0
                else:
                    wait += 1
                    lr_wait += 1
                    # ReduceLROnPlateau and early stopping on independent
                    # counters, as in the torch version -- patience must cover
                    # every decay down to min_lr or min_lr is dead config
                    if lr_wait >= self.lr_patience and lr > self.min_lr:
                        lr = max(lr * self.lr_factor, self.min_lr)
                        lr_wait = 0
                        if self.verbose:
                            print(f"    epoch {ep}: lr -> {lr:.2e}")
                    if wait >= self.patience:
                        break

        self.params = best_params if va_i is not None else params
        self.fitted = True
        return self

    def encode(self, S: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicted latent coefficients (r, n). The quantity POD-LSE's
        ``encode`` returns, so the two are directly comparable."""
        self._check()
        S = np.asarray(S, np.float64)
        W = sensor_windows(apply_sensor_stats(S, self.s_mean, self.s_scale),
                           self.n_delays, self.delay_stride)
        if idx is not None:
            W = W[np.asarray(idx)]
        dt = jnp.float32 if self.dtype == "float32" else jnp.float64
        Z = np.asarray(branch_forward(self.params, jnp.asarray(W, dt), self.cfg))
        return (Z * self.z_scale).T

    def predict(self, S: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        return self.latent.decode(self.encode(S, idx))

    def score(self, Q, S, idx=None) -> float:
        from .epod import nmse

        Q = np.asarray(Q, np.float64)
        Qt = Q[:, idx] if idx is not None else Q
        return nmse(Qt, self.predict(S, idx))

    def latent_score(self, Q, S, idx) -> float:
        """NMSE in latent space -- how well ``G`` recovers ``E(phi)``.

        Worth reading before the field score: if this is near 1 the branch is
        predicting the latent mean, and no decoder can rescue that.
        """
        from .epod import nmse

        Z = self.latent.encode(np.asarray(Q, np.float64))[:, idx]
        return nmse(Z, self.encode(S, idx))

    @property
    def n_params(self) -> int:
        return int(sum(x.size for x in jax.tree.leaves(self.params))) if self.params else 0

    def _check(self):
        if not self.fitted:
            raise RuntimeError("call fit() before encode()/predict()/score()")
