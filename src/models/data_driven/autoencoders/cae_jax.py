from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from . import Projector
from .ae_jax import AEJax
from .jax_utils import (
    _ACT,
    _DTYPES,
    ActSpec,
    _act_fn,
    _epoch_batches,
    _resolve_acts,
    adam_init,
    adam_update,
)

__all__ = [
    "CAEJaxConfig",
    "CAEJax",
    "conv2d",
    "conv_transpose2d",
    "get_cae_params",
    "encode_grid",
    "decode_grid",
    "masked_mse_loss",
    "cae_train_step",
    "fit_cae_params",
]

# --- conv primitives ---
# torch Conv2d and XLA both cross-correlate and both use OIHW, so this is direct
DN = ("NCHW", "OIHW", "NCHW")


def conv2d(x: jax.Array, W: jax.Array, b: jax.Array, s: int, p: int) -> jax.Array:
    y = jax.lax.conv_general_dilated(x, W, window_strides=(s, s), padding=((p, p), (p, p)), dimension_numbers=DN)
    return y + b[None, :, None, None]


def conv_transpose2d(x: jax.Array, W: jax.Array, b: jax.Array, s: int, p: int, op: tuple) -> jax.Array:
    # lax.conv_transpose has no output_padding, so build it from the primitive.
    # a transposed conv is a fractionally strided conv: dilate the input by s,
    # then stride-1 convolve with the kernel flipped and the channel axes swapped
    k = W.shape[-1]
    V = jnp.flip(jnp.transpose(W, (1, 0, 2, 3)), axis=(2, 3))
    lo = k - 1 - p
    # output_padding only pads the bottom/right, hence the asymmetry
    y = jax.lax.conv_general_dilated(
        x,
        V,
        window_strides=(1, 1),
        padding=((lo, lo + op[0]), (lo, lo + op[1])),
        lhs_dilation=(s, s),
        dimension_numbers=DN,
    )
    return y + b[None, :, None, None]


# --- config ---
@dataclass(frozen=True)
class CAEJaxConfig:
    """
    Static architecture for the convolutional autoencoder.

    Same split as AEJaxConfig: only the fields that change the computation graph
    get hashed, the training hyperparams are marked as a field so that changing
    them never triggers a JIT recompile.
    """

    channels: tuple = (16, 32, 64)
    kernel_size: int = 3
    stride: int = 2
    pad: int = 1
    n_latent: int = 10
    activation: ActSpec = "tanh"
    layer_activations: Optional[ActSpec] = None  # optional per-layer override
    dtype: DTypeLike = jnp.float32
    weight_decay: float = 0.0
    grid_shape: Optional[tuple] = None  # (Nu, Nx, Ny), set by fit()

    learning_rate: float = field(default=1e-3, compare=False)
    n_epochs: int = field(default=500, compare=False)
    batch_size: int = field(default=32, compare=False)
    val_fraction: float = field(default=0.2, compare=False)
    patience: int = field(default=50, compare=False)
    lr_factor: float = field(default=0.5, compare=False)
    lr_patience: int = field(default=10, compare=False)
    min_lr: float = field(default=1e-6, compare=False)
    threshold: float = field(default=1e-4, compare=False)
    seed: int = field(default=0, compare=False)

    def __post_init__(self):
        # coerces lists -> tuples so the config stays hashable as a jit static arg
        object.__setattr__(self, "channels", tuple(self.channels))
        if self.grid_shape is not None:
            object.__setattr__(self, "grid_shape", tuple(self.grid_shape))

        # if we provided an activation string
        if not isinstance(self.activation, str) and not callable(self.activation):
            object.__setattr__(self, "activation", tuple(self.activation))

        # if we provided an activation tuple
        if (
            self.layer_activations is not None
            and not isinstance(self.layer_activations, str)
            and not callable(self.layer_activations)
        ):
            object.__setattr__(self, "layer_activations", tuple(self.layer_activations))

        for a in (*self.enc_acts, *self.dec_acts):
            if isinstance(a, str) and a not in _ACT:
                raise ValueError(f"unknown activation {a!r}; choose from {sorted(_ACT)}")
            if not isinstance(a, str) and not callable(a):
                raise TypeError(f"activation must be a name or a callable, got {a!r}")

    @property
    def enc_acts(self) -> tuple:
        # every conv stage is activated, the bottleneck Linear is not
        return _resolve_acts(self.activation, len(self.channels), "activation")

    @property
    def dec_acts(self) -> tuple:
        # the final transposed conv stays linear, so one fewer than the encoder
        spec = self.activation if self.layer_activations is None else self.layer_activations
        return _resolve_acts(spec, len(self.channels) - 1, "layer_activations")

    @property
    def enc_plan(self) -> tuple:
        "one entry per conv stage: (c_in, c_out, (h_in, w_in), (h_out, w_out))"
        if self.grid_shape is None:
            raise AttributeError("grid_shape is unknown until fit(); use replace(cfg, ...)")
        k, s, p = self.kernel_size, self.stride, self.pad
        c, h, w = self.grid_shape
        plan = []
        for c_out in self.channels:
            ho, wo = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
            plan.append((c, c_out, (h, w), (ho, wo)))
            c, h, w = c_out, ho, wo
        return tuple(plan)

    @property
    def red_shape(self) -> tuple:
        # (C, H, W) at the bottleneck, before the flatten
        return (self.enc_plan[-1][1], *self.enc_plan[-1][3])

    @property
    def dec_plan(self) -> tuple:
        "mirrors the encoder: (c_in, c_out, output_padding, target_hw)"
        k, s, p = self.kernel_size, self.stride, self.pad
        c, h, w = self.red_shape
        outs = list(self.channels[-2::-1]) + [self.grid_shape[0]]
        targets = [stage[2] for stage in self.enc_plan][::-1]
        plan = []
        for c_out, (th, tw) in zip(outs, targets):
            # op is always (H + 2p - k) mod s, so it is in [0, s) by construction
            op = (th - ((h - 1) * s - 2 * p + k), tw - ((w - 1) * s - 2 * p + k))
            plan.append((c, c_out, op, (th, tw)))
            c, h, w = c_out, th, tw
        return tuple(plan)

    @property
    def flat_dim(self) -> int:
        c, h, w = self.red_shape
        return c * h * w

    def as_float64(self) -> CAEJaxConfig:
        return replace(self, dtype=jnp.float64)


# --- init stuff ---
def _uniform(key, shape: tuple, fan_in: int, dtype: DTypeLike) -> jax.Array:
    # torch uses kaiming-uniform as default random init
    k = 1.0 / np.sqrt(fan_in)
    return jax.random.uniform(key, shape, dtype=dtype, minval=-k, maxval=k)


# gives you the conv stacks plus the two bottleneck Linear layers
def get_cae_params(key, cfg: CAEJaxConfig) -> dict:
    k, dt = cfg.kernel_size, cfg.dtype
    enc, dec = [], []
    for c_in, c_out, _, _ in cfg.enc_plan:
        key, w_key, b_key = jax.random.split(key, 3)
        # torch Conv2d weight is (C_out, C_in, k, k), so fan_in uses C_in
        fan = c_in * k * k
        enc.append(
            {
                "W": _uniform(w_key, (c_out, c_in, k, k), fan, dt),
                "b": _uniform(b_key, (c_out,), fan, dt),
            }
        )
    for c_in, c_out, _, _ in cfg.dec_plan:
        key, w_key, b_key = jax.random.split(key, 3)
        # torch ConvTranspose2d weight is (C_in, C_out, k, k), and fan_in reads
        # size(1), so it is C_out here and not C_in
        fan = c_out * k * k
        dec.append(
            {
                "W": _uniform(w_key, (c_in, c_out, k, k), fan, dt),
                "b": _uniform(b_key, (c_out,), fan, dt),
            }
        )
    key, w1, b1, w2, b2 = jax.random.split(key, 5)
    return {
        "enc": enc,
        "dec": dec,
        "enc_fc": {
            "W": _uniform(w1, (cfg.flat_dim, cfg.n_latent), cfg.flat_dim, dt),
            "b": _uniform(b1, (cfg.n_latent,), cfg.flat_dim, dt),
        },
        "dec_fc": {
            "W": _uniform(w2, (cfg.n_latent, cfg.flat_dim), cfg.n_latent, dt),
            "b": _uniform(b2, (cfg.flat_dim,), cfg.n_latent, dt),
        },
    }


# --- forward/loss ---
# conv stack down to the bottleneck, then flatten and one Linear to the latent
def encode_grid(params: dict, G: jax.Array, cfg: CAEJaxConfig) -> jax.Array:
    # G is (B, Nu, Nx, Ny)
    s, p = cfg.stride, cfg.pad
    h = G
    for layer, a in zip(params["enc"], cfg.enc_acts):
        h = _act_fn(a)(conv2d(h, layer["W"], layer["b"], s, p))
    h = h.reshape(h.shape[0], -1)
    return h @ params["enc_fc"]["W"] + params["enc_fc"]["b"]


# the transposed mirror. last conv stays linear so the reconstruction is unbounded
def decode_grid(params: dict, Z: jax.Array, cfg: CAEJaxConfig) -> jax.Array:
    s, p = cfg.stride, cfg.pad
    h = (Z @ params["dec_fc"]["W"] + params["dec_fc"]["b"]).reshape(-1, *cfg.red_shape)
    acts = cfg.dec_acts
    for i, (layer, stage) in enumerate(zip(params["dec"], cfg.dec_plan)):
        h = conv_transpose2d(h, layer["W"], layer["b"], s, p, stage[2])
        if i < len(acts):
            h = _act_fn(acts[i])(h)
    return h


def cae_forward(params: dict, G: jax.Array, cfg: CAEJaxConfig) -> jax.Array:
    return decode_grid(params, encode_grid(params, G, cfg), cfg)


def masked_mse_loss(params: dict, G: jax.Array, cfg: CAEJaxConfig, mask: jax.Array) -> jax.Array:
    # mask is (1, 1, Nx, Ny), so Nu has to come from G to make this a true mean
    recon = cae_forward(params, G, cfg)
    return ((recon - G) ** 2 * mask).sum() / (mask.sum() * G.shape[0] * G.shape[1])


# --- flat <-> grid with the solid mask ---
# idx is np.flatnonzero(fluid_mask_flat), a compile-time constant, so .at[].set()
# lowers to a static scatter
def flat_to_grid(Q, cfg: CAEJaxConfig, idx: np.ndarray) -> jax.Array:
    # (N_x, N_t) -> (N_t, Nu, Nx, Ny)
    Nu, Nx, Ny = cfg.grid_shape
    n_t = Q.shape[1]
    A = jnp.asarray(Q, dtype=cfg.dtype).reshape(idx.size, Nu, n_t).transpose(1, 2, 0)
    full = jnp.zeros((Nu, n_t, Nx * Ny), dtype=cfg.dtype).at[:, :, idx].set(A)
    return full.reshape(Nu, n_t, Nx, Ny).transpose(1, 0, 2, 3)


def grid_to_flat(G: jax.Array, cfg: CAEJaxConfig, idx: np.ndarray) -> np.ndarray:
    # (N_t, Nu, Nx, Ny) -> (N_x, N_t)
    Nu, Nx, Ny = cfg.grid_shape
    Gf = G.reshape(G.shape[0], Nu, Nx * Ny)[:, :, idx]
    return Gf.transpose(2, 1, 0).reshape(-1, G.shape[0])


# --- learning/training ---
@partial(jax.jit, static_argnames=("cfg", "wd"))
def cae_train_step(params, opt_state, batch, lr, cfg: CAEJaxConfig, mask, wd=0.0):
    # mask is traced, not static, it is data and it never changes shape
    loss, grads = jax.value_and_grad(masked_mse_loss)(params, batch, cfg, mask)
    params, opt_state = adam_update(grads, opt_state, params, lr, wd=wd)
    return params, opt_state, loss


@partial(jax.jit, static_argnames="cfg")
def _eval_masked_loss(params, G, cfg: CAEJaxConfig, mask):
    return masked_mse_loss(params, G, cfg, mask)


# same gather-inside-jit fix as train_step_at, for the conv model
@partial(jax.jit, static_argnames=("cfg", "wd"))
def cae_train_step_at(params, opt_state, G, batches, i, lr, cfg: CAEJaxConfig, mask, wd=0.0):
    return cae_train_step(params, opt_state, G[batches[i]], lr, cfg, mask, wd)


def fit_cae_params(G, cfg: CAEJaxConfig, mask):
    """Train on grid data (N_t, Nu, Nx, Ny). Returns (params, history)."""
    G = jnp.asarray(G, dtype=cfg.dtype)
    n_t = G.shape[0]
    n_val = int(round(cfg.val_fraction * n_t))
    G_tr, G_val = G[: n_t - n_val], G[n_t - n_val :]

    lr = cfg.learning_rate
    key = jax.random.key(cfg.seed)
    key, init_key = jax.random.split(key)
    params = get_cae_params(init_key, cfg)
    opt_state = adam_init(params)

    # early stopping and the scheduler keep independent counters
    best_params, best_val, wait = params, np.inf, 0
    sched_best, sched_bad = np.inf, 0
    history = {"train": [], "val": [], "lr": [], "n_epochs_run": 0}

    for i in range(cfg.n_epochs):
        key, shuffle_key = jax.random.split(key)
        batches = _epoch_batches(shuffle_key, G_tr.shape[0], cfg.batch_size)

        # see the note in fit_params: scan measured as no gain here either
        run = jnp.zeros((), dtype=G.dtype)  # accumulate on device
        for bi in range(batches.shape[0]):
            params, opt_state, loss = cae_train_step_at(
                params, opt_state, G_tr, batches, bi, lr, cfg, mask, cfg.weight_decay
            )
            run = run + loss
        history["train"].append(float(run) / batches.shape[0])  # one sync per epoch
        history["lr"].append(lr)
        history["n_epochs_run"] += 1

        if n_val == 0:
            continue

        v = float(_eval_masked_loss(params, G_val, cfg, mask))
        history["val"].append(v)

        # ReduceLROnPlateau, torch defaults: mode='min', relative threshold
        if v < sched_best * (1.0 - cfg.threshold):
            sched_best, sched_bad = v, 0
        else:
            sched_bad += 1
            if sched_bad > cfg.lr_patience:
                lr = max(lr * cfg.lr_factor, cfg.min_lr)
                sched_bad = 0

        # early stopping
        if v < best_val * (1.0 - cfg.threshold):
            best_val, wait, best_params = v, 0, params
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    return (best_params if n_val > 0 else params), history


# --- Projector interface ---
class CAEJax(Projector):
    """convolutional autoencoder.

    Conv2d encoder / ConvTranspose2d decoder on the 2-D grid, ending in a dense
    bottleneck. Follows Racca et al. (2021) and Ozalp et al. (2024), single-CAE
    variant.

    After ``fit(X)``:

        params           : dict            {'enc','dec','enc_fc','dec_fc'}
        cfg              : CAEJaxConfig       static architecture
        loss_history     : list[float]     mean training loss per epoch
        val_loss_history : list[float]
        lr_history       : list[float]     for diagnosing the schedule
        _scale           : (N_x, 1)        per-field input normalisation
        Q_mean           : (N_x, 1)        temporal mean (from the base)
    """

    _scale: Optional[np.ndarray] = None  # per-field input normalization (N_x, 1)
    params: Optional[dict] = None
    cfg: Optional[CAEJaxConfig] = None
    _idx: Optional[np.ndarray] = None  # flat indices of the fluid points
    _mask: Optional[jax.Array] = None  # (1, 1, Nx, Ny)

    _HOST_FIELDS = AEJax._HOST_FIELDS
    _GRAPH_FIELDS = (
        "channels",
        "kernel_size",
        "stride",
        "pad",
        "n_latent",
        "activation",
        "layer_activations",
        "dtype",
        "weight_decay",
    )

    def __init__(self, n_latent: int = 10, **kwargs):
        known = {f.name for f in fields(CAEJaxConfig)} - {"grid_shape"}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(f"CAE got unexpected options {sorted(unknown)}")
        self.cfg = CAEJaxConfig(n_latent=n_latent, **kwargs)
        self.loss_history: list = []
        self.val_loss_history: list = []
        self.lr_history: list = []
        self.n_epochs_run = 0

    def __repr__(self) -> str:
        c = self.cfg
        state = f"fitted, {self.n_epochs_run} epochs" if self.fitted else "unfitted"
        return (
            f"CAE(n_latent={c.n_latent}, channels={c.channels}, "
            f"k={c.kernel_size}, s={c.stride}, p={c.pad}, "
            f"activation={AEJax._name(c.activation)}, "
            f"grid_shape={c.grid_shape}, dtype={np.dtype(c.dtype).name}, {state})"
        )

    # --- config passthrough. cfg is the single owner, these are views ---
    def __getattr__(self, name: str):
        if name in CAEJax._HOST_FIELDS or name in CAEJax._GRAPH_FIELDS:
            return getattr(self.__dict__["cfg"], name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in CAEJax._HOST_FIELDS or name in CAEJax._GRAPH_FIELDS:
            if name in CAEJax._GRAPH_FIELDS and self.__dict__.get("params") is not None:
                raise AttributeError(f"{name!r} changes the architecture; construct a new CAEJax instead")
            self.__dict__["cfg"] = replace(self.__dict__["cfg"], **{name: value})
            return
        object.__setattr__(self, name, value)

    @property
    def N_latent(self) -> int:
        return self.cfg.n_latent

    @property
    def history(self) -> dict:
        return {
            "train": self.loss_history,
            "val": self.val_loss_history,
            "lr": self.lr_history,
            "n_epochs_run": self.n_epochs_run,
        }

    def _build_mask(self) -> None:
        Nu, Nx, Ny = self.cfg.grid_shape
        self._idx = np.flatnonzero(self.fluid_mask_flat)
        self._mask = jnp.asarray(self.fluid_mask_flat.reshape(Nx, Ny), dtype=self.cfg.dtype)[None, None]

    def fit(self, X: np.ndarray) -> CAEJax:
        Q = self.preprocess_snapshot(X)  # (N_x, N_t), zero-mean
        assert self.grid_shape is not None, "CAE needs raw grid input to fit."
        self._scale = self._field_scale(Q)  # per-field std (N_x, 1)
        object.__setattr__(self, "cfg", replace(self.cfg, grid_shape=self.grid_shape))
        self._build_mask()

        G = flat_to_grid(Q / self._scale, self.cfg, self._idx)  # (N_t, Nu, Nx, Ny)
        self.params, hist = fit_cae_params(G, self.cfg, self._mask)
        self.loss_history = hist["train"]
        self.val_loss_history = hist["val"]
        self.lr_history = hist["lr"]
        self.n_epochs_run = hist["n_epochs_run"]
        self.fitted = True
        return self

    @property
    def n_params(self) -> int:
        return int(sum(a.size for a in jax.tree.leaves(self.params)))

    @property
    def scale(self) -> np.ndarray:
        """The per-field scale inputs are divided by, shape `(N_x, 1)`."""
        if self._scale is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._scale

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self.params is None or self.cfg is None:
            raise RuntimeError("call fit() before encode()")
        Q = self.preprocess_snapshot(X)
        G = flat_to_grid(Q / self._scale, self.cfg, self._idx)
        return np.asarray(encode_grid(self.params, G, self.cfg)).T  # (N_latent, N_t)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        if self.params is None or self.cfg is None:
            raise RuntimeError("call fit() before decode()")

        Zt = jnp.asarray(np.asarray(Z).T, dtype=self.cfg.dtype)
        G = decode_grid(self.params, Zt, self.cfg)
        Q_hat = np.asarray(grid_to_flat(G, self.cfg, self._idx))
        return Q_hat * self._scale + self.Q_mean  # (N_x, N_t)

    def decode_scaled(self, Z: jax.Array) -> jax.Array:
        """Decodes a batch of latent codes differentiably onto the fluid points.

        Args:
          Z: Latent codes, shape `(B, N_latent)`.

        Returns:
          Fields in the scaled units the network trains on, shape `(B, N_x)`,
          in the flat layout `decode` returns and the dtype of the trained
          parameters.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if self.params is None or self.cfg is None or self._idx is None:
            raise RuntimeError("call fit() before decode_scaled()")
        dtype = jax.tree.leaves(self.params)[0].dtype
        G = decode_grid(self.params, jnp.asarray(Z, dtype), self.cfg)
        return grid_to_flat(G, self.cfg, self._idx).T

    def normalize(self, Q: np.ndarray) -> np.ndarray:
        return np.asarray(Q) / self._scale

    def denormalize(self, Q: np.ndarray) -> np.ndarray:
        return np.asarray(Q) * self._scale

    def score(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error over the fluid points."""
        Q = self.preprocess_snapshot(X)
        return float(np.mean((Q + self.Q_mean - self.reconstruct(X)) ** 2))

    def relative_error(self, X: np.ndarray) -> float:
        """score(X) as a fraction of the variance of the fluid-point data."""
        Q = self.preprocess_snapshot(X)
        return self.score(X) / float(np.mean(Q**2))

    def save(self, path) -> None:
        """Persist params, config and the preprocessing state to a .npz."""
        if self.params is None:
            raise RuntimeError("nothing to save; call fit() first")
        cfg = {f.name: getattr(self.cfg, f.name) for f in fields(CAEJaxConfig)}
        cfg["dtype"] = np.dtype(cfg["dtype"]).name
        for k in ("activation", "layer_activations"):
            if AEJax._name(cfg[k]) != cfg[k]:
                raise ValueError(f"cannot serialise callable in {k!r}; register a name")
        flat = jax.tree_util.tree_flatten(self.params)[0]
        arrays = {f"p{i}": np.asarray(a) for i, a in enumerate(flat)}
        np.savez(
            path,
            _cfg=np.array(json.dumps(cfg, default=list)),
            _scale=self._scale,
            _Q_mean=self.Q_mean,
            _grid_shape=np.array(self.grid_shape),
            _fluid_mask=self.fluid_mask_flat,
            _history=np.array(json.dumps(self.history)),
            **arrays,
        )

    @classmethod
    def load(cls, path) -> CAEJax:
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["_cfg"]))
        cfg["dtype"] = _DTYPES[cfg["dtype"]]
        for k in ("channels", "activation", "layer_activations", "grid_shape"):
            if isinstance(cfg[k], list):
                cfg[k] = tuple(cfg[k])
        grid_shape = cfg.pop("grid_shape")
        cae = cls(**cfg)
        object.__setattr__(cae, "cfg", replace(cae.cfg, grid_shape=grid_shape))
        cae._scale = z["_scale"]
        cae.Q_mean = z["_Q_mean"]
        cae.grid_shape = tuple(int(v) for v in z["_grid_shape"])
        cae.fluid_mask_flat = z["_fluid_mask"]
        cae._build_mask()
        hist = json.loads(str(z["_history"]))
        cae.loss_history = hist["train"]
        cae.val_loss_history = hist["val"]
        cae.lr_history = hist["lr"]
        cae.n_epochs_run = hist["n_epochs_run"]

        skeleton = get_cae_params(jax.random.key(0), cae.cfg)
        _, treedef = jax.tree_util.tree_flatten(skeleton)
        leaves = [jnp.asarray(z[f"p{i}"]) for i in range(len(jax.tree.leaves(skeleton)))]
        cae.params = jax.tree_util.tree_unflatten(treedef, leaves)
        cae.fitted = True
        return cae
