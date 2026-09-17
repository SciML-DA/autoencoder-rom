from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from . import Projector
from .jax_utils import (
    _ACT,
    _DTYPES,
    ActSpec,
    AdamState,
    _act_fn,
    _epoch_batches,
    _resolve_acts,
    adam_init,
    adam_update,
)

__all__ = [
    "AEJaxConfig",
    "AdamState",
    "AEJax",
    "get_ae_params",
    "get_mlp_params",
    "forward",
    "mse_loss",
    "adam_init",
    "adam_update",
    "train_step",
    "fit_params",
]


# --- config ---
@dataclass(frozen=True)
class AEJaxConfig:
    """
    Static architecture. Frozen so it is hashable and safe as a jit static arg.

    Only fields that change the computation graph should get hashed.
    The training hyperparams are host-side and need to be excluded from hashing
    as altering their value shouldn't cause a JIT recompile (so we mark as a field)
    """

    hidden: tuple = (512, 128)
    n_latent: int = 10
    activation: ActSpec = "tanh"
    layer_activations: Optional[ActSpec] = None  # optional per-layer override
    dtype: DTypeLike = jnp.float32
    weight_decay: float = 0.0
    n_x: Optional[int] = None

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
        object.__setattr__(self, "hidden", tuple(self.hidden))

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
    def n_act_layers(self) -> int:
        return len(self.hidden)

    @property
    def enc_dims(self) -> tuple:
        if self.n_x is None:
            raise AttributeError("n_x is unknown until fit(); use replace(cfg, n_x=...)")
        return (self.n_x, *self.hidden, self.n_latent)

    @property
    def dec_dims(self) -> tuple:
        return self.enc_dims[::-1]

    @property
    def enc_acts(self) -> tuple:
        return _resolve_acts(self.activation, self.n_act_layers, "activation")

    @property
    def dec_acts(self) -> tuple:
        spec = self.activation if self.layer_activations is None else self.layer_activations
        return _resolve_acts(spec, self.n_act_layers, "layer_activations")

    def as_float64(self) -> AEJaxConfig:
        return replace(self, dtype=jnp.float64)


# --- init stuff ---
# gives you the big params list with the full untrained model
def get_ae_params(key, cfg: AEJaxConfig) -> dict:
    # - n_x - *hidden - n_latent is the full model
    enc_key, dec_key = jax.random.split(key)
    return {
        "enc": get_mlp_params(enc_key, cfg.enc_dims, cfg.dtype),
        "dec": get_mlp_params(dec_key, cfg.dec_dims, cfg.dtype),
    }


# this gets vectorized later over a list of keys, so it needs to be vmappable
def get_mlp_params(key, dims: tuple, dtype: DTypeLike) -> list[dict]:
    "dims is (n_x, *hidden, n_latent)"
    layers = list()
    for i in range(len(dims) - 1):
        key, w_key, b_key = jax.random.split(key, 3)
        # layer dims
        n_in, n_out = dims[i], dims[i + 1]
        # torch uses kaiming-uniform as default random init
        k = 1.0 / np.sqrt(n_in)
        layers.append(
            {
                "W": jax.random.uniform(w_key, (n_in, n_out), dtype=dtype, minval=-k, maxval=k),
                "b": jax.random.uniform(b_key, (n_out,), dtype=dtype, minval=-k, maxval=k),
            }
        )
    return layers


# --- forward/loss ---
# single forward pass over the layers in the params dict
def _mlp_forward(params: list[dict], x: jax.Array, acts: tuple) -> jax.Array:
    # x is (B, n_x). acts has one entry per activated layer
    h = x
    # last layer is linear. W is (n_in, n_out)
    for layer, a in zip(params[:-1], acts):
        h = _act_fn(a)(h @ layer["W"] + layer["b"])
    return h @ params[-1]["W"] + params[-1]["b"]


# encode/decode result. calls _mlp_forward rather than the jitted wrappers so
# train_step traces to one flat graph.
def forward(params: dict, x: jax.Array, cfg: AEJaxConfig) -> jax.Array:
    encoder = params["enc"]  # first bit
    encoder_acts = cfg.enc_acts
    decoder = params["dec"]  # second bit
    decoder_acts = cfg.dec_acts
    encoded = _mlp_forward(encoder, x, encoder_acts)
    return _mlp_forward(decoder, encoded, decoder_acts)


def mse_loss(params: dict, x: jax.Array, cfg: AEJaxConfig) -> jax.Array:
    # params is the network, x is (B, n_x)
    return jnp.mean((forward(params, x, cfg) - x) ** 2)


# --- learning/training ---
@partial(jax.jit, static_argnames=("cfg", "wd"))
def train_step(params, opt_state, batch, lr, cfg: AEJaxConfig, wd=0.0):
    # wd is static because adam_update branches on it
    loss, grads = jax.value_and_grad(mse_loss)(params, batch, cfg)
    params, opt_state = adam_update(grads, opt_state, params, lr, wd=wd)
    return params, opt_state, loss


@partial(jax.jit, static_argnames="cfg")
def _eval_loss(params, x, cfg: AEJaxConfig):
    return mse_loss(params, x, cfg)


# `X_tr[idx]` runs *outside* jit and re-derives the same gather on every step;
# profiling put ~530 ms in jnp's Python indexing machinery because of it. This
# wrapper takes the whole array plus a row of the batch index and does the
# gather inside the traced function, so it is built once at compile time.
# Same arithmetic on the same samples in the same order.
@partial(jax.jit, static_argnames=("cfg", "wd"))
def train_step_at(params, opt_state, X, batches, i, lr, cfg: AEJaxConfig, wd=0.0):
    return train_step(params, opt_state, X[batches[i]], lr, cfg, wd)


def fit_params(data, cfg: AEJaxConfig):
    """Train on already-normalised data (N_t, n_x). Returns (params, history).

    Early stopping and the LR schedule stay in Python on purpose: they run once
    per epoch on scalars and have genuine data-dependent control flow, so
    pushing them into the jaxpr with lax.cond buys nothing. Only ``lr`` crosses
    the boundary, as a traced argument -- close over it and it gets baked in as
    a constant and silently goes stale.
    """
    data = jnp.asarray(data, dtype=cfg.dtype)
    n_t = data.shape[0]
    n_val = int(round(cfg.val_fraction * n_t))
    X_tr, X_val = data[: n_t - n_val], data[n_t - n_val :]

    lr = cfg.learning_rate
    key = jax.random.key(cfg.seed)
    key, init_key = jax.random.split(key)
    params = get_ae_params(init_key, cfg)
    opt_state = adam_init(params)

    # early stopping and the scheduler keep independent counters, same as torch
    best_params, best_val, wait = params, np.inf, 0
    sched_best, sched_bad = np.inf, 0
    history = {"train": [], "val": [], "lr": [], "n_epochs_run": 0}

    for i in range(cfg.n_epochs):
        key, shuffle_key = jax.random.split(key)
        batches = _epoch_batches(shuffle_key, X_tr.shape[0], cfg.batch_size)

        # lax.scan over `batches` was tried here (one XLA call per epoch rather
        # than one dispatch per batch) and measured inside the +-3% noise floor.
        # With the gather already inside jit via train_step_at, the remaining
        # per-step dispatch is cheap, so the plain loop stays.
        run = jnp.zeros((), dtype=data.dtype)  # accumulate on device
        for bi in range(batches.shape[0]):
            params, opt_state, loss = train_step_at(params, opt_state, X_tr, batches, bi, lr, cfg, cfg.weight_decay)
            run = run + loss
        history["train"].append(float(run) / batches.shape[0])  # one sync per epoch
        history["lr"].append(lr)
        history["n_epochs_run"] += 1

        if n_val == 0:
            continue

        v = float(_eval_loss(params, X_val, cfg))
        history["val"].append(v)

        # ReduceLROnPlateau, torch defaults: mode='min', relative threshold
        if v < sched_best * (1.0 - cfg.threshold):
            sched_best, sched_bad = v, 0
        else:
            sched_bad += 1
            if sched_bad > cfg.lr_patience:
                lr = max(lr * cfg.lr_factor, cfg.min_lr)
                sched_bad = 0

        # early stopping. params are immutable, so this IS the snapshot -- no
        # deepcopy of a state_dict needed.
        if v < best_val * (1.0 - cfg.threshold):
            best_val, wait, best_params = v, 0, params
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    return (best_params if n_val > 0 else params), history


# --- Projector interface ---
class AEJax(Projector):
    """fully-connected autoencoder.

    Functions the same as the PyTorch ``AE``
    ``preprocess_snapshot`` removes the NaN solid mask and subtracts temporal mean.
    ``_scale`` divides each field by its own std so none are underweighted in MSE
    RNG is seeded inside ``fit`` because JAX is weird, so calling fit twice
    gives identical weights (this gives more control, previous implementation had the
    the fit be dependent on how many epochs the previous ran)

    Ideally after calling __init__, the internal state of the AE
    is never altered, all that is altered

    After ``fit(X)``:

        params           : dict            {'enc': [...], 'dec': [...]}
        cfg              : AEJaxConfig        static architecture
        loss_history     : list[float]     mean training loss per epoch
        val_loss_history : list[float]
        lr_history       : list[float]     for diagnosing the schedule
        _scale           : (N_x, 1)        per-field input normalisation
        Q_mean           : (N_x, 1)        temporal mean (from the base)
    """

    _scale: Optional[np.ndarray] = None  # per-field input normalization (N_x, 1)
    params: Optional[dict] = None
    cfg: Optional[AEJaxConfig] = None

    _HOST_FIELDS = (
        "learning_rate",
        "n_epochs",
        "batch_size",
        "val_fraction",
        "patience",
        "lr_factor",
        "lr_patience",
        "min_lr",
        "threshold",
        "seed",
    )
    _GRAPH_FIELDS = (
        "hidden",
        "n_latent",
        "activation",
        "layer_activations",
        "dtype",
        "weight_decay",
    )

    def __init__(self, n_latent: int = 10, **kwargs):
        known = {f.name for f in fields(AEJaxConfig)} - {"n_x"}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(f"AE got unexpected options {sorted(unknown)}")
        self.cfg = AEJaxConfig(n_latent=n_latent, **kwargs)
        self.loss_history: list = []
        self.val_loss_history: list = []
        self.lr_history: list = []
        self.n_epochs_run = 0

    def __repr__(self) -> str:
        c = self.cfg
        state = f"fitted, {self.n_epochs_run} epochs" if self.fitted else "unfitted"
        return (
            f"AE(n_latent={c.n_latent}, hidden={c.hidden}, "
            f"activation={self._name(c.activation)}, "
            f"layer_activations={self._name(c.layer_activations)}, "
            f"n_x={c.n_x}, dtype={np.dtype(c.dtype).name}, {state})"
        )

    @staticmethod
    def _name(a):
        if a is None or isinstance(a, str):
            return a
        if callable(a):
            return getattr(a, "__name__", repr(a))
        return tuple(AEJax._name(x) for x in a)

    @staticmethod
    def register_activation(name: str, fn: Callable) -> None:
        _ACT[name] = fn

    # --- config passthrough. cfg is the single owner, these are views ---
    def __getattr__(self, name: str):
        if name in AEJax._HOST_FIELDS or name in AEJax._GRAPH_FIELDS:
            return getattr(self.__dict__["cfg"], name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in AEJax._HOST_FIELDS or name in AEJax._GRAPH_FIELDS:
            if name in AEJax._GRAPH_FIELDS and self.__dict__.get("params") is not None:
                raise AttributeError(f"{name!r} changes the architecture; construct a new AEJax instead")
            self.__dict__["cfg"] = replace(self.__dict__["cfg"], **{name: value})
            return
        object.__setattr__(self, name, value)

    @property
    def N_latent(self) -> int:
        return self.cfg.n_latent

    @property
    def layer_dims(self) -> tuple:
        return self.cfg.hidden

    @property
    def activation_function(self) -> ActSpec:
        return self.cfg.activation

    @property
    def history(self) -> dict:
        return {
            "train": self.loss_history,
            "val": self.val_loss_history,
            "lr": self.lr_history,
            "n_epochs_run": self.n_epochs_run,
        }

    def fit(self, X: np.ndarray) -> AEJax:
        Q = self.preprocess_snapshot(X)  # (N_x, N_t), zero-mean
        self._scale = self._field_scale(Q)  # per-field std (N_x, 1)
        object.__setattr__(self, "cfg", replace(self.cfg, n_x=Q.shape[0]))

        self.params, hist = fit_params(
            (Q / self._scale).T,  # (N_t, N_x)
            self.cfg,
        )
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
        Zt = jnp.asarray((Q / self._scale).T, dtype=self.cfg.dtype)
        return np.asarray(_mlp_forward(self.params["enc"], Zt, self.cfg.enc_acts)).T  # (N_latent, N_t)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        if self.params is None or self.cfg is None:
            raise RuntimeError("call fit() before decode()")

        Zt = jnp.asarray(np.asarray(Z).T, dtype=self.cfg.dtype)
        Q_hat = _mlp_forward(self.params["dec"], Zt, self.cfg.dec_acts)
        return np.asarray(Q_hat).T * self._scale + self.Q_mean  # (N_x, N_t)

    def decode_scaled(self, Z: jax.Array) -> jax.Array:
        """Decodes a batch of latent codes differentiably.

        Args:
          Z: Latent codes, shape `(B, N_latent)`.

        Returns:
          Fields in the scaled units the network trains on, shape `(B, N_x)`,
          in the dtype of the trained parameters.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if self.params is None or self.cfg is None:
            raise RuntimeError("call fit() before decode_scaled()")
        dtype = jax.tree.leaves(self.params)[0].dtype
        return _mlp_forward(self.params["dec"], jnp.asarray(Z, dtype), self.cfg.dec_acts)

    def normalize(self, Q: np.ndarray) -> np.ndarray:
        return np.asarray(Q) / self._scale

    def denormalize(self, Q: np.ndarray) -> np.ndarray:
        return np.asarray(Q) * self._scale

    def score(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error. Accepts raw grid or flat input.

        Overrides the base, which subtracts a (N_x, N_t) reconstruction from a
        (Nu, N_t, Nx, Ny) input and raises a broadcast error.
        """
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
        cfg = {f.name: getattr(self.cfg, f.name) for f in fields(AEJaxConfig)}
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
    def load(cls, path) -> AEJax:
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["_cfg"]))
        cfg["dtype"] = _DTYPES[cfg["dtype"]]
        for k in ("hidden", "activation", "layer_activations"):
            if isinstance(cfg[k], list):
                cfg[k] = tuple(cfg[k])
        n_x = cfg.pop("n_x")
        ae = cls(**cfg)
        object.__setattr__(ae, "cfg", replace(ae.cfg, n_x=n_x))
        ae._scale = z["_scale"]
        ae.Q_mean = z["_Q_mean"]
        ae.grid_shape = tuple(int(v) for v in z["_grid_shape"])
        ae.fluid_mask_flat = z["_fluid_mask"]
        hist = json.loads(str(z["_history"]))
        ae.loss_history = hist["train"]
        ae.val_loss_history = hist["val"]
        ae.lr_history = hist["lr"]
        ae.n_epochs_run = hist["n_epochs_run"]

        skeleton = get_ae_params(jax.random.key(0), ae.cfg)
        _, treedef = jax.tree_util.tree_flatten(skeleton)
        leaves = [jnp.asarray(z[f"p{i}"]) for i in range(len(jax.tree.leaves(skeleton)))]
        ae.params = jax.tree_util.tree_unflatten(treedef, leaves)
        ae.fitted = True
        return ae
