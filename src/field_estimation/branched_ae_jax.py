"""JAX implementation of the two-branch autoencoder's sensor branch.

Ports the sensor branch `G` from `branched_ae.py`, which maps a causal window of
sensor history to a latent code and trains against a frozen encoder and
decoder. Parameters live in nested dictionaries, Adam is written out explicitly,
and `jax.jit` compiles the whole training step.

`BranchedAEJax` mirrors `branched_ae.BranchedAE`: `fit(Q, S, train_idx)`,
`encode`, `predict`, and `score` take the same arguments and return comparable
values.

`sensor_windows` builds windows of shape `(N_t, n_delays, n_channels)`, oldest
sample first, so window `t` covers samples `t - (n_delays - 1) * stride` through
`t`. The first `warmup` snapshots have partly zero-padded windows; `fit` raises
if `train_idx` includes them.

The loss is

    L = ||G(s) - E(phi)||^2 + lambda_field * ||D(G(s)) - phi||^2

where `E` and `D` are the latent space's `encode` and `decode_jax`.

Typical usage example:

  from field_estimation.branched_ae_jax import BranchedAEJax, LinearLatentJax
  from datasets import split_indices
  from field_estimation.epod import pod

  tr, _, te = split_indices(Q.shape[1], val_frac=0, test_frac=0.25, gap=100, warmup=24)
  Psi, _, _, q_mean = pod(Q[:, tr], r=64, subtract_mean=True)
  model = BranchedAEJax(LinearLatentJax(Psi, q_mean), branch="mlp").fit(Q, S, tr)
  model.score(Q, S, te)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, NamedTuple, NotRequired, Protocol, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from .epod import FloatArray, IndexArray, apply_sensor_stats, nmse, sensor_stats, sensor_windows

__all__ = [
    "SensorBranchConfig",
    "BranchedAEJax",
    "LinearLatentJax",
    "AutoencoderLatentJax",
    "sensor_windows",
    "init_branch",
    "branch_forward",
]

#: Precision the sensor branch computes in.
Dtype = Literal["float32", "float64"]
#: An elementwise activation.
Activation = Callable[[jax.Array], jax.Array]
#: A differentiable decoder from latent codes `(B, r)` to fields `(B, N_x)`.
DecodeFn = Callable[[jax.Array], jax.Array]


class Layer(TypedDict):
    """Weights and bias of one dense or convolutional layer."""

    W: jax.Array
    b: jax.Array


class BranchParams(TypedDict):
    """Parameters of a sensor branch.

    Attributes:
      head: Dense layers, applied last.
      conv: Convolutional layers, present for a `"cnn"` branch.
      gru: Gate weights and biases, present for a `"gru"` branch.
    """

    head: list[Layer]
    conv: NotRequired[list[Layer]]
    gru: NotRequired[dict[str, jax.Array]]


def _identity(x: jax.Array) -> jax.Array:
    """Returns the input unchanged.

    Args:
      x: Any array.

    Returns:
      `x`.
    """
    return x


_ACT: dict[str, Activation] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "identity": _identity,
    "gelu": jax.nn.gelu,
}


def _scalar_type(dtype: Dtype) -> DTypeLike:
    """Maps a precision name onto the JAX scalar type it selects.

    Args:
      dtype: `"float32"` or `"float64"`.

    Returns:
      The corresponding `jnp` scalar type.
    """
    return jnp.float32 if dtype == "float32" else jnp.float64


# ── Parameters ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SensorBranchConfig:
    """Static architecture of the sensor branch.

    Instances are frozen and hashable, so they can serve as a static argument to
    `jax.jit`.

    Attributes:
      n_channels: Number of sensor channels.
      n_delays: Window length in samples.
      n_latent: Dimension of the latent space.
      kind: Architecture. One of:

        - `"linear"`: A single affine map on the flattened window.
        - `"mlp"`: A multilayer perceptron over the flattened window.
        - `"cnn"`: One-dimensional convolutions along time.
        - `"gru"`: A recurrent network that reads out its final hidden state.

      hidden: Hidden widths, for `kind="mlp"`.
      activation: Activation name. One of `"tanh"`, `"relu"`, `"elu"`, `"gelu"`,
        or `"identity"`.
      cnn_channels: Channel widths, for `kind="cnn"`.
      kernel_size: Convolution kernel length, for `kind="cnn"`.
      gru_hidden: Hidden state width, for `kind="gru"`.
      dtype: Precision of the parameters.

    Raises:
      ValueError: If `kind` or `activation` is not one of the listed names.
    """

    n_channels: int
    n_delays: int
    n_latent: int
    kind: str = "mlp"
    hidden: tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    cnn_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 5
    gru_hidden: int = 64
    dtype: Dtype = "float32"

    def __post_init__(self) -> None:
        """Converts the width sequences to tuples and validates the names.

        Raises:
          ValueError: If `kind` or `activation` is not one of the listed names.
        """
        object.__setattr__(self, "hidden", tuple(self.hidden))
        object.__setattr__(self, "cnn_channels", tuple(self.cnn_channels))

        if self.kind not in ("linear", "mlp", "cnn", "gru"):
            raise ValueError(f"kind must be linear/mlp/cnn/gru, got {self.kind!r}")
        if self.activation not in _ACT:
            raise ValueError(f"activation must be one of {sorted(_ACT)}")

    @property
    def np_dtype(self) -> DTypeLike:
        """The JAX scalar type the parameters use."""
        return _scalar_type(self.dtype)


def _dense(key: jax.Array, n_in: int, n_out: int, dtype: DTypeLike) -> Layer:
    """Initializes a dense layer with Glorot-uniform weights and a zero bias.

    Args:
      key: A `jax.random` PRNG key.
      n_in: Input width.
      n_out: Output width.
      dtype: Scalar type of the parameters.

    Returns:
      The layer's weights, shape `(n_in, n_out)`, and bias, shape `(n_out,)`.
    """
    lim = jnp.sqrt(6.0 / (n_in + n_out))
    return {
        "W": jax.random.uniform(key, (n_in, n_out), dtype, -lim, lim),
        "b": jnp.zeros((n_out,), dtype),
    }


def init_branch(key: jax.Array, cfg: SensorBranchConfig) -> BranchParams:
    """Builds the initial parameters for a sensor branch.

    Args:
      key: A `jax.random` PRNG key.
      cfg: Branch architecture.

    Returns:
      The parameters, structured as `cfg.kind` requires.
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
        conv: list[Layer] = []
        c = cfg.n_channels
        for k, c_out in zip(keys, cfg.cnn_channels):
            lim = jnp.sqrt(6.0 / (c * cfg.kernel_size + c_out * cfg.kernel_size))
            conv.append(
                {
                    "W": jax.random.uniform(k, (cfg.kernel_size, c, c_out), dt, -lim, lim),
                    "b": jnp.zeros((c_out,), dt),
                }
            )
            c = c_out
        return {"conv": conv, "head": [_dense(keys[-1], c * cfg.n_delays, cfg.n_latent, dt)]}

    # A GRU has one block of weights and biases per gate (z, r, n), plus a
    # dense readout.
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


# ── Forward pass ──────────────────────────────────────────────────────────────


def _mlp(head: list[Layer], x: jax.Array, act: Activation, last_linear: bool = True) -> jax.Array:
    """Applies a stack of dense layers.

    Args:
      head: The dense layers, in order.
      x: Input, shape `(B, n_in)`.
      act: Activation applied after each hidden layer.
      last_linear: Leave the final layer without an activation.

    Returns:
      The output of the last layer.
    """
    for i, layer in enumerate(head):
        x = x @ layer["W"] + layer["b"]
        if not (last_linear and i == len(head) - 1):
            x = act(x)
    return x


def _gru_scan(p: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Runs a GRU over a batch of windows.

    Args:
      p: Gate weights and biases from `init_branch`.
      x: Input windows, shape `(B, L, C)`.

    Returns:
      The hidden state after the last step, shape `(B, H)`.
    """

    # Matches torch's `nn.GRU`, which keeps two biases per gate and applies the
    # reset gate to the recurrent bias `b_hn` inside the candidate:
    #     n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    def step(h: jax.Array, x_t: jax.Array) -> tuple[jax.Array, None]:
        """Advances the GRU by one time step.

        Args:
          h: Hidden state, shape `(B, H)`.
          x_t: Input at this step, shape `(B, C)`.

        Returns:
          The next hidden state, and `None` for the scan output.
        """
        z = jax.nn.sigmoid(x_t @ p["Wz"] + p["bz"] + h @ p["Uz"] + p["bhz"])
        r = jax.nn.sigmoid(x_t @ p["Wr"] + p["br"] + h @ p["Ur"] + p["bhr"])
        n = jnp.tanh(x_t @ p["Wn"] + p["bn"] + r * (h @ p["Un"] + p["bhn"]))
        return (1.0 - z) * n + z * h, None

    # `lax.scan` requires the carry to keep one dtype across steps, and the
    # parameters set the dtype of each step's output.
    h0 = jnp.zeros((x.shape[0], p["Wz"].shape[1]), p["Wz"].dtype)
    h, _ = jax.lax.scan(step, h0, jnp.swapaxes(x, 0, 1))
    return h


def _param_dtype(params: BranchParams) -> DTypeLike:
    """Returns the scalar type of a branch's parameters.

    Args:
      params: Branch parameters.

    Returns:
      The dtype of the first parameter array.

    Raises:
      ValueError: If `params` holds no arrays.
    """
    leaves: list[jax.Array] = jax.tree_util.tree_leaves(params)
    if not leaves:
        raise ValueError("branch parameters hold no arrays")
    return leaves[0].dtype


def branch_forward(params: BranchParams, x: jax.Array, cfg: SensorBranchConfig) -> jax.Array:
    """Maps a batch of sensor windows to latent codes.

    Casts the input to the parameter dtype first. The output layer has no
    activation, so the latent codes are unbounded.

    Args:
      params: Branch parameters from `init_branch`.
      x: Sensor windows, shape `(B, n_delays, n_channels)`.
      cfg: Branch architecture.

    Returns:
      Latent codes, shape `(B, n_latent)`.

    Raises:
      ValueError: If `params` lacks the layers `cfg.kind` requires.
    """
    # `lax.conv_general_dilated` and `lax.scan` reject mixed precision
    x = x.astype(_param_dtype(params))
    act = _ACT[cfg.activation]

    if cfg.kind in ("linear", "mlp"):
        return _mlp(params["head"], x.reshape(x.shape[0], -1), act)

    if cfg.kind == "cnn":
        conv = params.get("conv")
        if conv is None:
            raise ValueError("a cnn branch needs 'conv' parameters")
        h = x
        for layer in conv:
            h = (
                jax.lax.conv_general_dilated(h, layer["W"], (1,), "SAME", dimension_numbers=("NWC", "WIO", "NWC"))
                + layer["b"]
            )
            h = act(h)
        return _mlp(params["head"], h.reshape(h.shape[0], -1), act)

    gru = params.get("gru")
    if gru is None:
        raise ValueError("a gru branch needs 'gru' parameters")
    return _mlp(params["head"], _gru_scan(gru, x), act)


# ── Latent space ──────────────────────────────────────────────────────────────


class LinearLatentJax:
    """Wraps a POD basis as an encoder and decoder pair.

    The encoder is `Psi.T` and the decoder is `Psi`.

    Args:
      Psi: Spatial modes with orthonormal columns, shape `(N_x, r)`.
      q_mean: Temporal mean, shape `(N_x, 1)` or `(N_x,)`.
      dtype: Precision to hold the basis in.

    Raises:
      ValueError: If `Psi` is not two-dimensional, or if `q_mean` has a
        different length from the columns of `Psi`.
    """

    def __init__(self, Psi: FloatArray, q_mean: FloatArray, dtype: Dtype = "float32"):
        dt = _scalar_type(dtype)
        self.Psi: jax.Array = jnp.asarray(Psi, dt)
        if self.Psi.ndim != 2:
            raise ValueError(f"Psi must be (N_x, r), got shape {self.Psi.shape}")

        self.q_mean: jax.Array = jnp.asarray(q_mean, dt).reshape(-1, 1)
        if self.q_mean.shape[0] != self.Psi.shape[0]:
            raise ValueError(f"q_mean has {self.q_mean.shape[0]} entries but Psi has {self.Psi.shape[0]} rows")

        self.n_latent = int(self.Psi.shape[1])

    def encode(self, Q: FloatArray) -> FloatArray:
        """Projects physical fields onto the basis.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          POD coefficients, shape `(r, N_t)`.
        """
        return np.asarray(self.Psi.T @ (jnp.asarray(Q, self.Psi.dtype) - self.q_mean))

    def decode(self, Z: FloatArray) -> FloatArray:
        """Reconstructs physical fields from POD coefficients.

        Args:
          Z: POD coefficients, shape `(r, N_t)`.

        Returns:
          Physical fields, shape `(N_x, N_t)`.
        """
        return np.asarray(self.Psi @ jnp.asarray(Z, self.Psi.dtype) + self.q_mean)

    def decode_jax(self, Zt: jax.Array) -> jax.Array:
        """Reconstructs physical fields from a batch of POD coefficients.

        Args:
          Zt: POD coefficients, shape `(B, r)`.

        Returns:
          Physical fields, shape `(B, N_x)`.
        """
        return Zt @ self.Psi.T + self.q_mean.T


class _JaxAutoencoder(Protocol):
    """The members of a fitted `AEJax` or `CAEJax` that `AutoencoderLatentJax` reads."""

    @property
    def fitted(self) -> bool:
        """Whether `fit` has run."""
        ...

    @property
    def N_latent(self) -> int:
        """Dimension of the latent space."""
        ...

    @property
    def scale(self) -> FloatArray:
        """The per-field scale the projector divides inputs by, shape `(N_x, 1)`."""
        ...

    @property
    def Q_mean(self) -> FloatArray:
        """The temporal mean the projector subtracts, shape `(N_x, 1)`."""
        ...

    def encode(self, X: FloatArray) -> FloatArray:
        """Encodes physical fields into latent codes.

        Args:
          X: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Latent codes, shape `(N_latent, N_t)`.
        """
        ...

    def decode(self, Z: FloatArray) -> FloatArray:
        """Decodes latent codes into physical fields.

        Args:
          Z: Latent codes, shape `(N_latent, N_t)`.

        Returns:
          Physical fields, shape `(N_x, N_t)`.
        """
        ...

    def decode_scaled(self, Z: jax.Array) -> jax.Array:
        """Decodes a batch of latent codes differentiably.

        Args:
          Z: Latent codes, shape `(B, N_latent)`.

        Returns:
          Fields in the projector's scaled units, shape `(B, N_x)`.
        """
        ...


class AutoencoderLatentJax:
    """Wraps a fitted `AEJax` or `CAEJax` as an encoder and decoder pair.

    The projector's weights stay fixed.

    Args:
      projector: A fitted `AEJax` or `CAEJax`.
      dtype: Precision `decode_jax` returns. Match `BranchedAEJax.dtype`.

    Raises:
      ValueError: If the projector is not fitted.
    """

    def __init__(self, projector: _JaxAutoencoder, dtype: Dtype = "float32"):
        if not projector.fitted:
            raise ValueError("wrap a fitted AEJax or CAEJax; call projector.fit(X) first")

        dt = _scalar_type(dtype)
        self.projector = projector
        self.n_latent = int(projector.N_latent)
        self.scale_row: jax.Array = jnp.asarray(projector.scale, dt).reshape(1, -1)
        self.mean_row: jax.Array = jnp.asarray(projector.Q_mean, dt).reshape(1, -1)
        if self.scale_row.shape != self.mean_row.shape:
            raise ValueError(
                f"projector scale has {self.scale_row.shape[1]} entries but Q_mean has {self.mean_row.shape[1]}"
            )

    def encode(self, Q: FloatArray) -> FloatArray:
        """Encodes physical fields with the wrapped projector.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Latent codes, shape `(n_latent, N_t)`.
        """
        return np.asarray(self.projector.encode(Q), np.float64)

    def decode(self, Z: FloatArray) -> FloatArray:
        """Decodes latent codes with the wrapped projector.

        Args:
          Z: Latent codes, shape `(n_latent, N_t)`.

        Returns:
          Physical fields, shape `(N_x, N_t)`.
        """
        return np.asarray(self.projector.decode(Z), np.float64)

    def decode_jax(self, Zt: jax.Array) -> jax.Array:
        """Reconstructs physical fields from a batch of latent codes.

        Args:
          Zt: Latent codes, shape `(B, n_latent)`.

        Returns:
          Physical fields, shape `(B, N_x)`.
        """
        fields = self.projector.decode_scaled(Zt).astype(self.scale_row.dtype)
        return fields * self.scale_row + self.mean_row


# ── Optimizer ─────────────────────────────────────────────────────────────────


class AdamState(NamedTuple):
    """Moment estimates and step count for Adam.

    Attributes:
      m: First-moment estimates, one per parameter.
      v: Second-moment estimates, one per parameter.
      t: Number of steps taken.
    """

    m: BranchParams
    v: BranchParams
    t: jax.Array


def _zeros_like(x: jax.Array) -> jax.Array:
    """Returns zeros with the shape and dtype of an array.

    Args:
      x: Any array.

    Returns:
      An array of zeros shaped like `x`.
    """
    return jnp.zeros_like(x)


def adam_init(params: BranchParams) -> AdamState:
    """Creates a zeroed Adam state for a set of parameters.

    Args:
      params: Branch parameters.

    Returns:
      Zero moment estimates shaped like `params`, and a step count of 0.
    """
    z = jax.tree.map(_zeros_like, params)
    return AdamState(m=z, v=jax.tree.map(_zeros_like, params), t=jnp.zeros((), jnp.int32))


def adam_update(
    params: BranchParams,
    grads: BranchParams,
    st: AdamState,
    lr: float,
    wd: float = 0.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[BranchParams, AdamState]:
    """Applies one bias-corrected Adam step with decoupled weight decay.

    Args:
      params: Branch parameters.
      grads: Gradients, shaped like `params`.
      st: Adam state from the previous step.
      lr: Learning rate.
      wd: Weight decay coefficient.
      b1: Decay rate of the first-moment estimate.
      b2: Decay rate of the second-moment estimate.
      eps: Constant added to the denominator.

    Returns:
      The updated parameters and Adam state.
    """
    t = st.t + 1

    def first_moment(m_: jax.Array, g: jax.Array) -> jax.Array:
        """Updates one first-moment estimate.

        Args:
          m_: Previous first-moment estimate.
          g: Gradient for the same parameter.

        Returns:
          The updated first-moment estimate.
        """
        return b1 * m_ + (1 - b1) * g

    def second_moment(v_: jax.Array, g: jax.Array) -> jax.Array:
        """Updates one second-moment estimate.

        Args:
          v_: Previous second-moment estimate.
          g: Gradient for the same parameter.

        Returns:
          The updated second-moment estimate.
        """
        return b2 * v_ + (1 - b2) * g * g

    def correct_first(x: jax.Array) -> jax.Array:
        """Removes the startup bias from a first-moment estimate.

        Args:
          x: First-moment estimate.

        Returns:
          The bias-corrected estimate.
        """
        return x / (1 - b1**t).astype(x.dtype)

    def correct_second(x: jax.Array) -> jax.Array:
        """Removes the startup bias from a second-moment estimate.

        Args:
          x: Second-moment estimate.

        Returns:
          The bias-corrected estimate.
        """
        return x / (1 - b2**t).astype(x.dtype)

    def step(p: jax.Array, m_: jax.Array, v_: jax.Array) -> jax.Array:
        """Applies the Adam step and weight decay to one parameter.

        Args:
          p: Current parameter value.
          m_: Bias-corrected first-moment estimate.
          v_: Bias-corrected second-moment estimate.

        Returns:
          The updated parameter value.
        """
        return p - jnp.asarray(lr, p.dtype) * (m_ / (jnp.sqrt(v_) + eps) + wd * p)

    m = jax.tree.map(first_moment, st.m, grads)
    v = jax.tree.map(second_moment, st.v, grads)
    mc = jax.tree.map(correct_first, m)
    vc = jax.tree.map(correct_second, v)
    new = jax.tree.map(step, params, mc, vc)
    return new, AdamState(m=m, v=v, t=t)


def _global_norm(tree: BranchParams) -> jax.Array:
    """Computes the Euclidean norm over every array in a parameter tree.

    Args:
      tree: Arrays shaped like the branch parameters.

    Returns:
      The global norm, as a scalar.
    """
    leaves: list[jax.Array] = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))


def _clip(grads: BranchParams, max_norm: float) -> BranchParams:
    """Rescales gradients so their global norm is at most `max_norm`.

    Uses no Python branch, so `jax.jit` can trace it.

    Args:
      grads: Gradients shaped like the branch parameters.
      max_norm: Maximum global norm.

    Returns:
      The rescaled gradients.
    """
    n = _global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (n + 1e-12))

    def rescale(g: jax.Array) -> jax.Array:
        """Scales one gradient array.

        Args:
          g: Gradient array.

        Returns:
          The gradient multiplied by the clipping factor.
        """
        return g * scale

    return jax.tree.map(rescale, grads)


# ── Loss and training step ────────────────────────────────────────────────────


def _loss_fn(
    params: BranchParams,
    xb: jax.Array,
    zb: jax.Array,
    cfg: SensorBranchConfig,
    lam: float,
    decode_fn: DecodeFn | None,
    fb: jax.Array | None,
    zs: jax.Array,
) -> jax.Array:
    """Computes the training loss for one batch.

    Args:
      params: Branch parameters.
      xb: Sensor windows, shape `(B, n_delays, n_channels)`.
      zb: Latent targets, shape `(B, n_latent)`.
      cfg: Branch architecture.
      lam: Weight on the field term. 0 omits it.
      decode_fn: Differentiable decoder for the field term, or `None`.
      fb: Field targets, shape `(B, N_x)`, or `None`.
      zs: Per-mode latent scale the targets `zb` are divided by, shape
        `(1, n_latent)`.

    Returns:
      The scalar loss.

    Raises:
      ValueError: If the field term is on but `fb` is `None`.
    """
    z_hat = branch_forward(params, xb, cfg)
    loss = jnp.mean((z_hat - zb) ** 2)

    if lam > 0.0 and decode_fn is not None:
        if fb is None:
            raise ValueError("the field loss term needs field targets")
        loss = loss + lam * jnp.mean((decode_fn(z_hat * zs) - fb) ** 2)

    return loss


# `lam` is static because it decides whether the decoder pass is traced.
@partial(jax.jit, static_argnames=("cfg", "decode_fn", "lam", "wd", "clip"))
def train_step(
    params: BranchParams,
    opt_state: AdamState,
    X: jax.Array,
    Z: jax.Array,
    F: jax.Array | None,
    zs: jax.Array,
    batch: jax.Array,
    lr: float,
    cfg: SensorBranchConfig,
    lam: float,
    decode_fn: DecodeFn | None,
    wd: float = 0.0,
    clip: float = 5.0,
) -> tuple[BranchParams, AdamState, jax.Array]:
    """Runs one clipped Adam step on one batch.

    Gathers the batch inside the compiled step, so the full arrays stay on
    device and only the index array changes between steps.

    Args:
      params: Branch parameters.
      opt_state: Adam state.
      X: Every sensor window, shape `(N_t, n_delays, n_channels)`.
      Z: Every latent target, shape `(N_t, n_latent)`.
      F: Every field target, shape `(N_t, N_x)`, or `None` when `lam` is 0.
      zs: Per-mode latent scale the targets `Z` are divided by, shape
        `(1, n_latent)`.
      batch: Indices selecting this batch.
      lr: Learning rate.
      cfg: Branch architecture.
      lam: Weight on the field term.
      decode_fn: Differentiable decoder for the field term, or `None`.
      wd: Weight decay coefficient.
      clip: Maximum global gradient norm.

    Returns:
      The updated parameters, the updated Adam state, and the batch loss.

    Raises:
      ValueError: If `lam` is above 0 with a decoder but `F` is `None`.
    """
    xb, zb = X[batch], Z[batch]
    fb = F[batch] if F is not None else None
    loss, grads = jax.value_and_grad(_loss_fn)(params, xb, zb, cfg, lam, decode_fn, fb, zs)
    grads = _clip(grads, clip)
    params, opt_state = adam_update(params, grads, opt_state, lr, wd)
    return params, opt_state, loss


@partial(jax.jit, static_argnames=("cfg", "decode_fn", "lam"))
def eval_loss(
    params: BranchParams,
    X: jax.Array,
    Z: jax.Array,
    F: jax.Array | None,
    zs: jax.Array,
    cfg: SensorBranchConfig,
    lam: float,
    decode_fn: DecodeFn | None,
) -> jax.Array:
    """Computes the loss over a whole evaluation block.

    Args:
      params: Branch parameters.
      X: Sensor windows, shape `(N, n_delays, n_channels)`.
      Z: Latent targets, shape `(N, n_latent)`.
      F: Field targets, shape `(N, N_x)`, or `None`.
      zs: Per-mode latent scale the targets `Z` are divided by, shape
        `(1, n_latent)`.
      cfg: Branch architecture.
      lam: Weight on the field term.
      decode_fn: Differentiable decoder for the field term, or `None`.

    Returns:
      The scalar loss.

    Raises:
      ValueError: If `lam` is above 0 with a decoder but `F` is `None`.
    """
    return _loss_fn(params, X, Z, cfg, lam, decode_fn, F, zs)


def _epoch_batches(key: jax.Array, n: int, batch_size: int) -> jax.Array:
    """Shuffles `n` indices into equal batches.

    Drops the final partial batch, so every step has the same shape and
    `jax.jit` compiles the step once.

    Args:
      key: A `jax.random` PRNG key.
      n: Number of indices to shuffle.
      batch_size: Indices per batch.

    Returns:
      Batch indices, shape `(n // batch_size, batch_size)`.
    """
    perm = jax.random.permutation(key, n)
    nb = n // batch_size
    return perm[: nb * batch_size].reshape(nb, batch_size)


# ── Estimator ─────────────────────────────────────────────────────────────────


class _JaxLatent(Protocol):
    """An encoder and decoder pair that `BranchedAEJax` trains against.

    Attributes:
      n_latent: Dimension of the latent space.
    """

    n_latent: int

    def encode(self, Q: FloatArray) -> FloatArray:
        """Encodes physical fields into latent codes.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Latent codes, shape `(n_latent, N_t)`.
        """
        ...

    def decode(self, Z: FloatArray) -> FloatArray:
        """Decodes latent codes into physical fields.

        Args:
          Z: Latent codes, shape `(n_latent, N_t)`.

        Returns:
          Physical fields, shape `(N_x, N_t)`.
        """
        ...


@dataclass
class BranchedAEJax:
    """Trains a sensor branch against a frozen encoder and decoder pair, in JAX.

    Mirrors `branched_ae.BranchedAE`. Pass the whole record to each method and
    select snapshots with an index array, because the sensor windows are causal.

    Attributes:
      latent: A `LinearLatentJax` or `AutoencoderLatentJax`, or any latent space
        with `encode`, `decode`, and `n_latent`.
      branch: Sensor branch architecture. One of `"linear"`, `"mlp"`, `"cnn"`,
        or `"gru"`. See `SensorBranchConfig`.
      n_delays: Causal window length in samples.
      delay_stride: Sample spacing between consecutive lags.
      hidden: Hidden widths for an MLP branch.
      activation: Activation name for the branch.
      cnn_channels: Channel widths for a CNN branch.
      kernel_size: Convolution kernel length for a CNN branch.
      gru_hidden: Hidden state width for a GRU branch.
      lambda_field: Weight on the field-space loss term. 0 trains on the latent
        term alone.
      latent_weight: Scaling of the latent targets. `"energy"` leaves them
        unchanged; `"unit"` divides each mode by its standard deviation.
      learning_rate: Initial Adam step size.
      weight_decay: Decoupled weight decay coefficient.
      n_epochs: Maximum number of training epochs.
      batch_size: Minibatch size, capped at the number of training snapshots.
      val_fraction: Fraction of the training block held out for early stopping,
        taken contiguously from the end of the block.
      patience: Epochs without improvement before training stops.
      threshold: Relative improvement that resets `patience`.
      lr_factor: Factor the learning rate decays by on a plateau.
      lr_patience: Plateau epochs before the learning rate decays.
      min_lr: Lower bound on the learning rate.
      grad_clip: Maximum global gradient norm.
      seed: PRNG seed.
      dtype: Precision to train in.
      verbose: Print each learning-rate decay.
      params: The trained parameters, or `None` before `fit`.
      cfg: The branch architecture `fit` built, or `None` before `fit`.
      s_mean: Per-channel sensor mean, learned by `fit`.
      s_scale: Per-channel sensor scale, learned by `fit`.
      z_scale: Per-mode latent scale, learned by `fit`.
      loss_history: Mean training loss per epoch.
      val_loss_history: Validation loss per epoch.
      fitted: Whether `fit` has completed.
    """

    latent: _JaxLatent
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
    dtype: Dtype = "float32"
    verbose: bool = False

    params: BranchParams | None = field(default=None, repr=False)
    cfg: SensorBranchConfig | None = field(default=None, repr=False)
    s_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    z_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    loss_history: list[float] = field(default_factory=list[float], repr=False)
    val_loss_history: list[float] = field(default_factory=list[float], repr=False)
    fitted: bool = False

    @property
    def warmup(self) -> int:
        """Number of leading snapshots whose window is partly zero padding.

        Exclude these indices from every index array passed to this class.
        """
        return (self.n_delays - 1) * self.delay_stride

    # ── API ──

    def fit(self, Q: FloatArray, S: FloatArray, train_idx: IndexArray) -> BranchedAEJax:
        """Trains the sensor branch on the given snapshots.

        Args:
          Q: Full field record, shape `(N_x, N_t)`.
          S: Full synchronized sensor record, shape `(N_s, N_t)`.
          train_idx: Snapshot indices to train on. Every index must be at or
            after `warmup`.

        Returns:
          This estimator, fitted.

        Raises:
          ValueError: If an option is out of range, if `Q` and `S` hold
            different snapshot counts, if `train_idx` is empty or reaches into
            the warm-up region, if the sensor record holds non-finite values, if
            `val_fraction` leaves no training snapshots, or if `lambda_field` is
            above 0 for a latent space with no `decode_jax`.
        """
        if self.latent_weight not in ("energy", "unit"):
            raise ValueError(f"latent_weight must be 'energy' or 'unit', got {self.latent_weight!r}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        Q, S = np.asarray(Q, np.float64), np.asarray(S, np.float64)
        train_idx = np.asarray(train_idx)
        if Q.shape[1] != S.shape[1]:
            raise ValueError(
                f"field has {Q.shape[1]} snapshots, sensors {S.shape[1]}; these must be synchronised sample-for-sample"
            )
        if train_idx.size == 0:
            raise ValueError("train_idx is empty")
        if train_idx.min() < self.warmup:
            raise ValueError(
                f"train_idx starts at {train_idx.min()} but the first "
                f"{self.warmup} columns have partly-padded windows. Drop them."
            )
        if not np.isfinite(S[:, train_idx]).all():
            raise ValueError("non-finite sensor values; interpolate dropouts first")

        decode_fn: DecodeFn | None = getattr(self.latent, "decode_jax", None) if self.lambda_field > 0 else None
        if self.lambda_field > 0 and decode_fn is None:
            raise ValueError("lambda_field > 0 needs a latent space with a decode_jax method")

        dt = _scalar_type(self.dtype)
        # The sensor statistics come from the training columns only.
        self.s_mean, self.s_scale = sensor_stats(S[:, train_idx])

        W = sensor_windows(apply_sensor_stats(S, self.s_mean, self.s_scale), self.n_delays, self.delay_stride)
        Z = self.latent.encode(Q).T
        self.z_scale = (
            np.std(Z[train_idx], axis=0, keepdims=True) if self.latent_weight == "unit" else np.ones((1, Z.shape[1]))
        )
        self.z_scale = np.where(self.z_scale > 0, self.z_scale, 1.0)

        Xj = jnp.asarray(W, dt)
        Zj = jnp.asarray(Z / self.z_scale, dt)
        Fj = jnp.asarray(Q.T, dt) if self.lambda_field > 0 else None
        zsj = jnp.asarray(self.z_scale, dt)

        n_val = int(round(self.val_fraction * len(train_idx)))
        if len(train_idx) - n_val < 1:
            raise ValueError(f"val_fraction={self.val_fraction} leaves no training snapshots out of {len(train_idx)}")
        tr_i = jnp.asarray(train_idx[: len(train_idx) - n_val])
        va_i = jnp.asarray(train_idx[len(train_idx) - n_val :]) if n_val else None

        cfg = SensorBranchConfig(
            n_channels=S.shape[0],
            n_delays=self.n_delays,
            n_latent=int(Z.shape[1]),
            kind=self.branch,
            hidden=tuple(self.hidden),
            activation=self.activation,
            cnn_channels=tuple(self.cnn_channels),
            kernel_size=self.kernel_size,
            gru_hidden=self.gru_hidden,
            dtype=self.dtype,
        )
        self.cfg = cfg

        key = jax.random.PRNGKey(self.seed)
        key, k0 = jax.random.split(key)
        params = init_branch(k0, cfg)
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
                params, opt, loss = cast(
                    "tuple[BranchParams, AdamState, jax.Array]",
                    train_step(
                        params,
                        opt,
                        Xj,
                        Zj,
                        Fj,
                        zsj,
                        batches[i],
                        lr,
                        cfg,
                        self.lambda_field,
                        decode_fn,
                        self.weight_decay,
                        self.grad_clip,
                    ),
                )
                run += float(loss)
            self.loss_history.append(run / max(batches.shape[0], 1))

            if va_i is not None:
                v = float(
                    cast(
                        "jax.Array",
                        eval_loss(
                            params,
                            Xj[va_i],
                            Zj[va_i],
                            Fj[va_i] if Fj is not None else None,
                            zsj,
                            cfg,
                            self.lambda_field,
                            decode_fn,
                        ),
                    )
                )
                self.val_loss_history.append(v)
                if v < best * (1.0 - self.threshold):
                    best, best_params, wait, lr_wait = v, params, 0, 0
                else:
                    wait += 1
                    lr_wait += 1
                    # The learning-rate decay and early stopping keep separate
                    # counters, as `BranchedAE` does.
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

    def encode(self, S: FloatArray, idx: IndexArray | None = None) -> FloatArray:
        """Predicts latent codes from sensor data.

        Args:
          S: Full sensor record, shape `(N_s, N_t)`. Pass the whole record,
            because the window at each index reaches back into earlier
            snapshots.
          idx: Snapshots to evaluate. `None` evaluates every snapshot.

        Returns:
          Predicted latent codes, shape `(n_latent, len(idx))`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        params, cfg = self._trained()
        S = np.asarray(S, np.float64)
        W = sensor_windows(apply_sensor_stats(S, self.s_mean, self.s_scale), self.n_delays, self.delay_stride)
        if idx is not None:
            W = W[np.asarray(idx)]
        dt = _scalar_type(self.dtype)
        Z = np.asarray(branch_forward(params, jnp.asarray(W, dt), cfg))
        return (Z * self.z_scale).T

    def predict(self, S: FloatArray, idx: IndexArray | None = None) -> FloatArray:
        """Reconstructs fields from sensor data.

        Args:
          S: Full sensor record, shape `(N_s, N_t)`.
          idx: Snapshots to evaluate. `None` evaluates every snapshot.

        Returns:
          The reconstructed field, shape `(N_x, len(idx))`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        return self.latent.decode(self.encode(S, idx))

    def score(self, Q: FloatArray, S: FloatArray, idx: IndexArray | None = None) -> float:
        """Returns the normalized MSE of the reconstruction.

        Args:
          Q: Full field record, shape `(N_x, N_t)`.
          S: Full sensor record, shape `(N_s, N_t)`.
          idx: Snapshots to score. `None` scores every snapshot.

        Returns:
          Normalized MSE
        Raises:
          RuntimeError: If `fit` has not been called.
        """
        Q = np.asarray(Q, np.float64)
        Qt = Q[:, idx] if idx is not None else Q
        return nmse(Qt, self.predict(S, idx))

    def latent_score(self, Q: FloatArray, S: FloatArray, idx: IndexArray) -> float:
        """Returns the normalized MSE in latent space.

        Scores the sensor branch against the encoder's codes, independent of the
        decoder.

        Args:
          Q: Full field record, shape `(N_x, N_t)`.
          S: Full sensor record, shape `(N_s, N_t)`.
          idx: Snapshots to score.

        Returns:
          Normalized MSE between the true and predicted latent codes.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        Z = self.latent.encode(np.asarray(Q, np.float64))[:, idx]
        return nmse(Z, self.encode(S, idx))

    @property
    def n_params(self) -> int:
        """Number of trainable parameters in the sensor branch, or 0 before `fit`."""
        if not self.params:
            return 0
        leaves: list[jax.Array] = jax.tree.leaves(self.params)
        return int(sum(x.size for x in leaves))

    def _check(self) -> None:
        """Raises if the estimator has no trained parameters.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._trained()

    def _trained(self) -> tuple[BranchParams, SensorBranchConfig]:
        """Returns the trained parameters and the architecture they belong to.

        Checks the parameters themselves rather than only the `fitted` flag,
        which is a constructor argument and can be set without training.

        Returns:
          The parameters and the branch architecture.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if not self.fitted or self.params is None or self.cfg is None:
            raise RuntimeError("call fit() before encode()/predict()/score()")
        return self.params, self.cfg
