"""Convolutional autoencoder in JAX.

`CAEJax` trains the same network as the PyTorch `CAE`, with parameters in nested
dictionaries and a compiled training step.

Typical usage example:

  cae = CAEJax(n_latent=8, channels=(16, 32, 64)).fit(X)
  Z = cae.encode(X)
  Q_hat = cae.decode(Z)
  mse = cae.score(X)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.typing import DTypeLike

from .base import DecoderStage, EncoderStage, FloatArray, conv_stages
from .jax_utils import (
    Activation,
    ActivationSpec,
    AdamState,
    JaxAutoencoder,
    Layer,
    activation_fn,
    adam_update,
    resolve_activations,
    uniform,
)

__all__ = [
    "CAEJax",
    "CAEJaxConfig",
    "CAEParams",
    "conv2d",
    "conv_transpose2d",
    "decode_grid",
    "encode_grid",
    "flat_to_grid",
    "get_cae_params",
    "grid_to_flat",
    "masked_mse_loss",
]

#: Integer indices of the fluid points in the flattened `Nx * Ny` grid.
IndexArray = npt.NDArray[np.intp]

#: Axis order of inputs, kernels, and outputs, matching `torch.nn.Conv2d`.
_LAYOUT = ("NCHW", "OIHW", "NCHW")


class CAEParams(TypedDict):
    """Parameters of the convolutional autoencoder.

    Attributes:
      enc: Encoder convolutions, input first. Weights have shape
        `(c_out, c_in, k, k)`.
      dec: Decoder transposed convolutions, bottleneck first. Weights have shape
        `(c_in, c_out, k, k)`.
      enc_fc: Linear map from the flattened bottleneck grid to the latent space.
      dec_fc: Linear map from the latent space to the flattened bottleneck grid.
    """

    enc: list[Layer]
    dec: list[Layer]
    enc_fc: Layer
    dec_fc: Layer


@dataclass(frozen=True)
class CAEJaxConfig:
    """Static architecture of the convolutional autoencoder.

    Instances are hashable, so they can serve as a static argument to `jax.jit`.

    Attributes:
      grid_shape: Input grid shape `(Nu, Nx, Ny)`.
      channels: Output channels of each encoder convolution.
      kernel_size: Convolution kernel size.
      stride: Convolution stride.
      pad: Convolution padding.
      n_latent: Dimension of the latent space.
      activation: Activation after each encoder convolution.
      layer_activations: Activation after each decoder transposed convolution
        but the last. `None` uses `activation`.
      dtype: Precision of the parameters.
      encoder: Layer sizes of the encoder, computed from the other fields.
      decoder: Layer sizes of the decoder, computed from the other fields.

    Raises:
      ValueError: If the grid is too small for the convolution stages.
    """

    grid_shape: tuple[int, int, int]
    channels: tuple[int, ...]
    kernel_size: int
    stride: int
    pad: int
    n_latent: int
    activation: ActivationSpec = "tanh"
    layer_activations: ActivationSpec | None = None
    dtype: DTypeLike = np.float32
    encoder: tuple[EncoderStage, ...] = field(init=False, compare=False, repr=False)
    decoder: tuple[DecoderStage, ...] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Computes the layer sizes.

        Raises:
          ValueError: If the grid is too small for the convolution stages.
        """
        encoder, decoder = conv_stages(self.grid_shape, self.channels, self.kernel_size, self.stride, self.pad)
        object.__setattr__(self, "encoder", encoder)
        object.__setattr__(self, "decoder", decoder)

    @property
    def enc_acts(self) -> tuple[str | Activation, ...]:
        """One activation per encoder convolution."""
        return resolve_activations(self.activation, len(self.channels), "activation")

    @property
    def dec_acts(self) -> tuple[str | Activation, ...]:
        """One activation per decoder transposed convolution but the last."""
        spec = self.activation if self.layer_activations is None else self.layer_activations
        return resolve_activations(spec, len(self.channels) - 1, "layer_activations")

    @property
    def red_shape(self) -> tuple[int, int, int]:
        """Channels and grid size at the bottleneck."""
        last = self.encoder[-1]
        return (last.c_out, *last.size_out)

    @property
    def flat_dim(self) -> int:
        """Number of entries in the flattened bottleneck grid."""
        c, h, w = self.red_shape
        return c * h * w


# ── Network ───────────────────────────────────────────────────────────────────


def conv2d(x: jax.Array, W: jax.Array, b: jax.Array, s: int, p: int) -> jax.Array:
    """Applies a strided, zero-padded 2-D convolution.

    Args:
      x: Inputs, shape `(B, c_in, H, W)`.
      W: Kernel, shape `(c_out, c_in, k, k)`.
      b: Bias, shape `(c_out,)`.
      s: Stride.
      p: Padding on each side.

    Returns:
      Outputs, shape `(B, c_out, H_out, W_out)`.
    """
    y = jax.lax.conv_general_dilated(x, W, window_strides=(s, s), padding=((p, p), (p, p)), dimension_numbers=_LAYOUT)
    return y + b[None, :, None, None]


def conv_transpose2d(x: jax.Array, W: jax.Array, b: jax.Array, s: int, p: int, op: tuple[int, int]) -> jax.Array:
    """Applies a 2-D transposed convolution, as `torch.nn.ConvTranspose2d` does.

    Dilates the input by the stride and convolves with the flipped kernel.

    Args:
      x: Inputs, shape `(B, c_in, H, W)`.
      W: Kernel, shape `(c_in, c_out, k, k)`.
      b: Bias, shape `(c_out,)`.
      s: Stride.
      p: Padding the matching convolution applied.
      op: Extra rows and columns added to the bottom and right of the output.

    Returns:
      Outputs, shape `(B, c_out, H_out, W_out)`.
    """
    k = W.shape[-1]
    V = jnp.flip(jnp.transpose(W, (1, 0, 2, 3)), axis=(2, 3))
    lo = k - 1 - p
    y = jax.lax.conv_general_dilated(
        x,
        V,
        window_strides=(1, 1),
        padding=((lo, lo + op[0]), (lo, lo + op[1])),
        lhs_dilation=(s, s),
        dimension_numbers=_LAYOUT,
    )
    return y + b[None, :, None, None]


def get_cae_params(key: jax.Array, cfg: CAEJaxConfig) -> CAEParams:
    """Draws initial parameters for the encoder and decoder.

    Args:
      key: A `jax.random` key.
      cfg: The architecture.

    Returns:
      The parameters.
    """
    k, dt = cfg.kernel_size, cfg.dtype
    enc: list[Layer] = []
    for stage in cfg.encoder:
        key, w_key, b_key = jax.random.split(key, 3)
        fan = stage.c_in * k * k
        enc.append(
            {
                "W": uniform(w_key, (stage.c_out, stage.c_in, k, k), fan, dt),
                "b": uniform(b_key, (stage.c_out,), fan, dt),
            }
        )
    dec: list[Layer] = []
    for stage in cfg.decoder:
        key, w_key, b_key = jax.random.split(key, 3)
        fan = stage.c_out * k * k
        dec.append(
            {
                "W": uniform(w_key, (stage.c_in, stage.c_out, k, k), fan, dt),
                "b": uniform(b_key, (stage.c_out,), fan, dt),
            }
        )

    _, w1, b1, w2, b2 = jax.random.split(key, 5)
    n_flat, n_latent = cfg.flat_dim, cfg.n_latent
    return {
        "enc": enc,
        "dec": dec,
        "enc_fc": {"W": uniform(w1, (n_flat, n_latent), n_flat, dt), "b": uniform(b1, (n_latent,), n_flat, dt)},
        "dec_fc": {"W": uniform(w2, (n_latent, n_flat), n_latent, dt), "b": uniform(b2, (n_flat,), n_latent, dt)},
    }


def encode_grid(params: CAEParams, G: jax.Array, cfg: CAEJaxConfig) -> jax.Array:
    """Encodes a batch of scaled grid snapshots.

    Args:
      params: The parameters.
      G: Scaled snapshots with zeros at solid points, shape `(B, Nu, Nx, Ny)`.
      cfg: The architecture.

    Returns:
      Latent codes, shape `(B, n_latent)`.
    """
    h = G
    for layer, a in zip(params["enc"], cfg.enc_acts, strict=True):
        h = activation_fn(a)(conv2d(h, layer["W"], layer["b"], cfg.stride, cfg.pad))
    h = h.reshape(h.shape[0], -1)
    return h @ params["enc_fc"]["W"] + params["enc_fc"]["b"]


def decode_grid(params: CAEParams, Z: jax.Array, cfg: CAEJaxConfig) -> jax.Array:
    """Decodes a batch of latent codes onto the grid.

    Args:
      params: The parameters.
      Z: Latent codes, shape `(B, n_latent)`.
      cfg: The architecture.

    Returns:
      Scaled fields, shape `(B, Nu, Nx, Ny)`.
    """
    h = (Z @ params["dec_fc"]["W"] + params["dec_fc"]["b"]).reshape(-1, *cfg.red_shape)
    acts = cfg.dec_acts
    for i, (layer, stage) in enumerate(zip(params["dec"], cfg.decoder, strict=True)):
        h = conv_transpose2d(h, layer["W"], layer["b"], cfg.stride, cfg.pad, stage.output_padding)
        if i < len(acts):
            h = activation_fn(acts[i])(h)
    return h


def masked_mse_loss(params: CAEParams, G: jax.Array, cfg: CAEJaxConfig, mask: jax.Array) -> jax.Array:
    """Computes the mean squared reconstruction error over fluid points.

    Args:
      params: The parameters.
      G: Scaled snapshots with zeros at solid points, shape `(B, Nu, Nx, Ny)`.
      cfg: The architecture.
      mask: 1 at fluid points and 0 at solid points, shape `(1, 1, Nx, Ny)`.

    Returns:
      The scalar loss.
    """
    recon = decode_grid(params, encode_grid(params, G, cfg), cfg)
    return ((recon - G) ** 2 * mask).sum() / (mask.sum() * G.shape[0] * G.shape[1])


def flat_to_grid(Q: FloatArray | jax.Array, cfg: CAEJaxConfig, idx: IndexArray) -> jax.Array:
    """Places flat fields on the grid, with zeros at solid points.

    Args:
      Q: Flat fields, shape `(N_x, N_t)`.
      cfg: The architecture.
      idx: Fluid point indices in the flattened `Nx * Ny` grid.

    Returns:
      Fields, shape `(N_t, Nu, Nx, Ny)`.
    """
    Nu, Nx, Ny = cfg.grid_shape
    n_t = Q.shape[1]
    values = jnp.asarray(Q, dtype=cfg.dtype).reshape(Nu, idx.size, n_t).transpose(0, 2, 1)
    grid = jnp.zeros((Nu, n_t, Nx * Ny), dtype=cfg.dtype).at[:, :, idx].set(values)
    return grid.reshape(Nu, n_t, Nx, Ny).transpose(1, 0, 2, 3)


def grid_to_flat(G: jax.Array, cfg: CAEJaxConfig, idx: IndexArray) -> jax.Array:
    """Selects the fluid points of fields on the grid.

    Args:
      G: Fields, shape `(N_t, Nu, Nx, Ny)`.
      cfg: The architecture.
      idx: Fluid point indices in the flattened `Nx * Ny` grid.

    Returns:
      Flat fields, shape `(N_x, N_t)`.
    """
    Nu, Nx, Ny = cfg.grid_shape
    fluid = G.reshape(G.shape[0], Nu, Nx * Ny)[:, :, idx]
    return fluid.transpose(1, 2, 0).reshape(-1, G.shape[0])


# ── Compiled training step ────────────────────────────────────────────────────


@partial(jax.jit, static_argnames=("cfg", "wd"))
def _train_step(
    params: CAEParams,
    opt_state: AdamState[CAEParams],
    G: jax.Array,
    batches: jax.Array,
    i: int,
    lr: float,
    cfg: CAEJaxConfig,
    mask: jax.Array,
    wd: float,
) -> tuple[CAEParams, AdamState[CAEParams], jax.Array]:
    """Runs one Adam step on one batch, gathering the batch inside the compiled step.

    Args:
      params: The parameters.
      opt_state: Adam state.
      G: Every training snapshot, shape `(N_train, Nu, Nx, Ny)`.
      batches: Batch indices, shape `(n_batches, batch_size)`.
      i: Row of `batches` to train on.
      lr: Learning rate.
      cfg: The architecture.
      mask: 1 at fluid points and 0 at solid points, shape `(1, 1, Nx, Ny)`.
      wd: L2 penalty coefficient.

    Returns:
      The updated parameters, the updated Adam state, and the batch loss.
    """
    loss, grads = jax.value_and_grad(masked_mse_loss)(params, G[batches[i]], cfg, mask)
    params, opt_state = adam_update(grads, opt_state, params, lr, wd=wd)
    return params, opt_state, loss


@partial(jax.jit, static_argnames="cfg")
def _eval_loss(params: CAEParams, G: jax.Array, cfg: CAEJaxConfig, mask: jax.Array) -> jax.Array:
    """Computes the reconstruction loss over a block of snapshots.

    Args:
      params: The parameters.
      G: Scaled snapshots, shape `(N, Nu, Nx, Ny)`.
      cfg: The architecture.
      mask: 1 at fluid points and 0 at solid points, shape `(1, 1, Nx, Ny)`.

    Returns:
      The scalar loss.
    """
    return masked_mse_loss(params, G, cfg, mask)


# ── Projector ─────────────────────────────────────────────────────────────────


@dataclass(eq=False, repr=False)
class CAEJax(JaxAutoencoder[CAEParams]):
    """A convolutional autoencoder in JAX.

    The encoder applies one convolution per entry of `channels`, each followed
    by `activation`, then flattens and maps linearly to `n_latent` coefficients.
    The decoder maps linearly back to the bottleneck grid and applies the
    mirrored transposed convolutions, with the last one linear. Solid points
    enter as zeros and are excluded from the loss. `Autoencoder` documents the
    training options.

    Attributes:
      channels: Output channels of each encoder convolution.
      kernel_size: Convolution kernel size.
      stride: Convolution stride.
      pad: Convolution padding.

    Raises:
      ValueError: If an option is out of range, or an activation option has the
        wrong length or an unknown name.
      TypeError: If an activation is neither a name nor a callable.
    """

    channels: Sequence[int] = (16, 32, 64)
    kernel_size: int = 3
    stride: int = 2
    pad: int = 1

    _cfg: CAEJaxConfig | None = field(default=None, init=False)
    _idx: IndexArray | None = field(default=None, init=False)
    _mask: jax.Array | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range, or an activation option has
            the wrong length or an unknown name.
          TypeError: If an activation is neither a name nor a callable.
        """
        super().__post_init__()
        self.channels = tuple(self.channels)
        if not self.channels or any(c < 1 for c in self.channels):
            raise ValueError(f"channels must be non-empty with entries >= 1, got {self.channels}")
        if self.kernel_size < 1 or self.stride < 1 or self.pad < 0:
            raise ValueError(
                f"need kernel_size >= 1, stride >= 1, pad >= 0; got {self.kernel_size}, {self.stride}, {self.pad}"
            )
        resolve_activations(self.activation, len(self.channels), "activation")
        if self.layer_activations is not None:
            resolve_activations(self.layer_activations, len(self.channels) - 1, "layer_activations")

    @property
    def cfg(self) -> CAEJaxConfig:
        """The static architecture.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if self._cfg is None:
            raise RuntimeError("CAEJax is not fitted; call fit() first")
        return self._cfg

    def fit(self, X: npt.NDArray[Any]) -> CAEJax:
        """Trains the encoder and decoder on snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`, with NaN at solid
            points.

        Returns:
          This autoencoder, fitted.

        Raises:
          ValueError: If `X` is not a valid grid record, `val_fraction` leaves no
            training snapshots, the grid is too small for the convolution
            stages, or `dtype` is float64 without `jax_enable_x64`.
        """
        self._check_precision()
        Q, _ = self._prepare_fit(X)
        self._configure()
        cfg, wd, idx, mask = self.cfg, self.weight_decay, self._fluid_idx(), self._fluid_weights()

        def step(
            params: CAEParams, opt_state: AdamState[CAEParams], G: jax.Array, batches: jax.Array, i: int, lr: float
        ) -> tuple[CAEParams, AdamState[CAEParams], jax.Array]:
            """Runs one compiled training step."""
            return cast(
                "tuple[CAEParams, AdamState[CAEParams], jax.Array]",
                _train_step(params, opt_state, G, batches, i, lr, cfg, mask, wd),
            )

        def loss(params: CAEParams, G_val: jax.Array) -> jax.Array:
            """Computes the compiled validation loss."""
            return cast("jax.Array", _eval_loss(params, G_val, cfg, mask))

        data = flat_to_grid(Q / self.scale, cfg, idx)
        self.weights, history = self._train(data, self._init_params, step, loss)
        self._finish_fit(history)
        return self

    def encode(self, X: npt.NDArray[Any]) -> FloatArray:
        """Encodes snapshots into latent coefficients.

        Args:
          X: Snapshots in the grid, snapshot, or flat layout.

        Returns:
          Latent coefficients, shape `(n_latent, N_t)`.

        Raises:
          RuntimeError: If `fit` has not completed.
          ValueError: If `X` does not match the fitted grid.
        """
        params, cfg = self._trained_params(), self.cfg
        G = flat_to_grid(self.preprocess_snapshot(X) / self.scale, cfg, self._fluid_idx())
        return np.asarray(encode_grid(params, G, cfg)).T

    def decode(self, Z: npt.NDArray[Any]) -> FloatArray:
        """Decodes latent coefficients into flat fields.

        Args:
          Z: Latent coefficients, shape `(n_latent, N_t)`.

        Returns:
          Flat fields with the temporal mean restored, shape `(N_x, N_t)`.

        Raises:
          RuntimeError: If `fit` has not completed.
          ValueError: If `Z` is not two-dimensional with `n_latent` rows.
        """
        params, cfg = self._trained_params(), self.cfg
        Z = self._check_latent(Z)
        G = decode_grid(params, jnp.asarray(Z.T, dtype=cfg.dtype), cfg)
        return np.asarray(grid_to_flat(G, cfg, self._fluid_idx())) * self.scale + self.Q_mean

    def decode_scaled(self, Z: jax.Array) -> jax.Array:
        """Decodes a batch of latent codes differentiably onto the fluid points.

        Args:
          Z: Latent codes, shape `(B, n_latent)`.

        Returns:
          Fields in the scaled units the network trains on, in the flat layout
          `decode` returns, shape `(B, N_x)`.

        Raises:
          RuntimeError: If `fit` has not completed.
        """
        params, cfg = self._trained_params(), self.cfg
        G = decode_grid(params, jnp.asarray(Z, dtype=cfg.dtype), cfg)
        return grid_to_flat(G, cfg, self._fluid_idx()).T

    def _configure(self) -> None:
        """Builds the architecture and fluid mask for the recorded grid.

        Raises:
          ValueError: If the grid is too small for the convolution stages.
        """
        Nu, Nx, Ny = self._grid()
        fluid = self._fluid_mask()
        self._cfg = CAEJaxConfig(
            grid_shape=(Nu, Nx, Ny),
            channels=tuple(self.channels),
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad=self.pad,
            n_latent=self.n_latent,
            activation=self.activation,
            layer_activations=self.layer_activations,
            dtype=self.dtype,
        )
        self._idx = np.flatnonzero(fluid)
        self._mask = jnp.asarray(fluid.reshape(Nx, Ny), dtype=self.dtype)[None, None]

    def _init_params(self, key: jax.Array) -> CAEParams:
        """Draws initial parameters.

        Args:
          key: A `jax.random` key.

        Returns:
          The parameters.
        """
        return get_cae_params(key, self.cfg)

    def _fluid_idx(self) -> IndexArray:
        """Returns the fluid point indices.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if self._idx is None:
            raise RuntimeError("CAEJax is not fitted; call fit() first")
        return self._idx

    def _fluid_weights(self) -> jax.Array:
        """Returns the loss mask, 1 at fluid points, shape `(1, 1, Nx, Ny)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if self._mask is None:
            raise RuntimeError("CAEJax is not fitted; call fit() first")
        return self._mask
