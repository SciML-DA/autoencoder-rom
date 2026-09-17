"""Dense autoencoder in JAX.

`AEJax` trains the same network as the PyTorch `AE`, with parameters in nested
dictionaries and a compiled training step.

Typical usage example:

  ae = AEJax(n_latent=10, hidden=(128, 32)).fit(X)
  Z = ae.encode(X)
  Q_hat = ae.decode(Z)
  mse = ae.score(X)
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

from .base import FloatArray
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

__all__ = ["AEJax", "AEJaxConfig", "AEParams", "forward", "get_ae_params", "get_mlp_params", "mlp", "mse_loss"]


class AEParams(TypedDict):
    """Parameters of the dense autoencoder.

    Attributes:
      enc: Encoder layers, input first.
      dec: Decoder layers, latent first.
    """

    enc: list[Layer]
    dec: list[Layer]


@dataclass(frozen=True)
class AEJaxConfig:
    """Static architecture of the dense autoencoder.

    Instances are hashable, so they can serve as a static argument to `jax.jit`.

    Attributes:
      n_x: Number of rows in the flat field, `N_x`.
      hidden: Hidden layer widths of the encoder.
      n_latent: Dimension of the latent space.
      activation: Activation of the encoder's hidden layers.
      layer_activations: Activation of the decoder's hidden layers. `None` uses
        `activation`.
      dtype: Precision of the parameters.
    """

    n_x: int
    hidden: tuple[int, ...]
    n_latent: int
    activation: ActivationSpec = "tanh"
    layer_activations: ActivationSpec | None = None
    dtype: DTypeLike = np.float32

    @property
    def enc_dims(self) -> tuple[int, ...]:
        """Encoder layer widths, input first."""
        return (self.n_x, *self.hidden, self.n_latent)

    @property
    def dec_dims(self) -> tuple[int, ...]:
        """Decoder layer widths, latent first."""
        return self.enc_dims[::-1]

    @property
    def enc_acts(self) -> tuple[str | Activation, ...]:
        """One activation per encoder hidden layer."""
        return resolve_activations(self.activation, len(self.hidden), "activation")

    @property
    def dec_acts(self) -> tuple[str | Activation, ...]:
        """One activation per decoder hidden layer."""
        spec = self.activation if self.layer_activations is None else self.layer_activations
        return resolve_activations(spec, len(self.hidden), "layer_activations")


# ── Network ───────────────────────────────────────────────────────────────────


def get_mlp_params(key: jax.Array, dims: Sequence[int], dtype: DTypeLike) -> list[Layer]:
    """Draws initial weights and biases for a stack of dense layers.

    Args:
      key: A `jax.random` key.
      dims: Layer widths, input first.
      dtype: Scalar type of the parameters.

    Returns:
      One layer per consecutive pair of widths, with weights of shape
      `(n_in, n_out)`.
    """
    layers: list[Layer] = []
    for n_in, n_out in zip(dims[:-1], dims[1:], strict=True):
        key, w_key, b_key = jax.random.split(key, 3)
        layers.append({"W": uniform(w_key, (n_in, n_out), n_in, dtype), "b": uniform(b_key, (n_out,), n_in, dtype)})
    return layers


def get_ae_params(key: jax.Array, cfg: AEJaxConfig) -> AEParams:
    """Draws initial parameters for the encoder and decoder.

    Args:
      key: A `jax.random` key.
      cfg: The architecture.

    Returns:
      The parameters.
    """
    enc_key, dec_key = jax.random.split(key)
    return {
        "enc": get_mlp_params(enc_key, cfg.enc_dims, cfg.dtype),
        "dec": get_mlp_params(dec_key, cfg.dec_dims, cfg.dtype),
    }


def mlp(layers: Sequence[Layer], x: jax.Array, acts: Sequence[str | Activation]) -> jax.Array:
    """Applies dense layers with an activation after each but the last.

    Args:
      layers: The layers, input first.
      x: Inputs, shape `(B, n_in)`.
      acts: One activation per hidden layer.

    Returns:
      Outputs, shape `(B, n_out)`.
    """
    h = x
    for layer, a in zip(layers[:-1], acts, strict=True):
        h = activation_fn(a)(h @ layer["W"] + layer["b"])
    return h @ layers[-1]["W"] + layers[-1]["b"]


def forward(params: AEParams, x: jax.Array, cfg: AEJaxConfig) -> jax.Array:
    """Encodes and decodes a batch.

    Args:
      params: The parameters.
      x: Scaled snapshots, shape `(B, N_x)`.
      cfg: The architecture.

    Returns:
      Reconstructions, shape `(B, N_x)`.
    """
    return mlp(params["dec"], mlp(params["enc"], x, cfg.enc_acts), cfg.dec_acts)


def mse_loss(params: AEParams, x: jax.Array, cfg: AEJaxConfig) -> jax.Array:
    """Computes the mean squared reconstruction error of a batch.

    Args:
      params: The parameters.
      x: Scaled snapshots, shape `(B, N_x)`.
      cfg: The architecture.

    Returns:
      The scalar loss.
    """
    return jnp.mean((forward(params, x, cfg) - x) ** 2)


# ── Compiled training step ────────────────────────────────────────────────────


@partial(jax.jit, static_argnames=("cfg", "wd"))
def _train_step(
    params: AEParams,
    opt_state: AdamState[AEParams],
    X: jax.Array,
    batches: jax.Array,
    i: int,
    lr: float,
    cfg: AEJaxConfig,
    wd: float,
) -> tuple[AEParams, AdamState[AEParams], jax.Array]:
    """Runs one Adam step on one batch, gathering the batch inside the compiled step.

    Args:
      params: The parameters.
      opt_state: Adam state.
      X: Every training snapshot, shape `(N_train, N_x)`.
      batches: Batch indices, shape `(n_batches, batch_size)`.
      i: Row of `batches` to train on.
      lr: Learning rate.
      cfg: The architecture.
      wd: L2 penalty coefficient.

    Returns:
      The updated parameters, the updated Adam state, and the batch loss.
    """
    loss, grads = jax.value_and_grad(mse_loss)(params, X[batches[i]], cfg)
    params, opt_state = adam_update(grads, opt_state, params, lr, wd=wd)
    return params, opt_state, loss


@partial(jax.jit, static_argnames="cfg")
def _eval_loss(params: AEParams, X: jax.Array, cfg: AEJaxConfig) -> jax.Array:
    """Computes the reconstruction loss over a block of snapshots.

    Args:
      params: The parameters.
      X: Scaled snapshots, shape `(N, N_x)`.
      cfg: The architecture.

    Returns:
      The scalar loss.
    """
    return mse_loss(params, X, cfg)


# ── Projector ─────────────────────────────────────────────────────────────────


@dataclass(eq=False, repr=False)
class AEJax(JaxAutoencoder[AEParams]):
    """A fully connected autoencoder in JAX.

    The encoder maps the scaled, zero-mean field through the hidden widths
    `hidden` to `n_latent` coefficients, and the decoder mirrors it. The
    bottleneck and output layers are linear. `Autoencoder` documents the
    training options.

    Attributes:
      hidden: Hidden layer widths of the encoder. The decoder uses them in
        reverse order.

    Raises:
      ValueError: If an option is out of range, an entry of `hidden` is below 1,
        or an activation option has the wrong length or an unknown name.
      TypeError: If an activation is neither a name nor a callable.
    """

    hidden: Sequence[int] = (512, 128)

    _cfg: AEJaxConfig | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range, an entry of `hidden` is
            below 1, or an activation option has the wrong length or an unknown
            name.
          TypeError: If an activation is neither a name nor a callable.
        """
        super().__post_init__()
        self.hidden = tuple(self.hidden)
        if any(d < 1 for d in self.hidden):
            raise ValueError(f"hidden must all be >= 1, got {self.hidden}")
        resolve_activations(self.activation, len(self.hidden), "activation")
        if self.layer_activations is not None:
            resolve_activations(self.layer_activations, len(self.hidden), "layer_activations")

    @property
    def cfg(self) -> AEJaxConfig:
        """The static architecture.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if self._cfg is None:
            raise RuntimeError("AEJax is not fitted; call fit() first")
        return self._cfg

    def fit(self, X: npt.NDArray[Any]) -> AEJax:
        """Trains the encoder and decoder on snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`, with NaN at solid
            points.

        Returns:
          This autoencoder, fitted.

        Raises:
          ValueError: If `X` is not a valid grid record, `val_fraction` leaves no
            training snapshots, or `dtype` is float64 without
            `jax_enable_x64`.
        """
        self._check_precision()
        Q, _ = self._prepare_fit(X)
        self._configure()
        cfg, wd = self.cfg, self.weight_decay

        def step(
            params: AEParams, opt_state: AdamState[AEParams], X_tr: jax.Array, batches: jax.Array, i: int, lr: float
        ) -> tuple[AEParams, AdamState[AEParams], jax.Array]:
            """Runs one compiled training step."""
            return cast(
                "tuple[AEParams, AdamState[AEParams], jax.Array]",
                _train_step(params, opt_state, X_tr, batches, i, lr, cfg, wd),
            )

        def loss(params: AEParams, X_val: jax.Array) -> jax.Array:
            """Computes the compiled validation loss."""
            return cast("jax.Array", _eval_loss(params, X_val, cfg))

        data = jnp.asarray((Q / self.scale).T, dtype=cfg.dtype)
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
        x = jnp.asarray((self.preprocess_snapshot(X) / self.scale).T, dtype=cfg.dtype)
        return np.asarray(mlp(params["enc"], x, cfg.enc_acts)).T

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
        params = self._trained_params()
        Z = self._check_latent(Z)
        Q_hat = mlp(params["dec"], jnp.asarray(Z.T, dtype=self.cfg.dtype), self.cfg.dec_acts)
        return np.asarray(Q_hat).T * self.scale + self.Q_mean

    def decode_scaled(self, Z: jax.Array) -> jax.Array:
        """Decodes a batch of latent codes differentiably.

        Args:
          Z: Latent codes, shape `(B, n_latent)`.

        Returns:
          Fields in the scaled units the network trains on, shape `(B, N_x)`.

        Raises:
          RuntimeError: If `fit` has not completed.
        """
        params = self._trained_params()
        return mlp(params["dec"], jnp.asarray(Z, dtype=self.cfg.dtype), self.cfg.dec_acts)

    def _configure(self) -> None:
        """Builds the architecture for the recorded field size."""
        self._cfg = AEJaxConfig(
            n_x=int(self.Q_mean.shape[0]),
            hidden=tuple(self.hidden),
            n_latent=self.n_latent,
            activation=self.activation,
            layer_activations=self.layer_activations,
            dtype=self.dtype,
        )

    def _init_params(self, key: jax.Array) -> AEParams:
        """Draws initial parameters.

        Args:
          key: A `jax.random` key.

        Returns:
          The parameters.
        """
        return get_ae_params(key, self.cfg)
