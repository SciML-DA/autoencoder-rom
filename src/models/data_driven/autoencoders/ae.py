"""Dense autoencoder in PyTorch.

Typical usage example:

  ae = AE(n_latent=10, layer_dims=(128, 32)).fit(X)
  Z = ae.encode(X)
  Q_hat = ae.decode(Z)
  mse = ae.score(X)
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt
import torch
from torch import nn

from .base import FloatArray
from .torch_utils import TorchAutoencoder

__all__ = ["AE"]


@dataclass(eq=False, repr=False)
class AE(TorchAutoencoder):
    """A fully connected autoencoder.

    The encoder maps the scaled, zero-mean field through the hidden widths
    `layer_dims` to `n_latent` coefficients, and the decoder mirrors it. Hidden
    layers use `activation_function`; the bottleneck and output layers are
    linear. `Autoencoder` documents the training options.

    Attributes:
      layer_dims: Hidden layer widths of the encoder. The decoder uses them in
        reverse order.

    Raises:
      ValueError: If an option is out of range, an entry of `layer_dims` is
        below 1, `activation_function` is unknown, or `device` is not a valid
        torch device name.
    """

    layer_dims: Sequence[int] = (512, 128)

    _encoder: nn.Sequential | None = field(default=None, init=False)
    _decoder: nn.Sequential | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range, an entry of `layer_dims` is
            below 1, `activation_function` is unknown, or `device` is not a
            valid torch device name.
        """
        super().__post_init__()
        self.layer_dims = tuple(self.layer_dims)
        if any(d < 1 for d in self.layer_dims):
            raise ValueError(f"layer_dims must all be >= 1, got {self.layer_dims}")

    # ── Networks ──────────────────────────────────────────────────────────────

    @property
    def encoder(self) -> nn.Sequential:
        """The encoder network.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._encoder is None:
            raise RuntimeError("AE is not fitted; call fit() first")
        return self._encoder

    @property
    def decoder(self) -> nn.Sequential:
        """The decoder network.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._decoder is None:
            raise RuntimeError("AE is not fitted; call fit() first")
        return self._decoder

    @property
    def n_params(self) -> int:
        """Number of trainable parameters, or 0 before the networks are built."""
        if self._encoder is None or self._decoder is None:
            return 0
        return sum(p.numel() for net in (self._encoder, self._decoder) for p in net.parameters())

    def _networks_by_name(self) -> dict[str, nn.Module]:
        """Lists the encoder and decoder by name.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return {"encoder": self.encoder, "decoder": self.decoder}

    def _build_networks(self) -> None:
        """Creates freshly initialized encoder and decoder networks for the recorded field size."""
        n_x = int(self.Q_mean.shape[0])
        self._encoder = self._mlp([n_x, *self.layer_dims, self.N_latent]).to(self.device)
        self._decoder = self._mlp([self.N_latent, *reversed(self.layer_dims), n_x]).to(self.device)

    def _mlp(self, dims: Sequence[int]) -> nn.Sequential:
        """Builds linear layers with an activation between consecutive layers.

        Args:
          dims: Layer widths, input first.

        Returns:
          The network. Its last layer is linear.
        """
        layers: list[nn.Module] = []
        for n_in, n_out in itertools.pairwise(dims):
            if layers:
                layers.append(self._activation())
            layers.append(nn.Linear(n_in, n_out))
        return nn.Sequential(*layers)

    # ── Projector ─────────────────────────────────────────────────────────────

    def fit(self, X: npt.NDArray[Any]) -> AE:
        """Trains the encoder and decoder on snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`, with NaN at solid
            points.

        Returns:
          This autoencoder, fitted.

        Raises:
          ValueError: If `X` is not a valid grid record, or `val_fraction`
            leaves no training snapshots.
        """
        Q, n_val = self._prepare_fit(X)
        n_t = Q.shape[1]
        self._build_networks()
        encoder, decoder = self.encoder, self.decoder

        def loss_fn(batch: torch.Tensor) -> torch.Tensor:
            """Computes the mean squared reconstruction error of a batch."""
            return nn.functional.mse_loss(decoder(encoder(batch)), batch)

        data = self._to_tensor(Q)
        history = self._train((encoder, decoder), loss_fn, data[: n_t - n_val], data[n_t - n_val :])
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
        self._check_fitted()
        encoder = self.encoder
        encoder.eval()
        with torch.no_grad():
            Z: torch.Tensor = encoder(self._to_tensor(self.preprocess_snapshot(X)))
        return Z.cpu().numpy().T

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
        self._check_fitted()
        Z = self._check_latent(Z)
        decoder = self.decoder
        decoder.eval()
        with torch.no_grad():
            Q_hat: torch.Tensor = decoder(torch.as_tensor(Z.T, dtype=torch.float32, device=self.device))
        return Q_hat.cpu().numpy().T * self.scale + self.Q_mean

    def _to_tensor(self, Q: FloatArray) -> torch.Tensor:
        """Scales zero-mean flat fields and converts them to a float32 tensor.

        Args:
          Q: Zero-mean flat fields, shape `(N_x, N_t)`.

        Returns:
          Scaled fields, one snapshot per row, shape `(N_t, N_x)`.
        """
        return torch.as_tensor((Q / self.scale).T, dtype=torch.float32, device=self.device)
