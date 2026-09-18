"""Convolutional autoencoder in PyTorch.

Typical usage example:

  cae = CAE(n_latent=8, channels=(16, 32, 64)).fit(X)
  Z = cae.encode(X)
  Q_hat = cae.decode(Z)
  mse = cae.score(X)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .base import FloatArray, conv_stages
from .torch_utils import TorchAutoencoder

__all__ = ["CAE"]


@dataclass(eq=False, repr=False)
class CAE(TorchAutoencoder):
    """A convolutional autoencoder on the snapshot grid.

    The encoder applies one convolution per entry of `channels`, each followed
    by `activation`, then flattens and maps linearly to `n_latent`
    coefficients. The decoder maps linearly back to the bottleneck grid and
    applies the mirrored transposed convolutions, with the last one linear.
    Solid points enter as zeros and are excluded from the loss. `Autoencoder`
    documents the training options.

    Attributes:
      channels: Output channels of each encoder convolution.
      kernel_size: Convolution kernel size.
      stride: Convolution stride.
      pad: Convolution padding.

    Raises:
      ValueError: If an option is out of range, `activation` is
        unknown, or `device` is not a valid torch device name.
    """

    channels: Sequence[int] = (16, 32, 64)
    kernel_size: int = 3
    stride: int = 2
    pad: int = 1

    _enc_conv: nn.Sequential | None = field(default=None, init=False)
    _enc_fc: nn.Linear | None = field(default=None, init=False)
    _dec_fc: nn.Linear | None = field(default=None, init=False)
    _dec_conv: nn.Sequential | None = field(default=None, init=False)
    _red_shape: tuple[int, int, int] = field(default=(0, 0, 0), init=False)

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range, `activation` is
            unknown, or `device` is not a valid torch device name.
        """
        super().__post_init__()
        self.channels = tuple(self.channels)
        if not self.channels or any(c < 1 for c in self.channels):
            raise ValueError(f"channels must be non-empty with entries >= 1, got {self.channels}")
        if self.kernel_size < 1 or self.stride < 1 or self.pad < 0:
            raise ValueError(
                f"need kernel_size >= 1, stride >= 1, pad >= 0; got {self.kernel_size}, {self.stride}, {self.pad}"
            )

    # ── Networks ──────────────────────────────────────────────────────────────

    @property
    def enc_conv(self) -> nn.Sequential:
        """The convolutional encoder stages.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._enc_conv is None:
            raise RuntimeError("CAE is not fitted; call fit() first")
        return self._enc_conv

    @property
    def enc_fc(self) -> nn.Linear:
        """The linear map from the bottleneck grid to the latent space.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._enc_fc is None:
            raise RuntimeError("CAE is not fitted; call fit() first")
        return self._enc_fc

    @property
    def dec_fc(self) -> nn.Linear:
        """The linear map from the latent space to the bottleneck grid.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._dec_fc is None:
            raise RuntimeError("CAE is not fitted; call fit() first")
        return self._dec_fc

    @property
    def dec_conv(self) -> nn.Sequential:
        """The transposed-convolution decoder stages.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._dec_conv is None:
            raise RuntimeError("CAE is not fitted; call fit() first")
        return self._dec_conv

    @property
    def n_params(self) -> int:
        """Number of trainable parameters, or 0 before the networks are built."""
        if self._enc_conv is None:
            return 0
        return sum(p.numel() for net in self.networks() for p in net.parameters())

    def networks(self) -> list[nn.Module]:
        """Lists the networks in the order the encoder and decoder apply them.

        Returns:
          The convolutional encoder, the encoder's linear map, the decoder's
          linear map, and the transposed-convolution decoder.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return [self.enc_conv, self.enc_fc, self.dec_fc, self.dec_conv]

    def encode_grid(self, G: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of scaled grid snapshots, differentiably.

        Args:
          G: Scaled snapshots with zeros at solid points, shape
            `(B, Nu, Nx, Ny)`.

        Returns:
          Latent codes, shape `(B, n_latent)`.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return self.enc_fc(self.enc_conv(G).flatten(1))

    def decode_grid(self, Z: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of latent codes onto the grid, differentiably.

        Args:
          Z: Latent codes, shape `(B, n_latent)`.

        Returns:
          Scaled fields, shape `(B, Nu, Nx, Ny)`.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return self.dec_conv(self.dec_fc(Z).view(-1, *self._red_shape))

    def _networks_by_name(self) -> dict[str, nn.Module]:
        """Lists the networks by name, in the order `networks` returns them.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return {"enc_conv": self.enc_conv, "enc_fc": self.enc_fc, "dec_fc": self.dec_fc, "dec_conv": self.dec_conv}

    def _build_networks(self) -> None:
        """Creates freshly initialized encoder and decoder networks for the recorded grid.

        Raises:
          ValueError: If the grid is too small for the convolution stages.
        """
        c_in, nx, ny = self._grid()
        k, s, p = self.kernel_size, self.stride, self.pad
        encoder, decoder = conv_stages((c_in, nx, ny), self.channels, k, s, p)
        c_red, (h_red, w_red) = encoder[-1].c_out, encoder[-1].size_out
        self._red_shape = (c_red, h_red, w_red)
        flat = c_red * h_red * w_red

        enc: list[nn.Module] = []
        for stage in encoder:
            enc += [nn.Conv2d(stage.c_in, stage.c_out, k, s, p), self._activation()]
        self._enc_conv = nn.Sequential(*enc).to(self.device)
        self._enc_fc = nn.Linear(flat, self.N_latent).to(self.device)
        self._dec_fc = nn.Linear(self.N_latent, flat).to(self.device)

        dec: list[nn.Module] = []
        for stage in decoder:
            if dec:
                dec.append(self._activation())
            dec.append(nn.ConvTranspose2d(stage.c_in, stage.c_out, k, s, p, output_padding=stage.output_padding))
        self._dec_conv = nn.Sequential(*dec).to(self.device)

    # ── Projector ─────────────────────────────────────────────────────────────

    def fit(self, X: npt.NDArray[Any]) -> CAE:
        """Trains the encoder and decoder on snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`, with NaN at solid
            points.

        Returns:
          This autoencoder, fitted.

        Raises:
          ValueError: If `X` is not a valid grid record, `val_fraction` leaves no
            training snapshots, or the grid is too small for the convolution
            stages.
        """
        Q, n_val = self._prepare_fit(X)
        Nu, Nx, Ny = self._grid()
        n_t = Q.shape[1]
        self._build_networks()
        mask = torch.as_tensor(self._fluid_mask().reshape(Nx, Ny), dtype=torch.float32, device=self.device)[None, None]

        def loss_fn(batch: torch.Tensor) -> torch.Tensor:
            """Computes the mean squared reconstruction error over fluid points."""
            recon = self.decode_grid(self.encode_grid(batch))
            return ((recon - batch) ** 2 * mask).sum() / (mask.sum() * batch.shape[0] * Nu)

        data = self._to_tensor(Q)
        history = self._train(self.networks(), loss_fn, data[: n_t - n_val], data[n_t - n_val :])
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
        G = self._to_tensor(self.preprocess_snapshot(X))
        for net in self.networks():
            net.eval()
        with torch.no_grad():
            Z = self.encode_grid(G)
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
        for net in self.networks():
            net.eval()
        with torch.no_grad():
            G = self.decode_grid(torch.as_tensor(Z.T, dtype=torch.float32, device=self.device))
        return self._grid_to_flat(G.cpu().numpy()) * self.scale + self.Q_mean

    # ── Layouts ───────────────────────────────────────────────────────────────

    def _to_tensor(self, Q: FloatArray) -> torch.Tensor:
        """Scales zero-mean flat fields and places them on the grid as a tensor.

        Args:
          Q: Zero-mean flat fields, shape `(N_x, N_t)`.

        Returns:
          Scaled float32 fields with zeros at solid points, shape
          `(N_t, Nu, Nx, Ny)`.
        """
        Nu, Nx, Ny = self._grid()
        mask = self._fluid_mask()
        n_t = Q.shape[1]
        values = (Q / self.scale).reshape(Nu, int(mask.sum()), n_t).transpose(0, 2, 1)
        grid: FloatArray = np.zeros((Nu, n_t, Nx * Ny), dtype=Q.dtype)
        grid[:, :, mask] = values
        G = grid.reshape(Nu, n_t, Nx, Ny).transpose(1, 0, 2, 3)
        return torch.as_tensor(G, dtype=torch.float32, device=self.device)

    def _grid_to_flat(self, G: FloatArray) -> FloatArray:
        """Selects the fluid points of fields on the grid.

        Args:
          G: Fields, shape `(N_t, Nu, Nx, Ny)`.

        Returns:
          Flat fields, shape `(N_x, N_t)`.
        """
        return self._to_flat(G.transpose(1, 0, 2, 3))
