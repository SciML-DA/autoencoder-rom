"""Base classes shared by every projector.

Defines `Projector`, the interface POD, SPOD, and the autoencoders implement,
and `Autoencoder`, the options, validation, training history, and save/restore
methods the four autoencoders share. Also computes the layer geometry of the
convolutional autoencoders.

Every projector works on snapshots in one of three layouts:

    grid      (Nu, N_t, Nx, Ny)   NaN at solid points
    snapshot  (Nu, Nx, Ny)        one grid snapshot
    flat      (N_x, N_t)          fluid points only, N_x = Nu * N_fluid

The flat layout groups rows by component: row `u * N_fluid + f` holds component
`u` at the `f`-th fluid point.

Typical usage example:

  class MyProjector(Projector):
      def fit(self, X): ...
      def encode(self, X): ...
      def decode(self, Z): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple, Self

import numpy as np
import numpy.typing as npt

from ..configurable import Configurable, restore_tuples
from ..training import TrainingHistory

__all__ = [
    "Autoencoder",
    "DecoderStage",
    "EncoderStage",
    "FloatArray",
    "Projector",
    "conv_stages",
]

#: A real-valued array. Each parameter documents its own shape.
FloatArray = npt.NDArray[np.floating[Any]]
#: A boolean mask over the grid points.
BoolArray = npt.NDArray[np.bool_]


class Projector(ABC):
    """Reduces flow snapshots to latent coefficients and reconstructs them.

    Subclasses implement `fit`, `encode`, and `decode`. `fit` records the grid
    geometry and the temporal mean the other methods use.

    Attributes:
      N_latent: Dimension of the latent space.
      fitted: Whether `fit` has completed.
      grid_shape: Grid shape `(Nu, Nx, Ny)`.
      domain: Physical extent `[x0, x1, y0, y1]` of the grid.
      fluid_mask_flat: Mask over the `Nx * Ny` grid points, `True` at fluid
        points.
    """

    N_latent: int = 20
    fitted: bool = False
    grid_shape: tuple[int, int, int] | None = None
    domain: list[float] | None = None
    fluid_mask_flat: BoolArray | None = None
    _Q_mean: FloatArray | None = None
    _TKE: float | None = None

    # ── Interface ─────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, X: npt.NDArray[Any]) -> Projector:
        """Learns the projection from snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`.

        Returns:
          This projector, fitted.
        """

    @abstractmethod
    def encode(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Encodes snapshots into latent coefficients.

        Args:
          X: Snapshots in the grid, snapshot, or flat layout.

        Returns:
          Latent coefficients, shape `(N_latent, N_t)`.
        """

    @abstractmethod
    def decode(self, Z: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Decodes latent coefficients into flat fields.

        Args:
          Z: Latent coefficients, shape `(N_latent, N_t)`.

        Returns:
          Flat fields with the temporal mean restored, shape `(N_x, N_t)`.
        """

    @property
    def Q_mean(self) -> FloatArray:
        """The temporal mean of the training fields, shape `(N_x, 1)`.

        Raises:
          AttributeError: If `fit` has not been called.
        """
        if self._Q_mean is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Q_mean

    @Q_mean.setter
    def Q_mean(self, value: FloatArray) -> None:
        self._Q_mean = value

    def reconstruct(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Encodes and then decodes snapshots.

        Args:
          X: Snapshots in the grid, snapshot, or flat layout.

        Returns:
          Flat reconstructed fields, shape `(N_x, N_t)`.
        """
        return self.decode(self.encode(X))

    def score(self, X: npt.NDArray[Any]) -> float:
        """Computes the mean squared reconstruction error.

        Args:
          X: Snapshots in the grid, snapshot, or flat layout.

        Returns:
          The mean squared error over every fluid point and snapshot.
        """
        X_hat = self.reconstruct(X)
        return float(np.mean((self._flat(X) - X_hat) ** 2))

    # ── Sensor readout ────────────────────────────────────────────────────────

    def decode_at(self, Z: npt.NDArray[Any], idx: npt.NDArray[np.intp] | None = None) -> npt.NDArray[Any]:
        """Decodes latent coefficients and keeps the requested rows.

        Args:
          Z: Latent coefficients, shape `(N_latent, N_t)`.
          idx: Rows of the flat field to keep, such as sensor locations. `None`
            keeps every row.

        Returns:
          The selected rows of the decoded field, shape `(len(idx), N_t)`.
        """
        X_hat = self.decode(Z)
        return X_hat if idx is None else X_hat[idx]

    def spatial_basis(self, z0: npt.NDArray[Any] | None = None) -> FloatArray:
        """Computes the decoder Jacobian at a latent state.

        Each column is the change in the decoded field per unit change in one
        latent coefficient, computed by central differences on `decode`.

        Args:
          z0: Latent state to linearize about, shape `(N_latent,)`. `None` uses
            the origin.

        Returns:
          The Jacobian, shape `(N_x, N_latent)`.

        Raises:
          ValueError: If `z0` does not have `N_latent` entries.
        """
        n = self.N_latent
        z: FloatArray = np.zeros(n) if z0 is None else np.asarray(z0, dtype=np.float64).ravel()
        if z.size != n:
            raise ValueError(f"z0 has {z.size} entries, expected N_latent={n}")

        eps: FloatArray = 1e-4 * np.maximum(np.abs(z), 1.0)
        cols: list[FloatArray] = []
        for k in range(n):
            zp, zm = z.copy(), z.copy()
            zp[k] += eps[k]
            zm[k] -= eps[k]
            dp: FloatArray = self.decode(zp[:, np.newaxis])
            dm: FloatArray = self.decode(zm[:, np.newaxis])
            cols.append((dp - dm).ravel() / (2.0 * eps[k]))
        return np.column_stack(cols)

    # ── Layouts ───────────────────────────────────────────────────────────────

    def preprocess_snapshot(self, X: npt.NDArray[Any], subtract_mean: bool = True) -> FloatArray:
        """Converts snapshots to zero-mean flat fields.

        Before `fit` completes, also records the grid shape, the fluid mask, and
        the temporal mean from `X`.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`. After `fit`, the
            snapshot and flat layouts are accepted too.
          subtract_mean: Record the temporal mean of `X` rather than zeros. Used
            only before `fit` completes.

        Returns:
          Zero-mean flat fields, shape `(N_x, N_t)`.

        Raises:
          ValueError: If `X` does not match an accepted layout or the fitted
            grid, or holds non-finite values at fluid points.
        """
        if not self.fitted:
            return self._fit_geometry(X, subtract_mean)
        return self._flat(X) - self.Q_mean

    def _fit_geometry(self, X: npt.NDArray[Any], subtract_mean: bool = True) -> FloatArray:
        """Records the grid shape, fluid mask, and temporal mean of snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`, with NaN at solid
            points.
          subtract_mean: Record the temporal mean of `X` rather than zeros.

        Returns:
          Zero-mean flat fields, shape `(N_x, N_t)`.

        Raises:
          ValueError: If `X` is not four-dimensional, or holds non-finite values
            at fluid points.
        """
        X = np.asarray(X)
        if X.ndim != 4:
            raise ValueError(f"expected grid snapshots (Nu, N_t, Nx, Ny), got shape {X.shape}")
        Nu, _, Nx, Ny = X.shape
        self.grid_shape = (Nu, Nx, Ny)
        self.fluid_mask_flat = ~np.isnan(X[0, 0]).ravel()

        X_flat = self._to_flat(X)
        if not np.isfinite(X_flat).all():
            raise ValueError("snapshots hold non-finite values at points that are fluid in the first snapshot")
        self.Q_mean = X_flat.mean(axis=1, keepdims=True) if subtract_mean else np.zeros_like(X_flat[:, :1])
        Q = X_flat - self.Q_mean
        self._TKE = 0.5 * float(np.sum(np.mean(Q**2, axis=1)))
        return Q

    def _flat(self, X: npt.NDArray[Any]) -> FloatArray:
        """Converts snapshots in any accepted layout to flat fields.

        Args:
          X: Snapshots in the grid, snapshot, or flat layout.

        Returns:
          Flat fields, mean included, shape `(N_x, N_t)`.

        Raises:
          ValueError: If `X` does not match an accepted layout or the fitted
            grid.
        """
        X = np.asarray(X)
        n_x = self.Q_mean.shape[0]
        if X.ndim == 2:
            if X.shape[0] != n_x:
                raise ValueError(f"flat snapshots must have N_x={n_x} rows, got shape {X.shape}")
            return X
        if X.ndim == 3:
            X = X[:, np.newaxis]
        if X.ndim != 4:
            raise ValueError(f"expected grid, snapshot, or flat input, got shape {X.shape}")
        Nu, _, Nx, Ny = X.shape
        if (Nu, Nx, Ny) != self.grid_shape:
            raise ValueError(f"expected grid shape {self.grid_shape}, got {(Nu, Nx, Ny)}")
        return self._to_flat(X)

    def _to_flat(self, X: FloatArray) -> FloatArray:
        """Selects the fluid points of grid snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`.

        Returns:
          Flat fields, shape `(Nu * N_fluid, N_t)`, grouped by component.
        """
        mask = self._fluid_mask()
        X_fluid = X.reshape(X.shape[0], X.shape[1], -1)[:, :, mask]
        return X_fluid.transpose(0, 2, 1).reshape(-1, X.shape[1])

    def _to_physical_grid(self, X_hat: FloatArray) -> FloatArray:
        """Places flat fields back on the grid.

        Args:
          X_hat: Flat fields, shape `(N_x, N_t)` or `(N_x,)`.

        Returns:
          Fields on the grid with NaN at solid points, shape `(Nu, N_t, Nx, Ny)`,
          or `(Nu, Nx, Ny)` for a single snapshot.
        """
        Nu, Nx, Ny = self._grid()
        mask = self._fluid_mask()
        if X_hat.ndim == 1:
            X_hat = X_hat[:, np.newaxis]
        n_t = X_hat.shape[1]

        values = X_hat.reshape(Nu, int(mask.sum()), n_t).transpose(0, 2, 1)
        out: FloatArray = np.full((Nu, n_t, Nx * Ny), np.nan)
        out[:, :, mask] = values
        out = out.reshape(Nu, n_t, Nx, Ny)
        return out[:, 0] if n_t == 1 else out

    def _field_scale(self, Q: FloatArray) -> FloatArray:
        """Computes the standard deviation of each field component.

        Args:
          Q: Zero-mean flat fields, shape `(N_x, N_t)`.

        Returns:
          The standard deviation of each row's component, shape `(N_x, 1)`. A
          component with zero variance gets 1.
        """
        Nu = self._grid()[0]
        n_fluid = Q.shape[0] // Nu
        scale: FloatArray = np.ones((Q.shape[0], 1), dtype=Q.dtype)
        for u in range(Nu):
            rows = slice(u * n_fluid, (u + 1) * n_fluid)
            s = float(Q[rows].std())
            scale[rows, 0] = s if s > 0 else 1.0
        return scale

    def _grid(self) -> tuple[int, int, int]:
        """Returns the grid shape.

        Raises:
          RuntimeError: If the grid shape is not known.
        """
        if self.grid_shape is None:
            raise RuntimeError("grid_shape is unknown; fit on grid snapshots first")
        return self.grid_shape

    def _fluid_mask(self) -> BoolArray:
        """Returns the fluid mask.

        Raises:
          RuntimeError: If the fluid mask is not known.
        """
        if self.fluid_mask_flat is None:
            raise RuntimeError("fluid_mask_flat is unknown; fit on grid snapshots first")
        return self.fluid_mask_flat


@dataclass(eq=False, repr=False)
class Autoencoder(Projector, Configurable):
    """Options, validation, and training state shared by the autoencoders.

    `fit` scales each field component by its standard deviation, trains on the
    mean squared reconstruction error with Adam, holds out the last
    `val_fraction` of the snapshots, reduces the learning rate when the
    validation loss plateaus, stops early, and keeps the parameters with the
    lowest validation loss.

    Attributes:
      n_latent: Dimension of the latent space. `N_latent` holds the same value.
      learning_rate: Initial Adam step size.
      epochs: Maximum number of training epochs.
      batch_size: Minibatch size.
      val_fraction: Fraction of the snapshots held out for validation, taken
        from the end of the record. 0 trains on every snapshot and disables
        early stopping.
      weight_decay: L2 penalty coefficient.
      patience: Epochs without improvement before training stops.
      threshold: Relative improvement in the validation loss that counts as
        progress, for both early stopping and the learning-rate schedule.
      lr_factor: Factor the learning rate decays by on a plateau.
      lr_patience: Plateau epochs before the learning rate decays.
      min_lr: Lower bound on the learning rate.
      seed: Random seed for initialization and shuffling.
      grid_shape: Grid shape `(Nu, Nx, Ny)`. `fit` sets it from the data.
      domain: Physical extent `[x0, x1, y0, y1]` of the grid.
      training_history: Losses and learning rates of the last `fit`.

    Raises:
      ValueError: If an option is out of range or `grid_shape` does not have
        three entries.
    """

    n_latent: int = 10
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    patience: int = 50
    threshold: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-6
    seed: int = 0
    grid_shape: tuple[int, int, int] | None = None
    domain: list[float] | None = None

    training_history: TrainingHistory = field(default_factory=TrainingHistory, init=False)
    _scale: FloatArray | None = field(default=None, init=False)

    _config_exclude: ClassVar[tuple[str, ...]] = ("grid_shape",)

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range or `grid_shape` does not have
            three entries.
        """
        self.N_latent = self.n_latent
        if self.grid_shape is not None and len(self.grid_shape) != 3:
            raise ValueError(f"grid_shape must be (Nu, Nx, Ny), got {self.grid_shape}")
        if self.n_latent < 1:
            raise ValueError(f"n_latent must be >= 1, got {self.n_latent}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {self.threshold}")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError(f"lr_factor must be in (0, 1), got {self.lr_factor}")
        if self.lr_patience < 0:
            raise ValueError(f"lr_patience must be >= 0, got {self.lr_patience}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}")

    @property
    def scale(self) -> FloatArray:
        """The per-field scale inputs are divided by, shape `(N_x, 1)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if self._scale is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        return self._scale

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of trainable parameters, or 0 before `fit`."""

    def _prepare_fit(self, X: npt.NDArray[Any]) -> tuple[FloatArray, int]:
        """Records the geometry and field scale of the training snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`.

        Returns:
          The zero-mean flat fields, shape `(N_x, N_t)`, and the number of
          snapshots held out for validation.

        Raises:
          ValueError: If `X` is not a valid grid record, or `val_fraction`
            leaves no training snapshots.
        """
        self.fitted = False
        Q = self._fit_geometry(X)
        n_t = Q.shape[1]
        n_val = int(round(self.val_fraction * n_t))
        if n_t - n_val < 1:
            raise ValueError(f"val_fraction={self.val_fraction} leaves no training snapshots out of {n_t}")
        self._scale = self._field_scale(Q)
        return Q, n_val

    def _finish_fit(self, history: TrainingHistory) -> None:
        """Records the training history and marks the autoencoder fitted.

        Args:
          history: The history the training loop returned.
        """
        self.training_history = history
        self.fitted = True

    def _check_fitted(self) -> None:
        """Raises if `fit` has not completed.

        Raises:
          RuntimeError: If `fit` has not completed.
        """
        if not self.fitted or self._scale is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")

    def _check_latent(self, Z: npt.NDArray[Any]) -> FloatArray:
        """Checks the shape of latent coefficients.

        Args:
          Z: Latent coefficients.

        Returns:
          `Z` as an array, shape `(n_latent, N_t)`.

        Raises:
          ValueError: If `Z` is not two-dimensional with `n_latent` rows.
        """
        Z = np.asarray(Z)
        if Z.ndim != 2 or Z.shape[0] != self.N_latent:
            raise ValueError(f"Z must be (n_latent={self.N_latent}, N_t), got shape {Z.shape}")
        return Z

    # ── Saving and restoring ──────────────────────────────────────────────────

    @classmethod
    def from_data(cls, data: npt.NDArray[Any], **options: Any) -> Self:
        """Builds and fits an autoencoder.

        Args:
          data: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`.
          **options: Constructor options.

        Returns:
          The fitted autoencoder.
        """
        model = cls(**options)
        model.fit(data)
        return model

    def trained_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects the geometry, field statistics, history, and weights.

        Returns:
          The arrays, by name.

        Raises:
          RuntimeError: If `fit` has not completed.
        """
        self._check_fitted()
        arrays: dict[str, npt.NDArray[Any]] = {
            "Q_mean": self.Q_mean,
            "scale": self.scale,
            "grid_shape": np.asarray(self._grid()),
            "fluid_mask_flat": self._fluid_mask(),
            **self.training_history.to_arrays(),
        }
        if self._TKE is not None:
            arrays["TKE"] = np.asarray(self._TKE)
        return arrays | self._weight_arrays()

    @classmethod
    def from_trained(cls, options: Mapping[str, Any], arrays: Mapping[str, npt.NDArray[Any]], **overrides: Any) -> Self:
        """Rebuilds a fitted autoencoder.

        Args:
          options: Options from `config_options`. Options the class no longer
            defines are ignored.
          arrays: Arrays from `trained_arrays`.
          **overrides: Options to use instead of the stored ones, such as
            `device`.

        Returns:
          The fitted autoencoder.
        """
        stored = {k: v if k == "domain" else restore_tuples(v) for k, v in options.items()}
        model = cls(**cls._known_options(stored, overrides))
        model.Q_mean = np.asarray(arrays["Q_mean"])
        model._scale = np.asarray(arrays["scale"])
        Nu, Nx, Ny = (int(v) for v in arrays["grid_shape"])
        model.grid_shape = (Nu, Nx, Ny)
        model.fluid_mask_flat = np.asarray(arrays["fluid_mask_flat"], dtype=bool)
        if "TKE" in arrays:
            model._TKE = float(arrays["TKE"])
        model._load_weight_arrays(arrays)
        model._finish_fit(TrainingHistory.from_arrays(arrays))
        return model

    @abstractmethod
    def _weight_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects the trained weights.

        Returns:
          The weights, by name.
        """

    @abstractmethod
    def _load_weight_arrays(self, arrays: Mapping[str, npt.NDArray[Any]]) -> None:
        """Builds the networks from the recorded geometry and loads trained weights.

        Args:
          arrays: Arrays holding the entries `_weight_arrays` returned.
        """


# ── Convolutional geometry ────────────────────────────────────────────────────


class EncoderStage(NamedTuple):
    """One convolution of the convolutional encoder.

    Attributes:
      c_in: Input channels.
      c_out: Output channels.
      size_in: Input grid size `(Nx, Ny)`.
      size_out: Output grid size.
    """

    c_in: int
    c_out: int
    size_in: tuple[int, int]
    size_out: tuple[int, int]


class DecoderStage(NamedTuple):
    """One transposed convolution of the convolutional decoder.

    Attributes:
      c_in: Input channels.
      c_out: Output channels.
      output_padding: Extra rows and columns that restore the encoder's input
        size.
      size_out: Output grid size `(Nx, Ny)`.
    """

    c_in: int
    c_out: int
    output_padding: tuple[int, int]
    size_out: tuple[int, int]


def conv_stages(
    grid_shape: tuple[int, int, int],
    channels: Sequence[int],
    kernel_size: int,
    stride: int,
    pad: int,
) -> tuple[tuple[EncoderStage, ...], tuple[DecoderStage, ...]]:
    """Computes the layer sizes of a convolutional autoencoder.

    The decoder mirrors the encoder, so each transposed convolution restores the
    grid size and channel count of the matching convolution's input.

    Args:
      grid_shape: Input grid shape `(Nu, Nx, Ny)`.
      channels: Output channels of each encoder convolution.
      kernel_size: Convolution kernel size.
      stride: Convolution stride.
      pad: Convolution padding.

    Returns:
      The encoder stages, input first, and the decoder stages, bottleneck first.

    Raises:
      ValueError: If a convolution's input is smaller than its padded kernel.
    """
    k, s, p = kernel_size, stride, pad
    c, h, w = grid_shape
    encoder: list[EncoderStage] = []
    for i, c_out in enumerate(channels):
        if h + 2 * p < k or w + 2 * p < k:
            raise ValueError(
                f"stage {i} input {h}x{w} is smaller than kernel_size={k} with pad={p}; use fewer entries in channels"
            )
        size_out = ((h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1)
        encoder.append(EncoderStage(c, c_out, (h, w), size_out))
        c, (h, w) = c_out, size_out

    decoder: list[DecoderStage] = []
    for stage in reversed(encoder):
        (h, w), (th, tw) = stage.size_out, stage.size_in
        padding = (th - ((h - 1) * s - 2 * p + k), tw - ((w - 1) * s - 2 * p + k))
        decoder.append(DecoderStage(stage.c_out, stage.c_in, padding, stage.size_in))
    return tuple(encoder), tuple(decoder)
