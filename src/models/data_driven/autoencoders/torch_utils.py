"""Base class, training loop, and weight saving shared by the PyTorch autoencoders.

Typical usage example:

  @dataclass(eq=False, repr=False)
  class MyAE(TorchAutoencoder):
      def fit(self, X):
          Q, n_val = self._prepare_fit(X)
          ...
          self._finish_fit(self._train(nets, loss_fn, X_tr, X_val))
          return self
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from ..training import TrainingHistory
from .base import Autoencoder, FloatArray

__all__ = ["ACTIVATIONS", "TorchAutoencoder", "adam_options", "copy_state", "snapshot_buffers"]

#: Hidden layer activations, by name.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "identity": nn.Identity,
}

#: One state dictionary per network, holding a copy of its weights.
StateBuffers = list[dict[str, torch.Tensor]]
#: Maps a batch of scaled snapshots to the scalar training loss.
LossFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(eq=False, repr=False)
class TorchAutoencoder(Autoencoder):
    """Options and training loop shared by `AE` and `CAE`.

    Attributes:
      activation: Hidden layer activation. One of `"tanh"`, `"relu"`,
        `"elu"`, or `"identity"`.
      device: Torch device name, such as `"cpu"` or `"cuda"`. Saved
        configurations leave it out; pass it when restoring.

    Raises:
      ValueError: If an option is out of range, `activation` is
        unknown, or `device` is not a valid torch device name.
    """

    activation: str = "tanh"
    device: str = "cpu"

    _config_exclude: ClassVar[tuple[str, ...]] = ("grid_shape", "device")

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range, `activation` is
            unknown, or `device` is not a valid torch device name.
        """
        super().__post_init__()
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(ACTIVATIONS)}, got {self.activation!r}"
            )
        try:
            torch.device(self.device)
        except RuntimeError as e:
            raise ValueError(f"device {self.device!r} is not a valid torch device") from e

    def _prepare_fit(self, X: npt.NDArray[Any]) -> tuple[FloatArray, int]:
        """Records the geometry and field scale, then seeds torch.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`.

        Returns:
          The zero-mean flat fields, shape `(N_x, N_t)`, and the number of
          snapshots held out for validation.

        Raises:
          ValueError: If `X` is not a valid grid record, or `val_fraction`
            leaves no training snapshots.
        """
        prepared = super()._prepare_fit(X)
        torch.manual_seed(self.seed)
        return prepared

    @abstractmethod
    def _networks_by_name(self) -> dict[str, nn.Module]:
        """Lists the networks by name.

        Raises:
          RuntimeError: If the networks have not been built.
        """

    @abstractmethod
    def _build_networks(self) -> None:
        """Creates freshly initialized networks for the recorded geometry."""

    def _activation(self) -> nn.Module:
        """Creates one instance of the hidden layer activation."""
        return ACTIVATIONS[self.activation]()

    def _train(
        self, nets: Sequence[nn.Module], loss_fn: LossFn, X_tr: torch.Tensor, X_val: torch.Tensor
    ) -> TrainingHistory:
        """Trains networks and restores the weights with the lowest validation loss.

        Args:
          nets: Networks whose parameters the loss depends on.
          loss_fn: Maps a batch of scaled snapshots to the scalar loss.
          X_tr: Training snapshots, one per entry along the first axis.
          X_val: Validation snapshots, in the same layout. May be empty.

        Returns:
          The training history.
        """
        params = [p for net in nets for p in net.parameters()]
        opt = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay, **adam_options(params))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.lr_factor,
            patience=self.lr_patience,
            threshold=self.threshold,
            min_lr=self.min_lr,
        )

        best = snapshot_buffers(nets)
        best_val, wait, have_best = float("inf"), 0, False
        history = TrainingHistory()

        for _ in range(self.epochs):
            history.lr.append(float(opt.param_groups[0]["lr"]))
            history.train.append(self._train_epoch(nets, loss_fn, opt, X_tr))
            if X_val.shape[0] == 0:
                continue

            for net in nets:
                net.eval()
            with torch.no_grad():
                v = loss_fn(X_val).item()
            history.val.append(v)
            sched.step(v)
            if v < best_val * (1.0 - self.threshold):
                best_val, wait, have_best = v, 0, True
                copy_state(nets, best)
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if have_best:
            for net, state in zip(nets, best, strict=True):
                net.load_state_dict(state)
        return history

    def _train_epoch(
        self,
        nets: Sequence[nn.Module],
        loss_fn: LossFn,
        opt: torch.optim.Optimizer,
        X_tr: torch.Tensor,
    ) -> float:
        """Runs one shuffled pass over the training snapshots.

        Args:
          nets: Networks whose parameters the loss depends on.
          loss_fn: Maps a batch of scaled snapshots to the scalar loss.
          opt: Optimizer over the networks' parameters.
          X_tr: Training snapshots, one per entry along the first axis.

        Returns:
          The mean loss over the pass.
        """
        for net in nets:
            net.train()
        # Drawn on the CPU, then moved with a blocking copy.
        order = torch.randperm(X_tr.shape[0]).to(X_tr.device)

        total = 0.0
        for start in range(0, X_tr.shape[0], self.batch_size):
            batch = X_tr[order[start : start + self.batch_size]]
            opt.zero_grad()
            loss = loss_fn(batch)
            loss.backward()
            opt.step()
            total += loss.item() * batch.shape[0]
        return total / X_tr.shape[0]

    def _weight_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects each network's state dictionary.

        Returns:
          Each tensor as an array, named `<network>.<key>`.
        """
        return {
            f"{name}.{key}": value.detach().cpu().numpy()
            for name, net in self._networks_by_name().items()
            for key, value in net.state_dict().items()
        }

    def _load_weight_arrays(self, arrays: Mapping[str, npt.NDArray[Any]]) -> None:
        """Builds the networks and loads their state dictionaries.

        Args:
          arrays: Arrays holding the entries `_weight_arrays` returned.

        Raises:
          RuntimeError: If a network's stored state is missing a key or holds an
            unexpected one.
        """
        self._build_networks()
        for name, net in self._networks_by_name().items():
            prefix = f"{name}."
            state = {
                k.removeprefix(prefix): torch.from_numpy(np.array(v)) for k, v in arrays.items() if k.startswith(prefix)
            }
            net.load_state_dict(state)


def adam_options(params: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Selects the fastest Adam implementation torch supports for the parameters.

    Args:
      params: The parameters the optimizer updates.

    Returns:
      `{"fused": True}` for floating-point CUDA parameters, `{"foreach": True}`
      for other CUDA parameters, and an empty dictionary otherwise.
    """
    if not params:
        return {}
    p0 = params[0]
    if not p0.is_cuda:
        return {}
    if p0.dtype in (torch.float16, torch.float32, torch.float64):
        return {"fused": True}
    return {"foreach": True}


def snapshot_buffers(nets: Sequence[nn.Module]) -> StateBuffers:
    """Allocates buffers shaped like each network's state dictionary.

    Args:
      nets: Networks whose weights the buffers hold.

    Returns:
      One uninitialized buffer dictionary per network.
    """
    return [{k: torch.empty_like(v) for k, v in net.state_dict().items()} for net in nets]


def copy_state(nets: Sequence[nn.Module], buffers: StateBuffers) -> None:
    """Copies each network's current weights into its buffers.

    Args:
      nets: Networks to copy from.
      buffers: Buffers from `snapshot_buffers`, in the same order as `nets`.
    """
    with torch.no_grad():
        for net, buf in zip(nets, buffers, strict=True):
            for k, v in net.state_dict().items():
                buf[k].copy_(v)
