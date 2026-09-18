"""Training history shared by the autoencoders and the LSTM.

Typical usage example:

  history = TrainingHistory()
  for epoch in range(epochs):
      history.train.append(train_loss)
      history.val.append(val_loss)
      history.lr.append(lr)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["TrainingHistory"]


@dataclass
class TrainingHistory:
    """Losses and learning rates recorded while training.

    Attributes:
      train: Training loss per epoch.
      val: Validation loss per epoch. Empty without a validation block.
      lr: Learning rate used in each epoch.
    """

    train: list[float] = field(default_factory=list[float])
    val: list[float] = field(default_factory=list[float])
    lr: list[float] = field(default_factory=list[float])

    @property
    def n_epochs_run(self) -> int:
        """Number of epochs run."""
        return len(self.train)

    def to_arrays(self) -> dict[str, npt.NDArray[np.float64]]:
        """Converts the history to arrays.

        Returns:
          The losses and learning rates under `history_train`, `history_val`,
          and `history_lr`.
        """
        return {
            "history_train": np.asarray(self.train, dtype=np.float64),
            "history_val": np.asarray(self.val, dtype=np.float64),
            "history_lr": np.asarray(self.lr, dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, npt.NDArray[Any]]) -> TrainingHistory:
        """Reads a history written by `to_arrays`.

        Args:
          arrays: Arrays holding `history_train`, `history_val`, and
            `history_lr`.

        Returns:
          The history.
        """
        return cls(
            train=[float(v) for v in arrays["history_train"]],
            val=[float(v) for v in arrays["history_val"]],
            lr=[float(v) for v in arrays["history_lr"]],
        )
