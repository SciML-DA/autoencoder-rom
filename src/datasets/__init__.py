# pyright: strict
"""Loads snapshot data and splits it into train, validation, and test blocks.

`snapshots` reads a file into the `(Nu, Nt, Nx, Ny)` layout every `Projector`
expects. `splitting` divides that array into blocks a model can be scored on
honestly, and reports whether the division worked.

This package is pure numpy and imports without torch or jax, so a POD-only or
linear-estimator workflow can use it without loading either framework.

Typical usage example:

  from datasets import SPECS, load_snapshots, prepare_split

  X = load_snapshots(SPECS["bl"])
  X_train, X_val, X_test, meta = prepare_split(X)
"""

from .snapshots import SPECS, SnapshotSpec, data_path, load_snapshots
from .splitting import (
    LEAK_THRESHOLD,
    SHIFT_THRESHOLD,
    decorrelation_lag,
    linear_span_ceiling,
    prepare_split,
    split_diagnostics,
    split_indices,
)

__all__ = [
    # Loading
    "SnapshotSpec",
    "load_snapshots",
    "data_path",
    "SPECS",
    # Splitting
    "split_indices",
    "decorrelation_lag",
    "split_diagnostics",
    "linear_span_ceiling",
    "prepare_split",
    "LEAK_THRESHOLD",
    "SHIFT_THRESHOLD",
]
