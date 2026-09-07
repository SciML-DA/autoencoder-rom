# pyright: strict
"""Snapshot loading and train/val/test splitting.

Pure numpy: this package imports without torch or jax, which is what lets
a POD-only or linear-estimator workflow use it without paying for either.
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
    # loading
    "SnapshotSpec",
    "load_snapshots",
    "data_path",
    "SPECS",
    # splitting
    "split_indices",
    "decorrelation_lag",
    "split_diagnostics",
    "linear_span_ceiling",
    "prepare_split",
    "LEAK_THRESHOLD",
    "SHIFT_THRESHOLD",
]
