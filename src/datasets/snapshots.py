"""
snapshots.py
============

Loading snapshot data as the array layout every Projector expects:

    X : (Nu, Nt, Nx, Ny)    NaN marks solid-body points

Readers own their file format's native axis order and return (Nt, Nx, Ny) per
field, so `load_snapshots` only has to stack them. Adding a format means adding
a reader and an entry in READERS -- nothing else changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "data_path",
    "SnapshotSpec",
    "load_snapshots",
    "SPECS",
]


# hpc/convergence.slr stages the h5 onto node-local disk and exports DATA_ROOT
def data_path(filename: str) -> str:
    return os.path.join(os.environ.get("DATA_ROOT", "data"), filename)


# ── decimation ─────────────────────────────────────────────────────────────────


# plain [::factor] folds everything above the new Nyquist back into the resolved
# wavenumbers as noise no model can generalise


def _boxcar(A: np.ndarray, axis: int, factor: int) -> np.ndarray:
    if factor == 1:
        return A
    n = (A.shape[axis] // factor) * factor
    A = np.take(A, np.arange(n), axis=axis)
    shape = list(A.shape)
    shape[axis : axis + 1] = [n // factor, factor]
    return A.reshape(shape).mean(axis=axis + 1)


def _decimate(A: np.ndarray, xs: int, ys: int) -> np.ndarray:
    """Antialiased decimation of a (Nt, Nx, Ny) block"""
    return _boxcar(_boxcar(A, axis=1, factor=xs), axis=2, factor=ys)


def _decimate_yx(A: np.ndarray, xs: int, ys: int) -> np.ndarray:
    """Same, for a block still in the h5 native (Nt, Ny, Nx) order.

    Filtering before the transpose rather than after keeps the float32
    accumulation order identical to the original loader; transposing first
    changes the summation strides and shifts the result by ~1 ULP.
    """
    return _boxcar(_boxcar(A, axis=1, factor=ys), axis=2, factor=xs)


def _time_indices(
    n_t_full: int, stride: int, max_snapshots: Optional[int]
) -> list[int]:
    idx = list(range(0, n_t_full, stride))
    return idx[:max_snapshots] if max_snapshots else idx


# ── readers ────────────────────────────────────────────────────────────────────


# the raw h5 planes are stored (Nt, Ny, Nx), so every block is swapped on the way
# out. antialias needs full x resolution to filter, hence blocks pulled and
# decimated in turn rather than one read of the whole plane
def read_h5(
    path: str,
    fields: tuple,
    stride: int,
    max_snapshots: Optional[int],
    xs: int,
    ys: int,
    antialias: bool,
    chunk: int = 200,
) -> list[np.ndarray]:
    import h5py

    planes = []
    with h5py.File(path, "r") as f:
        idx = _time_indices(f[fields[0]].shape[0], stride, max_snapshots)
        for name in fields:
            out = []
            for s in range(0, len(idx), chunk):
                blk = idx[s : s + chunk]
                sel = slice(blk[0], blk[-1] + 1, stride)
                if antialias:
                    A = _decimate_yx(f[name][sel].astype(np.float32), xs, ys)
                    A = np.swapaxes(A, 1, 2)
                else:
                    A = np.swapaxes(f[name][sel, ::ys, ::xs].astype(np.float32), 1, 2)
                out.append(A)
            planes.append(np.concatenate(out, axis=0))
    return planes


# the .mat wakes are already (Nt, Nx, Ny) and small enough to load whole
def read_mat(
    path: str,
    fields: tuple,
    stride: int,
    max_snapshots: Optional[int],
    xs: int,
    ys: int,
    antialias: bool,
) -> list[np.ndarray]:
    import scipy.io as sio

    planes = []
    for name in fields:
        raw = sio.loadmat(path, variable_names=[name])[name]
        idx = _time_indices(raw.shape[0], stride, max_snapshots)
        A = raw[idx].astype(np.float32)
        del raw
        planes.append(_decimate(A, xs, ys) if antialias else A[:, ::xs, ::ys])
    return planes


READERS = {".h5": read_h5, ".hdf5": read_h5, ".mat": read_mat}


# ── spec ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SnapshotSpec:
    """Everything needed to turn a file on disk into an (Nu, Nt, Nx, Ny) array.

    downsample is (x, y) for every format; the reader maps it onto that file's
    native axes so callers never have to know the storage order.
    """

    filename: str
    fields: tuple
    downsample: tuple = (1, 1)
    stride: int = 1
    max_snapshots: Optional[int] = None
    antialias: bool = True
    chunk: int = 200

    def replace(self, **kwargs) -> SnapshotSpec:
        """A copy with fields overridden, for one-off resolution changes."""
        from dataclasses import replace as _replace

        return _replace(self, **kwargs)


def load_snapshots(spec: SnapshotSpec) -> np.ndarray:
    """Load a snapshot set as (Nu, Nt, Nx, Ny)."""
    path = data_path(spec.filename)
    ext = os.path.splitext(path)[1].lower()
    if ext not in READERS:
        raise ValueError(f"no reader for '{ext}', have {sorted(READERS)}")

    xs, ys = spec.downsample
    kwargs = dict(
        stride=spec.stride,
        max_snapshots=spec.max_snapshots,
        xs=xs,
        ys=ys,
        antialias=spec.antialias,
    )
    if READERS[ext] is read_h5:
        kwargs["chunk"] = spec.chunk

    X = np.stack(READERS[ext](path, spec.fields, **kwargs), axis=0)

    # a wrong transpose here trains fine and is silently garbage, so it is worth
    # one assertion at the single point every dataset passes through
    if X.ndim != 4:
        raise ValueError(f"expected (Nu, Nt, Nx, Ny), got {X.shape}")
    if X.shape[0] != len(spec.fields):
        raise ValueError(
            f"stacked {X.shape[0]} fields but spec lists {len(spec.fields)}"
        )
    return X


# ── known datasets ─────────────────────────────────────────────────────────────
# resolution defaults only. model hyperparameters stay with the experiment that
# chose them, not here

SPECS = {
    "bl": SnapshotSpec(
        filename="Challenge1.1_train.h5",
        fields=("Uplane", "Vplane", "Wplane"),
        downsample=(32, 4),
        # the snapshots are already decorrelated at lag 1 (measured r = -0.006),
        # so stride > 1 buys no independence and throws away samples on a problem
        # that is badly data-limited
        stride=1,
        max_snapshots=3000,  # None takes all 5000
    ),
    "circle": SnapshotSpec(
        filename="wakes/circle_re_100.mat",
        fields=("ux", "uy"),
        downsample=(2, 2),
        max_snapshots=500,
    ),
    "triangle": SnapshotSpec(
        filename="wakes/triangle_re_100.mat",
        fields=("ux", "uy"),
        downsample=(2, 2),
        max_snapshots=500,
    ),
}
