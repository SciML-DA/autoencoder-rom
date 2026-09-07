# pyright: strict
"""
snapshots.py
============


Loads snapshot data in the array layout the `Projector` class expects:

    X : (Nu, Nt, Nx, Ny)    NaN points are solid-body points


Readers own their file format's native axis order and return (Nt, Nx, Ny) per
field, so `load_snapshots` only has to stack them. Adding a format means adding
a reader and an entry in READERS -- nothing else changes. That promise is now
enforced by the `Reader` protocol: a reader that does not match its signature
fails the type check rather than at the call site.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # h5py is imported lazily at runtime, see `read_h5`
    import h5py

__all__ = [
    "data_path",
    "SnapshotSpec",
    "load_snapshots",
    "SPECS",
]

#: what every reader produces and `load_snapshots` stacks
Planes = list[npt.NDArray[np.float32]]


# experiments/bl/hpc/convergence.slr stages the h5 onto node-local disk and
# exports DATA_ROOT
def data_path(filename: str) -> str:
    return os.path.join(os.environ.get("DATA_ROOT", "data"), filename)


# ── decimation ─────────────────────────────────────────────────────────────────


# plain [::factor] folds everything above the new Nyquist back into the resolved
# wavenumbers as noise no model can generalise


def _boxcar(A: npt.NDArray[Any], axis: int, factor: int) -> npt.NDArray[Any]:
    if factor < 1:
        raise ValueError(f"decimation factor must be >= 1, got {factor}")
    if factor == 1:
        return A
    if not -A.ndim <= axis < A.ndim:
        raise ValueError(f"axis {axis} out of range for a {A.ndim}-d array")

    n = (A.shape[axis] // factor) * factor
    if n == 0:
        raise ValueError(
            f"axis {axis} has length {A.shape[axis]}, shorter than the "
            f"decimation factor {factor}; every sample would be discarded"
        )

    keep: npt.NDArray[np.intp] = np.arange(n, dtype=np.intp)
    trimmed = np.take(A, keep, axis=axis)
    shape: list[int] = list(trimmed.shape)
    shape[axis : axis + 1] = [n // factor, factor]
    return trimmed.reshape(shape).mean(axis=axis + 1)


def _decimate(A: npt.NDArray[Any], xs: int, ys: int) -> npt.NDArray[Any]:
    """Antialiased decimation of a (Nt, Nx, Ny) block"""
    return _boxcar(_boxcar(A, axis=1, factor=xs), axis=2, factor=ys)


def _decimate_yx(A: npt.NDArray[Any], xs: int, ys: int) -> npt.NDArray[Any]:
    """Same, for a block still in the h5 native (Nt, Ny, Nx) order.

    Filtering before the transpose rather than after keeps the float32
    accumulation order identical to the original loader; transposing first
    changes the summation strides and shifts the result by ~1 ULP.
    """
    return _boxcar(_boxcar(A, axis=1, factor=ys), axis=2, factor=xs)


def _time_indices(n_t_full: int, stride: int, max_snapshots: int | None) -> list[int]:
    """Which time indices to read.

    ``max_snapshots is not None``, not truthiness: 0 means "no snapshots", and
    under the old falsy test it meant "all of them" -- ``_time_indices(100, 1, 0)``
    returned all 100. `None` is the way to ask for everything, which is what the
    annotation has always said.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    idx = list(range(0, n_t_full, stride))
    if max_snapshots is None:
        return idx
    if max_snapshots < 0:
        raise ValueError(f"max_snapshots must be >= 0 or None, got {max_snapshots}")
    return idx[:max_snapshots]


# ── readers ────────────────────────────────────────────────────────────────────


class Reader(Protocol):
    """The contract `load_snapshots` calls every format through.

    Keyword-only after ``fields`` so the dispatch in `load_snapshots` can pass
    one uniform set of arguments. ``chunk`` is meaningful only to the streaming
    h5 reader, and is accepted and ignored by the others -- which is what lets
    the dispatch stay a plain call instead of an identity test on the function
    object followed by a hand-built kwargs dict.
    """

    def __call__(
        self,
        path: str,
        fields: tuple[str, ...],
        *,
        stride: int,
        max_snapshots: int | None,
        xs: int,
        ys: int,
        antialias: bool,
        chunk: int,
    ) -> Planes: ...


def _dataset(f: h5py.File, name: str, path: str) -> h5py.Dataset:
    """The named h5 object, checked to be a Dataset.

    ``h5py.File.__getitem__`` returns ``Dataset | Group | Datatype``, and only
    ``Dataset`` has ``.shape``, ``.astype`` and slicing. Narrowing here is what
    lets the read loop be checked at all, and -- more to the point -- a file
    with a Group at that name now fails with a sentence naming the file and the
    offending field, instead of an ``AttributeError`` from inside the loop.
    """
    import h5py

    obj = f[name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"{name!r} in {path} is a {type(obj).__name__}, not a dataset; the spec's `fields` must name array datasets"
        )
    return obj


# the raw h5 planes are stored (Nt, Ny, Nx), so every block is swapped on the way
# out. antialias needs full x resolution to filter, hence blocks pulled and
# decimated in turn rather than one read of the whole plane
def read_h5(
    path: str,
    fields: tuple[str, ...],
    *,
    stride: int,
    max_snapshots: int | None,
    xs: int,
    ys: int,
    antialias: bool,
    chunk: int = 200,
) -> Planes:
    import h5py

    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")

    planes: Planes = []
    with h5py.File(path, "r") as f:
        idx = _time_indices(_dataset(f, fields[0], path).shape[0], stride, max_snapshots)
        for name in fields:
            dset = _dataset(f, name, path)
            out: Planes = []
            for s in range(0, len(idx), chunk):
                blk = idx[s : s + chunk]
                sel = slice(blk[0], blk[-1] + 1, stride)
                if antialias:
                    filtered = _decimate_yx(dset[sel].astype(np.float32), xs, ys)
                else:
                    filtered = dset[sel, ::ys, ::xs].astype(np.float32)
                out.append(np.swapaxes(filtered, 1, 2))
            if not out:
                raise ValueError(f"{name!r} in {path} yielded no snapshots")
            planes.append(np.concatenate(out, axis=0))
    return planes


# the .mat wakes are already (Nt, Nx, Ny) and small enough to load whole
def read_mat(
    path: str,
    fields: tuple[str, ...],
    *,
    stride: int,
    max_snapshots: int | None,
    xs: int,
    ys: int,
    antialias: bool,
    chunk: int = 200,  # unused: the whole file is read at once. See `Reader`.
) -> Planes:
    import scipy.io as sio

    del chunk

    planes: Planes = []
    for name in fields:
        # spmatrix=False is scipy 2.1's coming default; passing it explicitly
        # silences the deprecation rather than inheriting a behaviour change
        # later. It governs only how *sparse* variables come back, and these
        # files hold dense wake fields, so the choice costs nothing here.
        contents: dict[str, Any] = sio.loadmat(path, variable_names=[name], spmatrix=False)
        if name not in contents:
            raise KeyError(
                f"{name!r} not found in {path}; it holds {sorted(k for k in contents if not k.startswith('__'))}"
            )
        raw: npt.NDArray[Any] = contents[name]
        idx = _time_indices(raw.shape[0], stride, max_snapshots)
        A: npt.NDArray[np.float32] = raw[idx].astype(np.float32)
        del raw
        planes.append(_decimate(A, xs, ys) if antialias else A[:, ::xs, ::ys])
    return planes


READERS: dict[str, Reader] = {".h5": read_h5, ".hdf5": read_h5, ".mat": read_mat}


# ── spec ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SnapshotSpec:
    """Everything needed to turn a file on disk into an (Nu, Nt, Nx, Ny) array.

    downsample is (x, y) for every format; the reader maps it onto that file's
    native axes so callers never have to know the storage order.
    """

    filename: str
    fields: tuple[str, ...]
    downsample: tuple[int, int] = (1, 1)
    stride: int = 1
    max_snapshots: int | None = None
    antialias: bool = True
    chunk: int = 200

    def replace(self, **kwargs: Any) -> SnapshotSpec:
        """A copy with fields overridden, for one-off resolution changes."""
        from dataclasses import replace as _replace

        return _replace(self, **kwargs)


def load_snapshots(spec: SnapshotSpec) -> npt.NDArray[np.float32]:
    """Load a snapshot set as (Nu, Nt, Nx, Ny)."""
    if not spec.fields:
        raise ValueError(f"spec for {spec.filename!r} lists no fields")

    path = data_path(spec.filename)
    ext = os.path.splitext(path)[1].lower()
    reader = READERS.get(ext)
    if reader is None:
        raise ValueError(f"no reader for '{ext}', have {sorted(READERS)}")

    xs, ys = spec.downsample
    X = np.stack(
        reader(
            path,
            spec.fields,
            stride=spec.stride,
            max_snapshots=spec.max_snapshots,
            xs=xs,
            ys=ys,
            antialias=spec.antialias,
            chunk=spec.chunk,
        ),
        axis=0,
    )

    # a wrong transpose here trains fine and is silently garbage, so it is worth
    # one assertion at the single point every dataset passes through
    if X.ndim != 4:
        raise ValueError(f"expected (Nu, Nt, Nx, Ny), got {X.shape}")
    if X.shape[0] != len(spec.fields):
        raise ValueError(f"stacked {X.shape[0]} fields but spec lists {len(spec.fields)}")
    return X


# ── known datasets ─────────────────────────────────────────────────────────────
# resolution defaults only. model hyperparameters stay with the experiment that
# chose them, not here


def _env_override(key: str, spec: SnapshotSpec) -> SnapshotSpec:
    """Point a dataset at a pre-decimated copy, from the environment.

    `experiments/bl/scripts/make_subset.py` writes a smaller file than the spec
    assumes so that only a fraction has to cross the network. Redirecting a spec
    at one means moving `filename` and `downsample` *together*: the subset is
    already decimated, so reusing the original factor silently changes
    resolution rather than failing. Both are therefore required as a pair.

        BL_FILE=Challenge1.1_train_sub.h5 BL_DOWNSAMPLE=4,4

    Read at import, so the variables have to be exported before the process
    starts -- which is what the job scripts do. Raising here means a malformed
    value fails `import datasets` rather than loading at the wrong resolution.
    """
    filename = os.environ.get(f"{key.upper()}_FILE")
    downsample = os.environ.get(f"{key.upper()}_DOWNSAMPLE")
    if filename is None and downsample is None:
        return spec
    if filename is None or downsample is None:
        raise ValueError(
            f"{key.upper()}_FILE and {key.upper()}_DOWNSAMPLE must be set together; "
            f"got file={filename!r} downsample={downsample!r}. Setting one alone "
            "changes resolution silently."
        )

    parts = downsample.split(",")
    if len(parts) != 2:
        raise ValueError(f"{key.upper()}_DOWNSAMPLE must be 'xs,ys', got {downsample!r} ({len(parts)} values)")
    try:
        xs, ys = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"{key.upper()}_DOWNSAMPLE must be 'xs,ys' integers, got {downsample!r}") from None
    if xs < 1 or ys < 1:
        raise ValueError(f"{key.upper()}_DOWNSAMPLE factors must be >= 1, got ({xs}, {ys})")
    return spec.replace(filename=filename, downsample=(xs, ys))


_SPECS: dict[str, SnapshotSpec] = {
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

SPECS: dict[str, SnapshotSpec] = {k: _env_override(k, v) for k, v in _SPECS.items()}
