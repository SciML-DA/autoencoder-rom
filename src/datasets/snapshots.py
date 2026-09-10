# pyright: strict
"""Loads snapshot data in the array layout that `Projector` expects.

Every loader in this module returns the same layout, `(Nu, Nt, Nx, Ny)`, where
`Nu` counts the velocity components, `Nt` the snapshots, and `Nx` and `Ny` the
grid. NaN marks a solid-body point.

Each reader owns its file format's native axis order and returns `(Nt, Nx, Ny)`
per field, so `load_snapshots` only stacks them. To support another format,
write a reader and add an entry to `READERS`. The `Reader` protocol fixes the
signature, so a reader that doesn't match fails the type check rather than the
call.

Typical usage example:

  from datasets import SPECS, load_snapshots

  X = load_snapshots(SPECS["bl"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # `read_h5` imports h5py at call time; this import serves annotations only.
    import h5py

__all__ = [
    "data_path",
    "SnapshotSpec",
    "load_snapshots",
    "SPECS",
]

#: The per-field planes a reader returns and `load_snapshots` stacks.
Planes = list[npt.NDArray[np.float32]]


def data_path(filename: str) -> str:
    """Resolves a data filename against the configured data root.

    `experiments/bl/hpc/convergence.slr` stages the HDF5 file onto node-local
    disk and exports `DATA_ROOT` to point at it.

    Args:
      filename: Path relative to the data root, such as `wakes/circle_re_100.mat`.

    Returns:
      `filename` joined to `$DATA_ROOT`, or to `data/` when that variable is
      unset.
    """
    return os.path.join(os.environ.get("DATA_ROOT", "data"), filename)


# ── Decimation ─────────────────────────────────────────────────────────────────


def _boxcar(A: npt.NDArray[Any], axis: int, factor: int) -> npt.NDArray[Any]:
    """Averages neighboring samples along one axis.

    Args:
      A: Array to decimate. Any shape and dtype.
      axis: Axis to average along. Negative values index from the end.
      factor: Number of samples to average into one. A factor of 1 returns `A`
        unchanged.

    Returns:
      `A` with `axis` shortened by `factor`. Samples beyond the last whole group
      are discarded.

    Raises:
      ValueError: If `factor` is below 1, if `axis` is out of range for `A`, or
        if `axis` is shorter than `factor`.
    """
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
    """Decimates a block that already uses the target axis order.

    Args:
      A: Block shaped `(Nt, Nx, Ny)`.
      xs: Decimation factor along x.
      ys: Decimation factor along y.

    Returns:
      The block with both spatial axes decimated.

    Raises:
      ValueError: If either factor is below 1 or exceeds its axis length.
    """
    return _boxcar(_boxcar(A, axis=1, factor=xs), axis=2, factor=ys)


def _decimate_yx(A: npt.NDArray[Any], xs: int, ys: int) -> npt.NDArray[Any]:
    """Decimates a block still in the HDF5 native axis order.

    Filters before the caller transposes. Transposing first changes the
    summation strides, which shifts the float32 result by about one unit in the
    last place.

    Args:
      A: Block shaped `(Nt, Ny, Nx)`.
      xs: Decimation factor along x, which is the last axis here.
      ys: Decimation factor along y, which is the middle axis here.

    Returns:
      The block with both spatial axes decimated, still in `(Nt, Ny, Nx)` order.

    Raises:
      ValueError: If either factor is below 1 or exceeds its axis length.
    """
    return _boxcar(_boxcar(A, axis=1, factor=ys), axis=2, factor=xs)


def _time_indices(n_t_full: int, stride: int, max_snapshots: int | None) -> list[int]:
    """Chooses which time indices to read from a file.

    Compares `max_snapshots` against `None`.

    Args:
      n_t_full: Number of snapshots the file holds.
      stride: Take every `stride`-th snapshot.
      max_snapshots: Maximum number of indices to return. 0 selects none, and
        `None` selects every index the stride reaches.

    Returns:
      Ascending time indices, at most `max_snapshots` of them.

    Raises:
      ValueError: If `stride` is below 1, or `max_snapshots` is negative.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    idx = list(range(0, n_t_full, stride))

    if max_snapshots is None:
        return idx
    if max_snapshots < 0:
        raise ValueError(f"max_snapshots must be >= 0 or None, got {max_snapshots}")
    return idx[:max_snapshots]


# ── Readers ────────────────────────────────────────────────────────────────────


class Reader(Protocol):
    """The interface `load_snapshots` calls every file format through.

    Every argument after `fields` is keyword-only, so `load_snapshots` passes one
    uniform set of arguments to whichever reader it selects. `chunk` applies only
    to the streaming HDF5 reader; the others accept it and ignore it.
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
    ) -> Planes:
        """Reads one file into per-field planes.

        Args:
          path: File to read.
          fields: Variable names to read, one per velocity component.
          stride: Take every `stride`-th snapshot.
          max_snapshots: Maximum snapshots to read, or `None` for all of them.
          xs: Decimation factor along x.
          ys: Decimation factor along y.
          antialias: Average before decimating rather than striding.
          chunk: Snapshots to hold in memory at once, where the reader streams.

        Returns:
          One `(Nt, Nx, Ny)` float32 array per entry in `fields`, in that order.
        """
        ...


def _dataset(f: h5py.File, name: str, path: str) -> h5py.Dataset:
    """Looks up a named HDF5 object and checks that it is a dataset.

    `h5py.File.__getitem__` returns a `Dataset`, a `Group`, or a `Datatype`, and
    only `Dataset` supports `.shape`, `.astype`, and slicing. Narrowing the type
    here lets the type checker verify the read loop. It also turns a group at
    that name into an error that names the file and the field, rather than an
    `AttributeError` raised from inside the loop.

    Args:
      f: Open HDF5 file.
      name: Object name to look up.
      path: File path, used only to name the file in the error message.

    Returns:
      The named dataset.

    Raises:
      TypeError: If `name` refers to a group or a datatype.
      KeyError: If `name` is absent from the file.
    """
    import h5py

    obj = f[name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"{name!r} in {path} is a {type(obj).__name__}, not a dataset; the spec's `fields` must name array datasets"
        )
    return obj


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
    """Reads HDF5 planes, one array per field.

    The file stores planes as `(Nt, Ny, Nx)`, so this reader swaps the last two
    axes on the way out. Antialiasing needs full x resolution to filter against,
    so the reader pulls and decimates `chunk` snapshots at a time rather than
    reading a whole plane into memory.

    Args:
      path: HDF5 file to read.
      fields: Dataset names to read, one per velocity component.
      stride: Take every `stride`-th snapshot.
      max_snapshots: Maximum snapshots to read, or `None` for all of them.
      xs: Decimation factor along x.
      ys: Decimation factor along y.
      antialias: Average before decimating rather than striding.
      chunk: Snapshots to read and decimate per iteration.

    Returns:
      One `(Nt, Nx, Ny)` float32 array per entry in `fields`, in that order.

    Raises:
      ValueError: If `chunk` is below 1, or if a field yields no snapshots.
      TypeError: If a name in `fields` refers to a group rather than a dataset.
      OSError: If `path` is missing or unreadable.
    """
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


def read_mat(
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
    """Reads MATLAB wake files, one array per field.

    These files already use the target axis order and are small enough to read
    whole, so this reader ignores `chunk`. `Reader` explains why it accepts it.

    Args:
      path: MATLAB file to read.
      fields: Variable names to read, one per velocity component.
      stride: Take every `stride`-th snapshot.
      max_snapshots: Maximum snapshots to read, or `None` for all of them.
      xs: Decimation factor along x.
      ys: Decimation factor along y.
      antialias: Average before decimating rather than striding.
      chunk: Ignored.

    Returns:
      One `(Nt, Nx, Ny)` float32 array per entry in `fields`, in that order.

    Raises:
      KeyError: If a name in `fields` is absent from the file. The message lists
        the variables the file does hold.
      FileNotFoundError: If `path` is missing.
    """
    import scipy.io as sio

    del chunk

    planes: Planes = []
    for name in fields:
        # `spmatrix=False` selects the behavior that SciPy 2.1 makes the default
        # and silences the deprecation warning.
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


#: Maps a lowercase file extension to the reader that handles it.
READERS: dict[str, Reader] = {".h5": read_h5, ".hdf5": read_h5, ".mat": read_mat}


# ── Specs ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SnapshotSpec:
    """Everything needed to turn a file on disk into an `(Nu, Nt, Nx, Ny)` array.

    Attributes:
      filename: Path relative to the data root. `data_path` resolves it.
      fields: Variable names to read, one per velocity component. Their order
        fixes the order of the leading axis.
      downsample: Decimation factors as `(x, y)` for every format. The reader
        maps them onto that file's native axes, so you never need to know the
        storage order.
      stride: Take every `stride`-th snapshot.
      max_snapshots: Maximum snapshots to read, or `None` for all of them.
      antialias: Average before decimating rather than striding.
      chunk: Snapshots a streaming reader holds in memory at once.
    """

    filename: str
    fields: tuple[str, ...]
    downsample: tuple[int, int] = (1, 1)
    stride: int = 1
    max_snapshots: int | None = None
    antialias: bool = True
    chunk: int = 200

    def replace(self, **kwargs: Any) -> SnapshotSpec:
        """Copies this spec with some fields overridden.

        Args:
          **kwargs: Attribute names and replacement values.

        Returns:
          A new spec. This one is unchanged, because the class is frozen.

        Raises:
          TypeError: If a keyword names no attribute of this class.
        """
        from dataclasses import replace as _replace

        return _replace(self, **kwargs)


def load_snapshots(spec: SnapshotSpec) -> npt.NDArray[np.float32]:
    """Loads a snapshot set described by a spec.

    Args:
      spec: The file to read and the resolution to read it at.

    Returns:
      A float32 array shaped `(Nu, Nt, Nx, Ny)`, where `Nu` matches the length
      of `spec.fields`. NaN marks a solid-body point.

    Raises:
      ValueError: If `spec` lists no fields, if no reader handles the file
        extension, or if the stacked result has the wrong shape.
      KeyError: If a name in `spec.fields` is absent from the file.
      OSError: If the file is missing or unreadable.
    """
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

    # An incorrect transpose could be quite bad, so it's worth asserting shape
    if X.ndim != 4:
        raise ValueError(f"expected (Nu, Nt, Nx, Ny), got {X.shape}")
    if X.shape[0] != len(spec.fields):
        raise ValueError(f"stacked {X.shape[0]} fields but spec lists {len(spec.fields)}")
    return X


# ── Known datasets ─────────────────────────────────────────────────────────────


def _env_override(key: str, spec: SnapshotSpec) -> SnapshotSpec:
    """Points a dataset at a pre-decimated copy, using the environment.

    `experiments/bl/scripts/make_subset.py` writes a smaller file than the spec
    assumes, so that only a fraction of the data crosses the network. To redirect
    a spec at one, set both variables together:

      BL_FILE=Challenge1.1_train_sub.h5 BL_DOWNSAMPLE=4,4

    Both are required as a pair because the subset is already decimated. Setting
    only the filename reuses the original factor and changes the resolution.

    This function runs at import. Eexport the variables before starting the
    process (which is what the job scripts do). A malformed value fails
    `import datasets`.

    Args:
      key: Dataset name, uppercased to form the variable prefix. `bl` reads
        `BL_FILE` and `BL_DOWNSAMPLE`.
      spec: The spec to redirect.

    Returns:
      A spec pointed at the override, or `spec` unchanged when neither variable
      is set.

    Raises:
      ValueError: If only one of the pair is set, if the downsample value is not
        two comma-separated integers, or if either factor is below 1.
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


# Resolution defaults
_SPECS: dict[str, SnapshotSpec] = {
    "bl": SnapshotSpec(
        filename="Challenge1.1_train.h5",
        fields=("Uplane", "Vplane", "Wplane"),
        downsample=(32, 4),
        stride=1,
        max_snapshots=3000,
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

#: The known datasets, with any environment overrides applied.
SPECS: dict[str, SnapshotSpec] = {k: _env_override(k, v) for k, v in _SPECS.items()}
