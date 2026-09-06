"""Reader for the April porous-disc wind-tunnel experiment.

Loads PIV snapshots and the synchronised force-balance record from the RDS
project space, resolved through `$RDS_ROOT`:

    /rds/general/project/immanuel/live/Seagate/april_experiment

Layout, per the folder's own readme:

    4p5d_10ms_yaw_<a>_<b>_<c>/
        piv_raw_images/
        piv_snapshots/                       (55, 90) per snapshot
        piv_snapshots_highres/               (99, 159) per snapshot
    synced_forces/<date>/
        10ms_<y1>_<y2>_sync(HH-MM-SS).dat    paired with a PIV buffer
        10ms_<y1>_<y2>(HH-MM-SS).dat         not paired
        baseline[_<y1>_<y2>](HH-MM-SS).dat   tunnel off, drift reference
    meanfield_4p5d_10ms_yaw_0_0_0.npz        X, Y, u_bar, v_bar, Rxx/Rxy/Ryy, mask

Use `build_case` as the entry point. It resolves the three things that are easy
to get wrong and are decided in one place here: the PIV/force pairing, the
baseline drift correction, and the mask convention.

Axis order: stored arrays are (Ny, Nx), with X spanning the 159 axis and Y the
99 axis. Every `Projector` in this repository expects (Nu, Nt, Nx, Ny), so
`load_run` transposes and `_check_grid` asserts the orientation rather than
assuming it.

Coordinates: snapshot `.npz` files hold only `u` and `v`. X, Y, and the body
mask come from the mean-field file. `grid_from_scaling` derives the grid from
the 5.7384 px/mm calibration instead, and `check_scaling` cross-checks the two.

Typical usage:

    from experiments.april_wake.case_reader import build_case

    case = build_case("4p5d_10ms_yaw_0_0_0", n_snapshots=2000)
    Q, S = case.flat(), case.S     # (N_x, N_t) and (N_s, N_t)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Optional

import numpy as np

__all__ = [
    "ROOT",
    "PX_PER_MM",
    "F_PIV_HZ",
    "RUNS",
    "MeanField",
    "load_meanfield",
    "load_run",
    "list_snapshots",
    "pair_indices",
    "force_index_for_pair",
    "sync_forces",
    "smooth_forces",
    "notch_forces",
    "probe_signals",
    "auto_probe_points",
    "F_FORCE_HZ",
    "N_FORCE_CHANNELS",
    "V_SIGN",
    "grid_from_scaling",
    "check_scaling",
    "ForceRecord",
    "inspect_dat",
    "read_dat",
    "find_force_files",
    "baseline_drift_fit",
    "apply_drift_correction",
    "Case",
    "build_case",
    "concat_cases",
    "yaw_from_run",
    "MEANFIELD",
]

# `RDS_ROOT` lets you point at a local copy without editing anything
ROOT = os.environ.get(
    "RDS_ROOT", "/rds/general/project/immanuel/live/Seagate/april_experiment"
)

PX_PER_MM = 5.7384  # April experiment calibration, from readme.md
F_PIV_HZ = 250.0  # PIV acquisition frequency
DT_INTERFRAME = 100e-6  # s, inter-frame separation within a pair

# Force balance, from synced_forces/*/readme.txt ("2500 Hz, 30 sec") and
# confirmed by file size: 7 440 000 bytes / 8 / 12 = 77 500 samples = 31.0 s.
F_FORCE_HZ = 2500.0
N_FORCE_CHANNELS = 12  # two 6-component balances, discs 2 and 3
FORCE_PER_PIV = int(round(F_FORCE_HZ / F_PIV_HZ))  # 10

# The stored v component uses the opposite sign to the convention here: the
# camera views the discs from the far wall, mirroring the in-plane cross-stream
# axis. The reference notebook applies the same flip. Without it the
# cross-stream velocity and every Reynolds shear stress derived from it carry
# the wrong sign.
V_SIGN = -1.0

RUNS = (
    "4p5d_10ms_yaw_0_0_0",
    "4p5d_10ms_yaw_20_10_0",
    "4p5d_10ms_yaw_-20_-10_0",
    "4p5d_10ms_yaw_30_15_0",
    "4p5d_10ms_yaw_-30_-15_0",
)

MEANFIELD = "meanfield_4p5d_10ms_yaw_0_0_0.npz"


# ── PIV ───────────────────────────────────────────────────────────────────────


@dataclass
class MeanField:
    """Mean field and grid. All 2-D arrays are (Nx, Ny) after transposition."""

    x: np.ndarray  # (Nx,) streamwise coordinate, mm
    y: np.ndarray  # (Ny,) cross-stream coordinate, mm
    X: np.ndarray  # (Nx, Ny) mesh
    Y: np.ndarray  # (Nx, Ny) mesh
    u_bar: np.ndarray  # (Nx, Ny) mean streamwise velocity
    v_bar: np.ndarray  # (Nx, Ny) mean cross-stream velocity
    Rxx: np.ndarray  # (Nx, Ny) Reynolds stresses
    Rxy: np.ndarray
    Ryy: np.ndarray
    fluid_mask: np.ndarray  # (Nx, Ny) True where the flow is valid

    @property
    def shape(self) -> tuple:
        return self.u_bar.shape

    @property
    def dx(self) -> float:
        return float(np.diff(self.x).mean())

    @property
    def dy(self) -> float:
        return float(np.diff(self.y).mean())

    def tke(self) -> np.ndarray:
        """Turbulent kinetic energy from the in-plane stresses (2-component)."""
        return 0.5 * (self.Rxx + self.Ryy)


def _check_grid(Xs: np.ndarray, Ys: np.ndarray) -> None:
    """Checks that the stored meshes are (Ny, Nx) before transposition.

    X must vary along the last axis and stay constant down the first; Y the
    reverse. A flipped export otherwise yields a POD of a transposed field.

    Raises:
        ValueError: If either mesh is not two-dimensional, or if the
            orientation does not match.
    """
    if Xs.ndim != 2 or Ys.ndim != 2:
        raise ValueError(f"expected 2-D meshes, got {Xs.shape} and {Ys.shape}")
    x_along_cols = np.ptp(Xs, axis=1).mean()
    x_along_rows = np.ptp(Xs, axis=0).mean()
    y_along_cols = np.ptp(Ys, axis=1).mean()
    y_along_rows = np.ptp(Ys, axis=0).mean()
    if not (x_along_cols > x_along_rows and y_along_rows > y_along_cols):
        raise ValueError(
            "grid orientation is not the expected (Ny, Nx): X should vary "
            f"across columns (got spread {x_along_cols:.3g} vs {x_along_rows:.3g} "
            f"down rows) and Y down rows (got {y_along_rows:.3g} vs "
            f"{y_along_cols:.3g}). Check the export before transposing."
        )


def load_meanfield(path: Optional[str] = None) -> MeanField:
    """Load the mean-field file, transposed to (Nx, Ny)."""
    path = path or os.path.join(ROOT, MEANFIELD)
    with np.load(path) as z:
        missing = {"X", "Y", "u_bar", "v_bar", "mask"} - set(z.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}; has {z.files}")
        Xs, Ys = z["X"], z["Y"]
        _check_grid(Xs, Ys)
        X, Y = Xs.T, Ys.T  # -> (Nx, Ny)
        mask = z["mask"].T
        out = {
            k: z[k].T for k in ("u_bar", "v_bar", "Rxx", "Rxy", "Ryy") if k in z.files
        }

    # mask polarity is not documented; infer it from which side carries the flow
    fluid = _resolve_mask_polarity(mask, out.get("u_bar"))

    zeros = np.zeros_like(X)
    return MeanField(
        x=X[:, 0],
        y=Y[0, :],
        X=X,
        Y=Y,
        u_bar=out.get("u_bar", zeros),
        v_bar=out.get("v_bar", zeros),
        Rxx=out.get("Rxx", zeros),
        Rxy=out.get("Rxy", zeros),
        Ryy=out.get("Ryy", zeros),
        fluid_mask=fluid,
    )


def _resolve_mask_polarity(mask: np.ndarray, u_bar: Optional[np.ndarray]):
    """Returns a mask that is True on fluid points.

    The stored mask carries no documented polarity, so this resolves it from the
    data: the fluid side is whichever carries non-zero mean velocity. Falls back
    to treating the majority side as fluid.
    """
    if u_bar is not None and np.any(u_bar != 0):
        inside = np.abs(u_bar[mask]).mean() if mask.any() else 0.0
        outside = np.abs(u_bar[~mask]).mean() if (~mask).any() else 0.0
        return mask if inside >= outside else ~mask
    return mask if mask.sum() >= mask.size / 2 else ~mask


def list_snapshots(run: str, highres: bool = True, root: Optional[str] = None):
    """Sorted snapshot paths for a run.

    Sorted by the pair index parsed from `PIV_PAIR_000201-000202.npz` rather
    than lexicographically, so a change in zero-padding cannot reorder time.

    Args:
        run: Run directory name.
        highres: Whether to read `piv_snapshots_highres` instead of
            `piv_snapshots`.
        root: Data root. Defaults to `ROOT`.

    Returns:
        Sorted snapshot paths.

    Raises:
        FileNotFoundError: If the snapshot directory does not exist.
    """
    root = root or ROOT
    sub = "piv_snapshots_highres" if highres else "piv_snapshots"
    d = os.path.join(root, run, sub)
    if not os.path.isdir(d):
        raise FileNotFoundError(d)

    files = [f for f in os.listdir(d) if f.endswith(".npz")]

    def key(f):
        m = re.search(r"(\d+)", f)
        return int(m.group(1)) if m else -1

    return [os.path.join(d, f) for f in sorted(files, key=key)]


def load_run(
    run: str,
    highres: bool = True,
    stride: int = 1,
    max_snapshots: Optional[int] = None,
    root: Optional[str] = None,
    dtype=np.float32,
) -> np.ndarray:
    """Load one run as (Nu, Nt, Nx, Ny), NaN preserved at invalid vectors.

    NaN marks solid-body and invalid points, matching both the snapshot files
    and the repository convention, so this function only transposes.

    6085 snapshots at (99, 159) occupy about 1.2 GB in float64 and 600 MB in
    float32. Pass `max_snapshots` while prototyping.

    Args:
        run: Run directory name.
        highres: Whether to read the high-resolution snapshots.
        stride: Snapshot decimation factor.
        max_snapshots: Maximum snapshots to read.
        root: Data root. Defaults to `ROOT`.
        dtype: Output dtype.

    Returns:
        Velocity fields, shape (Nu, Nt, Nx, Ny).

    Raises:
        FileNotFoundError: If the run has no snapshots.
    """
    paths = list_snapshots(run, highres, root)[::stride]
    if max_snapshots:
        paths = paths[:max_snapshots]
    if not paths:
        raise FileNotFoundError(f"no snapshots for run {run!r}")

    us = np.empty((len(paths),) + _peek_shape(paths[0])[::-1], dtype=dtype)
    vs = np.empty_like(us)
    for i, p in enumerate(paths):
        with np.load(p) as z:
            us[i] = z["u"].T  # (Ny, Nx) -> (Nx, Ny)
            vs[i] = V_SIGN * z["v"].T  # see V_SIGN
    return np.stack([us, vs])  # (2, Nt, Nx, Ny)


def pair_indices(
    run: str,
    highres: bool = True,
    stride: int = 1,
    max_snapshots: Optional[int] = None,
    root: Optional[str] = None,
):
    """The PIV pair index of each snapshot, in the same order as `load_run`.

    Pass the result to `sync_forces`. Call this with the same parameters used
    for `load_run`, or the two orderings diverge.

    Args:
        run: Run directory name.
        highres: Whether to read the high-resolution snapshots.
        stride: Snapshot decimation factor.
        max_snapshots: Maximum snapshots to read.
        root: Data root. Defaults to `ROOT`.

    Returns:
        The PIV pair index of each snapshot, in `load_run` order.

    Raises:
        ValueError: If a filename carries no parsable pair index.
    """
    paths = list_snapshots(run, highres, root)[::stride]
    if max_snapshots:
        paths = paths[:max_snapshots]
    out = []
    for p in paths:
        m = re.search(r"(\d+)", os.path.basename(p))
        if m is None:
            raise ValueError(
                f"cannot read a pair index from {os.path.basename(p)}; "
                "force synchronisation depends on it"
            )
        out.append(int(m.group(1)))
    return out


def _peek_shape(path: str) -> tuple:
    with np.load(path) as z:
        return z["u"].shape  # (Ny, Nx)


# ── grid cross-check ──────────────────────────────────────────────────────────


def grid_from_scaling(n_x: int, n_y: int, window_step_px: float):
    """Grid spacing in mm from the calibration, the readme's alternative route.

    Args:
        n_x: Number of grid points along X.
        n_y: Number of grid points along Y.
        window_step_px: PIV interrogation window step in pixels, not the window
            size. With 50% overlap on 32 px windows this is 16.

    Returns:
        A tuple `(x, y)` of coordinate arrays in mm.
    """
    d = window_step_px / PX_PER_MM
    return np.arange(n_x) * d, np.arange(n_y) * d


def check_scaling(mf: MeanField, tol: float = 0.02) -> dict:
    """Cross-check the mean-field grid against the px/mm calibration.

    Compares the mean-field grid spacing against the px/mm calibration. A
    disagreement means the vector spacing differs from the assumed value, which
    scales every length and every Strouhal number derived from it.

    Args:
        mf: Mean field holding the grid.
        tol: Relative tolerance for the isotropy check.

    Returns:
        A dict with the measured spacings, the implied window step in pixels,
        and whether the two routes agree.
    """
    dx, dy = mf.dx, mf.dy
    step_px = dx * PX_PER_MM
    return {
        "dx_mm": dx,
        "dy_mm": dy,
        "isotropic": abs(dx - dy) / max(dx, dy) < tol,
        "implied_window_step_px": step_px,
        "nearest_power_of_two": 2 ** round(np.log2(step_px)) if step_px > 0 else 0,
        "consistent": abs(step_px - 2 ** round(np.log2(step_px))) / step_px < 0.1,
    }


# ── forces ────────────────────────────────────────────────────────────────────

# 10ms_20_10_sync(21-03-56).dat  ->  speed, yaw1, yaw2, sync flag, timestamp
_FORCE_RE = re.compile(
    r"^(?P<kind>baseline|(?P<speed>\d+)ms)"
    r"(?:_(?P<yaw1>-?\d+)_(?P<yaw2>-?\d+))?"
    r"(?P<sync>_sync)?"
    r"\((?P<h>\d+)-(?P<m>\d+)-(?P<s>\d+)\)\.dat$"
)


@dataclass
class ForceRecord:
    path: str
    is_baseline: bool
    is_sync: bool
    speed_ms: Optional[int]
    yaw: tuple  # (yaw1, yaw2) in degrees, () if not in the name
    seconds: int  # wall-clock time of day, for the drift fit
    data: Optional[np.ndarray] = None  # (n_samples, n_channels)

    @property
    def yaw_key(self) -> tuple:
        """Yaw configuration, with the unyawed case normalised to (0, 0).

        `baseline(18-20-04).dat` carries no yaw field while its run counterpart
        is `10ms_0_0_sync(...)`. Left as-is, the zero-yaw baselines key on ()
        and the zero-yaw run keys on (0, 0), so `baseline_drift_fit` would
        report no baselines for the one configuration that has the most.
        """
        return self.yaw if self.yaw else (0, 0)


def find_force_files(root: Optional[str] = None) -> list:
    """Parse every .dat under synced_forces/ into ForceRecords (no data read)."""
    root = root or ROOT
    base = os.path.join(root, "synced_forces")
    out = []
    for dirpath, _, filenames in os.walk(base):
        for fn in filenames:
            if not fn.endswith(".dat"):
                continue
            m = _FORCE_RE.match(fn)
            if not m:
                continue
            g = m.groupdict()
            out.append(
                ForceRecord(
                    path=os.path.join(dirpath, fn),
                    is_baseline=g["kind"] == "baseline",
                    is_sync=bool(g["sync"]),
                    speed_ms=int(g["speed"]) if g["speed"] else None,
                    yaw=(
                        (int(g["yaw1"]), int(g["yaw2"]))
                        if g["yaw1"] is not None
                        else ()
                    ),
                    seconds=int(g["h"]) * 3600 + int(g["m"]) * 60 + int(g["s"]),
                )
            )
    return sorted(out, key=lambda r: r.seconds)


def inspect_dat(path: str, n_lines: int = 40) -> dict:
    """Sniff a .dat: text or binary, header size, delimiter, column count.

    The top-level readme does not document the format. This detects it instead
    of assuming one.

    Args:
        path: Path to a `.dat` file.
        n_lines: Lines of the header to scan.

    Returns:
        A dict describing the file: whether it is binary, its size, and for text
        files the header length, delimiter, and column count.
    """
    with open(path, "rb") as fh:
        head = fh.read(8192)
    size = os.path.getsize(path)

    if b"\x00" in head:
        return {"path": path, "binary": True, "size_bytes": size, "preview": head[:64]}

    text = head.decode("utf-8", errors="replace").splitlines()[:n_lines]
    delim, n_header, n_cols = None, 0, 0
    for i, line in enumerate(text):
        for d in ("\t", ",", ";", None):
            parts = line.split(d) if d else line.split()
            vals = 0
            for p in parts:
                try:
                    float(p)
                    vals += 1
                except ValueError:
                    pass
            if vals >= 3 and vals == len(parts):
                delim, n_header, n_cols = d, i, len(parts)
                break
        if delim is not None or n_cols:
            break

    return {
        "path": path,
        "binary": False,
        "size_bytes": size,
        "n_header_lines": n_header,
        "delimiter": repr(delim) if delim else "whitespace",
        "n_columns": n_cols,
        "header": text[:n_header],
        # truncated: these are full-precision floats and 12 of them per line
        # runs to ~300 characters, which buries everything else in the report
        "first_data_lines": [
            (ln[:110] + " ...") if len(ln) > 110 else ln
            for ln in text[n_header : n_header + 3]
        ],
    }


def read_dat(path: str, n_channels: int = N_FORCE_CHANNELS) -> np.ndarray:
    """Read one force .dat as (n_channels, n_samples).

    The files hold raw little-endian float64 with no header, written by MATLAB
    in column-major order. The ordering is not recoverable from the file itself:
    a C-order reshape produces an array of the correct shape with interleaved
    channels, and nothing downstream detects it.

    Args:
        path: Path to a `.dat` file.
        n_channels: Number of channels in the record.

    Returns:
        The record, shape (n_channels, n_samples), matching the (N_s, N_t)
        convention used by `field_estimation.epod`.

    Raises:
        ValueError: If the file is empty, or if its length is not divisible by
            `n_channels`.
    """
    raw = np.fromfile(path, dtype="<f8")
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    if raw.size % n_channels:
        raise ValueError(
            f"{path}: {raw.size} float64 values is not divisible by "
            f"{n_channels} channels. Either the channel count is wrong or the "
            "file is truncated."
        )
    return raw.reshape((n_channels, -1), order="F")


def smooth_forces(F: np.ndarray, window: int) -> np.ndarray:
    """Moving average of ``window`` samples along the raw 2500 Hz force record.

    The reference notebook uses `window=100`, or 40 ms, applied before
    decimation to the PIV rate.

    No filtering was applied to the balance during acquisition, so the raw
    record carries sharp rig resonances. Left in, they dominate the leading
    sensor POD modes, and the components above the 125 Hz PIV Nyquist alias into
    the resolved band during decimation.

    `sync_forces(method="block")` already averages the 10 samples spanning each
    PIV frame, which antialiases the decimation but applies ten times less
    smoothing. Apply both. For a narrowband resonance use `notch_forces`
    instead: a moving average wide enough to suppress a 16 Hz peak also
    suppresses the shedding at 34 Hz.

    Args:
        F: Force record, shape (n_channels, n_samples), at `F_FORCE_HZ`.
        window: Moving-average length in samples. Values of 0 or 1 return `F`
            unchanged.

    Returns:
        The smoothed record, same shape as `F`.

    Raises:
        ValueError: If `F` is not two-dimensional, or if `window` exceeds the
            record length.
    """
    F = np.asarray(F, float)
    if window is None or window <= 1:
        return F
    if F.ndim != 2:
        raise ValueError(f"expected (n_channels, n_samples), got {F.shape}")
    if window > F.shape[1]:
        raise ValueError(f"window {window} exceeds the {F.shape[1]}-sample record")
    k = np.ones(window) / window
    # 'same' keeps the length so the pair-index arithmetic is unchanged; the
    # first and last window/2 samples are edge-biased and are dropped anyway by
    # the warm-up the delay embedding already requires.
    return np.stack([np.convolve(row, k, mode="same") for row in F])


def notch_forces(F: np.ndarray, freqs: Sequence[float], q: float = 8.0,
                 fs: float = None) -> np.ndarray:
    """
    Removes narrowband components from the raw force record.

    Zero-phase (filtfilt), so the force sample stays simultaneous with its PIV
    frame; an ordinary IIR notch would delay the record and break the pairing.

    Tested and rejected for the 14-17 Hz peak on the April data. That peak looks
    like a structural resonance -- up to 2.6e6 times the median spectral density
    -- but notching it halved the sensor observability (rho^2 on mode 1 fell
    0.663 -> 0.321) and moved the linear ceiling from NMSE 0.757 to 0.884. At
    D = 50 mm and U = 10 m/s, 15 Hz is St ~ 0.075, the wake-meandering band that
    dominates POD modes 1-3, so the peak is flow, not rig. Leave this off unless
    a specific frequency is shown to be non-physical.

    Args:
        F: Raw force record, shape (n_channels, n_samples).
        freqs: Frequencies to notch, in Hz.
        q: Quality factor; higher is narrower.
        fs: Sample rate. Defaults to F_FORCE_HZ.

    Returns:
        The filtered record, same shape as F.

    Raises:
        ValueError: If a frequency lies outside (0, fs / 2).
    """
    from scipy.signal import filtfilt, iirnotch

    F = np.asarray(F, float)
    if not freqs:
        return F
    fs = fs or F_FORCE_HZ
    for f0 in freqs:
        if not (0 < f0 < fs / 2):
            raise ValueError(f"notch {f0} Hz is outside (0, {fs / 2}) Hz")
        b, a = iirnotch(f0, q, fs)
        F = filtfilt(b, a, F, axis=1)
    return F


def auto_probe_points(fluid_mask: np.ndarray, n: int = 4) -> list:
    """
    Places n probes on valid fluid points, spread across the field of view.

    Hand-picked indices are fragile here: the mask covers 63% of the grid, the
    stored arrays are (Ny, Nx) while this module works in (Nx, Ny), and the
    low-res and high-res exports have different shapes. Coordinates copied
    between any two of those land on masked points.

    Uses farthest-point sampling from the centroid, so the result is
    deterministic and the probes are well separated rather than clustered.

    Args:
        fluid_mask: (Nx, Ny) boolean, True on valid points.
        n: Number of probes.

    Returns:
        A list of n (ix, iy) tuples, all on fluid points.

    Raises:
        ValueError: If the mask holds fewer than n fluid points.
    """
    pts = np.argwhere(np.asarray(fluid_mask, bool))
    if len(pts) < n:
        raise ValueError(f"{len(pts)} fluid points available, need {n}")
    chosen = [int(np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1)))]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(n - 1):
        chosen.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(pts - pts[chosen[-1]], axis=1))
    return [(int(pts[i][0]), int(pts[i][1])) for i in chosen]


def _nearest_fluid(fluid_mask: np.ndarray, ix: int, iy: int) -> tuple:
    """The valid grid point closest to (ix, iy), for an error message."""
    pts = np.argwhere(np.asarray(fluid_mask, bool))
    if not len(pts):
        return (-1, -1)
    i = int(np.argmin(np.hypot(pts[:, 0] - ix, pts[:, 1] - iy)))
    return (int(pts[i][0]), int(pts[i][1]))


def probe_signals(X: np.ndarray, points: Sequence[tuple]) -> np.ndarray:
    """Velocity at a few grid points, as extra observation channels.

    The reference notebook supplies four probes alongside the twelve force
    channels. A reconstruction using them is not a force-only reconstruction,
    and its score is not comparable with one that omits them.

    Args:
        X: Velocity fields, shape (Nu, Nt, Nx, Ny).
        points: Probe locations as `(ix, iy)` grid indices.

    Returns:
        Probe signals, shape (Nu * len(points), Nt), ordered by point then by
        component.

    Raises:
        ValueError: If `X` is not four-dimensional, or if a probe lands on an
            invalid vector.
        IndexError: If a probe lies outside the grid.
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"expected (Nu, Nt, Nx, Ny), got {X.shape}")
    Nu, Nt, Nx, Ny = X.shape
    out = []
    for ix, iy in points:
        if not (0 <= ix < Nx and 0 <= iy < Ny):
            raise IndexError(f"probe ({ix}, {iy}) is outside the {Nx}x{Ny} grid")
        col = X[:, :, ix, iy]  # (Nu, Nt)
        if not np.isfinite(col).all():
            # 63% of this grid is the solid body, so a hand-picked index is more
            # likely to be masked than not, and the message has to say what to
            # use instead -- the failure otherwise costs a whole queue wait to
            # learn nothing but "not that one".
            fluid = np.isfinite(X).all(axis=(0, 1))
            near = _nearest_fluid(fluid, ix, iy)
            n_bad = int((~np.isfinite(col)).any(axis=0).sum())
            raise ValueError(
                f"probe ({ix}, {iy}) sits on an invalid vector in {n_bad} of "
                f"{Nt} frames -- it would feed NaN into the observation matrix. "
                f"{100 * (~fluid).mean():.0f}% of this {Nx}x{Ny} grid is masked. "
                f"Nearest fluid point: {near}. "
                f"Prefer --n-probes N, which places them automatically; "
                f"well-separated fluid points here are "
                f"{auto_probe_points(fluid, min(4, int(fluid.sum())))}."
            )
        out.append(col)
    return np.concatenate(out, axis=0)


def force_index_for_pair(pair_index: int) -> int:
    """Force sample index synchronous with PIV pair `pair_index`.

    `PIV_PAIR_000201-000202.npz` derives from raw images 201 and 202, so vector
    field j uses images 2j+1 and 2j+2 and pair index a gives j = (a - 1) / 2.
    The balance samples at 2500 Hz against the PIV's 250 Hz, so field j lands on
    force sample 10j.

    The reference notebook hardcodes the resulting offset of 1000, which holds
    only for a run starting at pair 201. Deriving it from the filename keeps a
    run that starts elsewhere aligned.

    Args:
        pair_index: PIV pair index parsed from a snapshot filename.

    Returns:
        Index of the synchronous force sample.
    """
    return FORCE_PER_PIV * ((pair_index - 1) // 2)


def sync_forces(
    force: np.ndarray,
    pair_indices,
    method: str = "block",
) -> np.ndarray:
    """Resample a (n_ch, n_samples) force record onto PIV snapshot times.

    Args:
        force: Force record, shape (n_channels, n_samples).
        pair_indices: PIV pair index per snapshot, from `pair_indices`.
        method: Resampling method. One of:
            `"block"`: Mean over the 10 force samples spanning each PIV frame.
                The default, and an exact antialiasing filter for the
                decimation.
            `"decimate"`: Every 10th sample. Reproduces the reference notebook,
                and aliases 125-1250 Hz onto the resolved band. Rig resonances
                were not filtered during acquisition, so that band carries real
                energy.
            `"nearest"`: The single nearest sample, without averaging.

    Returns:
        The resampled record, shape (n_channels, len(pair_indices)).

    Raises:
        ValueError: If `method` is unknown, or if the force record is too short
            to cover every PIV frame.
    """
    idx = np.asarray([force_index_for_pair(int(p)) for p in pair_indices])
    n = force.shape[1]
    if idx.max() + FORCE_PER_PIV > n:
        keep = idx + FORCE_PER_PIV <= n
        raise ValueError(
            f"force record has {n} samples but PIV needs up to "
            f"{idx.max() + FORCE_PER_PIV}. Only {keep.sum()}/{len(idx)} PIV "
            "frames are covered -- truncate the snapshot list to match."
        )

    if method == "decimate":
        return force[:, idx]
    if method == "nearest":
        return force[:, idx + FORCE_PER_PIV // 2]
    if method == "block":
        # (n_ch, n_frames, FORCE_PER_PIV) -> mean over the last axis
        win = idx[:, None] + np.arange(FORCE_PER_PIV)[None, :]
        return force[:, win].mean(axis=2)
    raise ValueError(f"method must be block/decimate/nearest, got {method!r}")



# ── baseline drift correction ─────────────────────────────────────────────────


def baseline_drift_fit(records: list, degree: int = 1) -> dict:
    """Fit each channel's tunnel-off drift against wall-clock time, per yaw.

    Follows the procedure in the folder readme: take every tunnel-off
    measurement at a given yaw, fit balance drift against wall-clock time, and
    evaluate that fit at the time of the real measurement. Without it, a drift
    ramp remains in every channel and becomes the leading sensor POD mode.

    Args:
        records: `ForceRecord` objects, including the baselines.
        degree: Polynomial degree of the drift fit.

    Returns:
        A dict mapping each yaw key to polynomial coefficients, shape
        (degree + 1, n_channels).

    Raises:
        ValueError: If a yaw configuration has too few baseline files for the
            requested degree.
    """
    fits = {}
    by_yaw = {}
    for r in records:
        if r.is_baseline:
            by_yaw.setdefault(r.yaw_key, []).append(r)

    for yaw, rs in by_yaw.items():
        if len(rs) < degree + 1:
            raise ValueError(
                f"yaw {yaw}: {len(rs)} baseline file(s), need at least "
                f"{degree + 1} for a degree-{degree} drift fit"
            )
        t = np.array([r.seconds for r in rs], dtype=float)
        # each baseline file is one tunnel-off recording; its per-channel mean
        # is the balance's zero at that wall-clock time
        means = np.stack(
            [
                np.nanmean(r.data if r.data is not None else read_dat(r.path), axis=1)
                for r in rs
            ]
        )  # (n_files, n_channels)
        fits[yaw] = np.polyfit(t, means, degree)
    return fits


def apply_drift_correction(data: np.ndarray, seconds: int, coeffs: np.ndarray):
    """Subtracts the fitted drift, evaluated at this measurement's wall time.

    Args:
        data: Force record, shape (n_channels, n_samples).
        seconds: Wall-clock time of the measurement.
        coeffs: Drift coefficients from `baseline_drift_fit`.

    Returns:
        The corrected record, same shape as `data`.

    Raises:
        ValueError: If the fit and the data disagree on channel count.
    """
    offset = np.polyval(coeffs, float(seconds))  # (n_channels,)
    if offset.shape[0] != data.shape[0]:
        raise ValueError(
            f"drift fit has {offset.shape[0]} channels, data has {data.shape[0]}"
        )
    return data - offset[:, None]


# ── assembled case ────────────────────────────────────────────────────────────


@dataclass
class Case:
    """One run, loaded and paired: snapshots, synchronised forces, geometry.

    `X` holds the repository's (Nu, Nt, Nx, Ny) layout with NaN at invalid
    vectors, as the `Projector` classes expect. `flat` produces the (N_x, N_t)
    column-per-snapshot matrix `field_estimation.epod` expects, under the same masking
    convention, so linear and neural models fit on identical data.

    Invalid points are dropped rather than zero-filled. The reference notebook
    zero-fills, which the SVD reads as a measured zero rather than as missing
    data. Pass `flat(fill=0.0)` to reproduce that.
    """

    run: str
    X: np.ndarray  # (Nu, Nt, Nx, Ny), NaN at invalid
    S: np.ndarray  # (n_channels, Nt), synchronised, drift-corrected
    pairs: list  # PIV pair index per snapshot
    fluid_mask: np.ndarray  # (Nx, Ny) True where the point survived the mask rule
    mf: Optional["MeanField"] = None
    force_file: str = ""
    drift_applied: bool = False
    invalid_frac: Optional[np.ndarray] = None  # (Nx, Ny), BEFORE any masking

    @property
    def shape(self) -> tuple:
        return self.X.shape

    @property
    def n_t(self) -> int:
        return self.X.shape[1]

    @property
    def grid_shape(self) -> tuple:
        return (self.X.shape[0], self.X.shape[2], self.X.shape[3])

    def flat(self, fill: Optional[float] = None) -> np.ndarray:
        """(N_fluid * Nu, N_t) data matrix, columns as snapshots.

        Row ordering is `f * Nu + u`, fluid point major and component minor,
        matching `Projector._to_flat`.

        Args:
            fill: Value substituted at invalid vectors. `None` drops those
                points instead of keeping them.

        Returns:
            The data matrix, shape (N_fluid * Nu, N_t).
        """
        Nu, Nt, Nx, Ny = self.X.shape
        A = self.X.reshape(Nu, Nt, Nx * Ny)
        if fill is None:
            A = A[:, :, self.fluid_mask.ravel()]
        else:
            A = np.where(np.isnan(A), fill, A)
        return A.transpose(2, 0, 1).reshape(-1, Nt)

    def unflat(self, Q: np.ndarray, fill_used: bool = False, dtype=None) -> np.ndarray:
        """Inverse of ``flat``: (N_x, N_t) -> (Nu, Nt, Nx, Ny) with NaN holes.

        Works for a record assembled by `concat_cases`, provided the `Case`
        carries the intersected mask. Use this rather than `case.X` to rebuild
        the grid for a concatenated record: `case.X` holds one run's snapshots
        and does not align with a concatenated `Q`.

        Args:
            Q: Data matrix, shape (N_x, N_t).
            fill_used: Whether `Q` came from `flat(fill=...)`, which keeps every
                grid point.
            dtype: Output dtype. Defaults to `Q.dtype`. Pass float32 when
                feeding an autoencoder; a multi-run grid exceeds a gigabyte in
                float64.

        Returns:
            The gridded record, shape (Nu, Nt, Nx, Ny), with NaN at masked
            points.
        """
        Nu, _, Nx, Ny = self.X.shape
        n_t = Q.shape[1]
        dtype = dtype or Q.dtype
        keep = np.ones(Nx * Ny, bool) if fill_used else self.fluid_mask.ravel()
        A = Q.reshape(int(keep.sum()), Nu, n_t).transpose(1, 2, 0)  # (Nu, Nt, N_keep)
        out = np.full((Nu, n_t, Nx * Ny), np.nan, dtype=dtype)
        out[:, :, keep] = A
        return out.reshape(Nu, n_t, Nx, Ny)

    def yaw(self) -> tuple:
        return yaw_from_run(self.run)


def yaw_from_run(run: str) -> tuple:
    """Parses the yaw pair from a run name.

    `4p5d_10ms_yaw_30_15_0` gives `(30, 15)`. Only the first two discs appear in
    the force filenames, although the rig documentation records balances on
    discs 2 and 3.

    Args:
        run: Run directory name.

    Returns:
        The yaw pair in degrees.
    """
    tail = run.split("yaw_")[-1].split("_")
    return (int(tail[0]), int(tail[1]))


def _fill_dropouts(X: np.ndarray, fluid: np.ndarray, how: str = "interp") -> np.ndarray:
    """Fills invalid vectors at the points being kept, in time.

    A point invalid in a handful of the 6085 frames carries a real measurement
    in the other 6080. Discarding it for all time to avoid filling those few is
    the trade that costs most of the field.

    Points invalid in *every* frame -- the solid body -- are set to a constant.
    After the mean is subtracted they then carry exactly zero fluctuation, so
    they contribute nothing to the POD and nothing to the error, which is the
    same thing dropping them achieves, without removing them from the grid.

    Args:
        X: Velocity fields, shape (Nu, Nt, Nx, Ny), NaN at invalid.
        fluid: (Nx, Ny) boolean, True at points being kept.
        how: `"interp"` fills by linear interpolation in time; `"zero"` writes
            zeros, which is what the reference notebook's `nan_to_num` does.
            Zero is a *measured value* to an SVD and a gap is not, so interp is
            the default; use zero only to reproduce the notebook.

    Returns:
        `X` with the kept points' invalid vectors filled.

    Raises:
        ValueError: If `how` is not "interp" or "zero".
    """
    if how not in ("interp", "zero"):
        raise ValueError(f"how must be 'interp' or 'zero', got {how!r}")
    X = np.array(X, copy=True)
    Nu, Nt, Nx, Ny = X.shape
    if how == "zero":
        bad = ~np.isfinite(X)
        bad[:, :, ~fluid] = False
        X[bad] = 0.0
        return X

    t = np.arange(Nt)
    # Only visit points that actually have a gap: on this grid that is a few
    # thousand of ~16k, and the check is far cheaper than the interpolation.
    holes = np.isnan(X).any(axis=1) & fluid[None, :, :]   # (Nu, Nx, Ny)
    for u, ix, iy in np.argwhere(holes):
        col = X[u, :, ix, iy]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        else:
            col[bad] = np.interp(t[bad], t[~bad], col[~bad])
    return X


def build_case(
    run: str,
    n_snapshots: Optional[int] = None,
    stride: int = 1,
    highres: bool = True,
    sync_method: str = "block",
    drift_correct: bool = True,
    force_smooth: int = 0,
    notch: Optional[Sequence[float]] = None,
    notch_q: float = 8.0,
    probes: Optional[Sequence[tuple]] = None,
    n_probes: int = 0,
    mask_tol: float = 1.0,
    mask_fill: str = "interp",
    root: Optional[str] = None,
    dtype=np.float32,
    verbose: bool = True,
) -> Case:
    """Load one run and pair it with its synchronised, drift-corrected forces.

    The entry point for every downstream script, so that the PIV/force pairing,
    the drift correction, and the mask convention are decided in one place.

    Args:
        run: Run directory name.
        n_snapshots: Maximum snapshots to read.
        stride: Snapshot decimation factor.
        highres: Whether to read the high-resolution snapshots.
        sync_method: Force resampling method; see `sync_forces`.
        drift_correct: Whether to subtract the fitted tunnel-off drift. Falls
            back to a constant offset given one baseline file, and warns without
            failing given none. An uncorrected record's leading sensor POD mode
            is a drift ramp.
        force_smooth: Moving-average window applied to the raw force record; see
            `smooth_forces`.
        notch: Frequencies to notch out of the raw force record; see
            `notch_forces`.
        notch_q: Notch quality factor.
        probes: In-field velocity probe locations; see `probe_signals`.
        mask_tol: Keep a grid point unless it is invalid in more than this
            fraction of frames, filling what remains. The default of 1.0 keeps
            every point, which is what the reference notebook does -- it calls
            `nan_to_num` on the whole field and has no mask rule at all.

            0.0 is the old strict "valid in every snapshot" rule. It is right
            for a solid body, which never moves, and wrong for spurious PIV
            vectors, which do not appear at random: they cluster in shear
            layers and the wake core. On this dataset it discarded 9920 of
            15741 points (63%) to remove a body of at most 112, and the points
            it discarded were valid in ~99% of frames each. Pass 0.0 only to
            reproduce an old result.
        mask_fill: How the kept points' invalid vectors are filled; see
            `_fill_dropouts`.
        root: Data root. Defaults to `ROOT`.
        dtype: Snapshot dtype.
        verbose: Whether to print a summary of what was loaded.

    Returns:
        The assembled `Case`.

    Raises:
        FileNotFoundError: If the run or its sync force file is missing.
    """
    say = print if verbose else (lambda *a, **k: None)

    X = load_run(run, highres=highres, stride=stride, max_snapshots=n_snapshots,
                 root=root, dtype=dtype)
    pairs = pair_indices(run, highres=highres, stride=stride,
                         max_snapshots=n_snapshots, root=root)
    Nu, Nt, Nx, Ny = X.shape

    # Fraction of frames in which each point has an invalid vector, recorded
    # before anything masks X. Downstream code sets X[:, :, ~fluid] = NaN in
    # place, so this is the only surviving record of what the raw dropout
    # actually looked like -- and the mask audit that reads it back off X
    # instead measures its own output and concludes the mask is fine.
    invalid = np.isnan(X).any(axis=0)              # (Nt, Nx, Ny)
    invalid_frac = invalid.mean(axis=0)            # (Nx, Ny)

    # A point is dropped when it is invalid in more than `mask_tol` of frames.
    # At the default of 0 that is the strict "valid in every snapshot" rule.
    #
    # The strict rule is right for a solid body, which never moves, and wrong
    # for spurious PIV vectors, which do: this run carries ~112 invalid vectors
    # per frame (0.7%) against a body of 254 points (1.6%), and the union over
    # 6085 frames discards 9920 points (63%). Spurious vectors cluster in shear
    # layers and the wake core, so the strict rule preferentially deletes the
    # part of the field the load cells could plausibly see -- and then the
    # reconstruction is scored on what is left, which is largely freestream.
    fluid = invalid_frac <= mask_tol
    n_hole = int(((invalid_frac > 0) & fluid).sum())
    X = _fill_dropouts(X, fluid, mask_fill)
    say(f"  {run}: X {X.shape}, {int((~fluid).sum())}/{Nx * Ny} points masked "
        f"({100 * (~fluid).mean():.2f}%) at mask_tol={mask_tol:g}; "
        f"{n_hole} kept points had gaps, filled by {mask_fill}")
    if mask_tol < 1.0:
        always = int((invalid_frac >= 1.0).sum())
        say(f"     (of the {int((~fluid).sum())} dropped, {always} are invalid in "
            f"every frame -- the rest carry real data that is being discarded)")

    recs = find_force_files(root)
    yaw = yaw_from_run(run)
    match = [r for r in recs if r.is_sync and r.yaw_key == yaw]
    if not match:
        raise FileNotFoundError(
            f"no sync force file for yaw {yaw}; have "
            f"{sorted({r.yaw_key for r in recs if r.is_sync})}"
        )
    if len(match) > 1:
        say(f"  !! {len(match)} sync files at yaw {yaw}, using the earliest: "
            + ", ".join(os.path.basename(r.path) for r in match))
    rec = match[0]
    F = read_dat(rec.path)

    drift_applied = False
    if drift_correct:
        baselines = [r for r in recs if r.is_baseline and r.yaw_key == yaw]
        if len(baselines) >= 2:
            fits = baseline_drift_fit(baselines, degree=1)
            F = apply_drift_correction(F, rec.seconds, fits[yaw])
            drift_applied = True
            say(f"  drift: linear fit over {len(baselines)} baseline files")
        elif len(baselines) == 1:
            off = np.nanmean(read_dat(baselines[0].path), axis=1)
            F = F - off[:, None]
            drift_applied = True
            say("  drift: only one baseline file -- constant offset, no slope")
        else:
            say(f"  !! no baseline files at yaw {yaw}; drift NOT removed. The "
                "leading sensor POD mode will be a ramp with no fluid content.")

    if notch:
        F = notch_forces(F, notch, q=notch_q)
        say(f"  forces: notched {list(notch)} Hz (Q={notch_q:g}, zero-phase) "
            "-- rig resonance")

    if force_smooth and force_smooth > 1:
        F = smooth_forces(F, force_smooth)
        say(f"  forces: {force_smooth}-sample moving average "
            f"({1000 * force_smooth / F_FORCE_HZ:.0f} ms, ~{F_FORCE_HZ / force_smooth / 2:.0f} Hz cutoff) "
            "before decimation")

    S = sync_forces(F, pairs, method=sync_method)
    say(f"  forces: S {S.shape} from {os.path.basename(rec.path)} "
        f"({sync_method} resampling)")

    if n_probes and not probes:
        probes = auto_probe_points(fluid, n_probes)
        say(f"  probes: auto-placed {n_probes} at {probes}")
    if probes:
        P = probe_signals(X, probes)
        S = np.concatenate([S, P], axis=0)
        say(f"  probes: +{P.shape[0]} velocity channels at {list(probes)} "
            f"-> S {S.shape}   (NOT a force-only reconstruction)")

    try:
        mf = load_meanfield(os.path.join(root or ROOT, MEANFIELD))
    except Exception as e:  # the mean field is only needed for axis labels
        say(f"  (no mean field: {e})")
        mf = None

    return Case(run=run, X=X, S=S, pairs=pairs, fluid_mask=fluid, mf=mf,
                force_file=rec.path, drift_applied=drift_applied,
                invalid_frac=invalid_frac)


def concat_cases(cases: list) -> tuple:
    """Concatenate several runs into one (Q, S, run_id) triple.

    Used for the cross-yaw study: fit on some yaw settings, test on a held-out
    one. The fluid mask is intersected across runs, so every row of `Q` refers
    to the same physical point in all of them.

    Concatenation introduces a discontinuity at each seam. Split on run
    boundaries so no delay window spans one.

    Args:
        cases: Loaded `Case` objects, all on the same grid.

    Returns:
        A tuple `(Q, S, run_id)`, where `run_id` labels each column with its
        source run.

    Raises:
        ValueError: If `cases` is empty, or if the runs use different grids.
    """
    if not cases:
        raise ValueError("no cases")
    shapes = {c.grid_shape for c in cases}
    if len(shapes) != 1:
        raise ValueError(f"runs have different grids: {shapes}")
    mask = np.logical_and.reduce([c.fluid_mask for c in cases])

    Qs, Ss, ids = [], [], []
    for i, c in enumerate(cases):
        Nu, Nt, Nx, Ny = c.X.shape
        A = c.X.reshape(Nu, Nt, Nx * Ny)[:, :, mask.ravel()]
        Qs.append(A.transpose(2, 0, 1).reshape(-1, Nt))
        Ss.append(c.S)
        ids.append(np.full(Nt, i))
    return np.concatenate(Qs, 1), np.concatenate(Ss, 1), np.concatenate(ids)
