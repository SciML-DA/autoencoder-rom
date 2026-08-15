"""
wake_experiment.py
==================

Reader for the April porous-disc wind-tunnel experiment on RDS:

    /rds/general/project/immanuel/live/Seagate/april_experiment

Layout, as confirmed against the folder's own ``readme.md``:

    4p5d_10ms_yaw_<a>_<b>_<c>/
        piv_raw_images/
        piv_snapshots/             (55, 90)  per snapshot
        piv_snapshots_highres/     (99, 159) per snapshot   <- use this one
    synced_forces/<date>/
        10ms_<y1>_<y2>_sync(HH-MM-SS).dat    paired with a PIV buffer
        10ms_<y1>_<y2>(HH-MM-SS).dat         not paired
        baseline[_<y1>_<y2>](HH-MM-SS).dat   tunnel OFF, drift reference
        readme.txt
    meanfield_4p5d_10ms_yaw_0_0_0.npz        X, Y, u_bar, v_bar, Rxx/Rxy/Ryy, mask
    readme.md
    epod_reconstruction_clean.ipynb

Two facts about this data that are easy to get wrong, and are handled here:

**Axis order.** The stored arrays are ``(99, 159)`` with X spanning 0..446 mm
over the 159 axis and Y spanning 0..279 mm over the 99 axis -- i.e. they are
``(Ny, Nx)``, row-major in Y. Every Projector in this repo wants
``(Nu, Nt, Nx, Ny)``. `load_run` transposes, and `_check_grid` asserts the
orientation rather than trusting it, because a wrong transpose here trains
perfectly happily and is silently garbage.

**Coordinates are not in the snapshots.** Snapshot ``.npz`` files hold only
``u`` and ``v``. X, Y and the body mask come from the mean-field file. The
readme's alternative route -- image pixels over the 5.7384 px/mm scaling factor
-- is implemented as `grid_from_scaling` and used by `check_scaling` as an
independent cross-check.

Usage
-----
::

    from datasets.wake_experiment import ROOT, load_meanfield, load_run

    mf = load_meanfield()
    U  = load_run("4p5d_10ms_yaw_0_0_0", max_snapshots=2000)   # (2, Nt, Nx, Ny)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
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
    "grid_from_scaling",
    "check_scaling",
    "ForceRecord",
    "inspect_dat",
    "read_dat",
    "find_force_files",
    "baseline_drift_fit",
    "apply_drift_correction",
]

# `RDS_ROOT` lets you point at a local copy without editing anything
ROOT = os.environ.get(
    "RDS_ROOT", "/rds/general/project/immanuel/live/Seagate/april_experiment"
)

PX_PER_MM = 5.7384  # April experiment calibration, from readme.md
F_PIV_HZ = 250.0  # PIV acquisition frequency
DT_INTERFRAME = 100e-6  # s, inter-frame separation within a pair

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
    """Assert the stored arrays really are (Ny, Nx) before we transpose them.

    X must vary along the last axis and be constant down the first; Y the other
    way round. If a future export flips this, the assertion fires here instead
    of producing a plausible-looking POD of a transposed field.
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
        out = {k: z[k].T for k in ("u_bar", "v_bar", "Rxx", "Rxy", "Ryy") if k in z.files}

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
    """Return a mask that is True on fluid points.

    The stored `mask` is a bool array with no stated convention. Rather than
    guess, decide from the data: the fluid side is whichever side carries the
    non-zero mean velocity. Falls back to "the majority side is fluid", which is
    right whenever the body occupies less than half the field of view -- here
    the body is ~112 points out of 15741.
    """
    if u_bar is not None and np.any(u_bar != 0):
        inside = np.abs(u_bar[mask]).mean() if mask.any() else 0.0
        outside = np.abs(u_bar[~mask]).mean() if (~mask).any() else 0.0
        return mask if inside >= outside else ~mask
    return mask if mask.sum() >= mask.size / 2 else ~mask


def list_snapshots(run: str, highres: bool = True, root: Optional[str] = None):
    """Sorted snapshot paths for a run.

    Sorted by the *pair index* parsed out of `PIV_PAIR_000201-000202.npz`, not
    lexicographically, so a change in zero-padding cannot silently reorder time.
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

    NaN is exactly the repo's convention for solid-body / invalid points, and
    the snapshots already use it (112 per frame in the yaw_0 run), so nothing is
    reinterpreted here -- only transposed.

    6085 snapshots at (99, 159) is ~1.2 GB in float64 and ~600 MB in float32.
    Default is float32; pass ``max_snapshots`` while prototyping.
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
            vs[i] = z["v"].T
    return np.stack([us, vs])  # (2, Nt, Nx, Ny)


def _peek_shape(path: str) -> tuple:
    with np.load(path) as z:
        return z["u"].shape  # (Ny, Nx)


# ── grid cross-check ──────────────────────────────────────────────────────────


def grid_from_scaling(n_x: int, n_y: int, window_step_px: float):
    """Grid spacing in mm from the calibration, the readme's alternative route.

    ``window_step_px`` is the PIV interrogation window *step*, not the window
    size -- with 50% overlap on 32 px windows it is 16.
    """
    d = window_step_px / PX_PER_MM
    return np.arange(n_x) * d, np.arange(n_y) * d


def check_scaling(mf: MeanField, tol: float = 0.02) -> dict:
    """Cross-check the mean-field grid against the px/mm calibration.

    Two independent routes to the same physical spacing. If they disagree, the
    vector spacing is not what you assumed and every length -- and therefore
    every Strouhal number -- is wrong by that factor. Cheap to check, expensive
    to discover later.
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

    The format is not documented in the top-level readme (there is a readme.txt
    inside synced_forces/ that may say more). This works it out rather than
    assuming, so `read_dat` can be written once against the answer.
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


def read_dat(path: str, n_header: Optional[int] = None, delimiter=None) -> np.ndarray:
    """Read one force .dat as (n_samples, n_channels).

    Header length and delimiter are sniffed by default; pass them explicitly
    once `inspect_dat` has told you what they are, to skip the sniff.
    """
    if n_header is None or delimiter is None:
        info = inspect_dat(path)
        if info.get("binary"):
            raise NotImplementedError(
                f"{path} is binary; inspect_dat says {info['size_bytes']} bytes. "
                "Check synced_forces/*/readme.txt for the record layout."
            )
        n_header = info["n_header_lines"] if n_header is None else n_header
        if delimiter is None:
            d = info["delimiter"]
            delimiter = None if d == "whitespace" else eval(d)
    return np.loadtxt(path, skiprows=n_header, delimiter=delimiter, ndmin=2)


# ── baseline drift correction ─────────────────────────────────────────────────


def baseline_drift_fit(records: list, degree: int = 1) -> dict:
    """Fit each channel's tunnel-off drift against wall-clock time, per yaw.

    The procedure the folder's readme prescribes: take all the `baseline`
    (tunnel off) measurements at a given yaw configuration, fit the balance
    drift over wall-clock time, and evaluate that fit at the time of the real
    measurement to get what to subtract.

    Skipping this leaves a slow ramp in every channel. The sensor POD will then
    hand you that ramp as its leading mode -- a mode with no fluid content
    whatsoever, which the LSE will happily map onto flow structures.

    Returns {yaw_key: poly_coeffs (degree+1, n_channels)}.
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
        means = np.stack(
            [np.nanmean(r.data if r.data is not None else read_dat(r.path), axis=0)
             for r in rs]
        )
        fits[yaw] = np.polyfit(t, means, degree)
    return fits


def apply_drift_correction(data: np.ndarray, seconds: int, coeffs: np.ndarray):
    """Subtract the fitted drift, evaluated at this measurement's wall time."""
    offset = np.polyval(coeffs, float(seconds))  # (n_channels,)
    if offset.shape[-1] != data.shape[-1]:
        raise ValueError(
            f"drift fit has {offset.shape[-1]} channels, data has {data.shape[-1]}"
        )
    return data - offset
