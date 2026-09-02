#!/usr/bin/env python
"""
inspect_experiment.py
=====================

One-shot survey of the April experiment on RDS. Run this first, on cx3.

It validates everything `datasets.wake_experiment` assumes -- grid orientation,
mask polarity, NaN convention, calibration consistency -- and resolves the one
thing still unknown from the documentation: the layout of the force `.dat`
files. Everything it prints is either a check that must pass or a number the
force reader needs.

    module load tools/prod SciPy-bundle/2025.06-gfbf-2025a
    python scripts/inspect_experiment.py

Login-node safe: it reads one snapshot per run and the head of two .dat files,
nothing more.
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from datasets.wake_experiment import (  # noqa: E402
    F_FORCE_HZ,
    F_PIV_HZ,
    FORCE_PER_PIV,
    PX_PER_MM,
    ROOT,
    RUNS,
    check_scaling,
    find_force_files,
    force_index_for_pair,
    inspect_dat,
    list_snapshots,
    load_meanfield,
    pair_indices,
    read_dat,
    sync_forces,
)


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    print(f"RDS_ROOT = {ROOT}")
    if not os.path.isdir(ROOT):
        print("  !! not a directory. Set RDS_ROOT or run this on cx3.")
        return 2

    # ── mean field and grid ──────────────────────────────────────────────────
    rule("MEAN FIELD / GRID")
    mf = load_meanfield()
    print(f"  shape (Nx, Ny)      : {mf.shape}")
    print(f"  x range             : {mf.x.min():.1f} .. {mf.x.max():.1f} mm")
    print(f"  y range             : {mf.y.min():.1f} .. {mf.y.max():.1f} mm")
    print(f"  dx, dy              : {mf.dx:.4f}, {mf.dy:.4f} mm")

    sc = check_scaling(mf)
    print(
        f"  implied window step : {sc['implied_window_step_px']:.2f} px "
        f"(at {PX_PER_MM} px/mm)"
    )
    print(f"  isotropic grid      : {sc['isotropic']}")
    print(
        f"  power-of-two step   : {sc['consistent']} "
        f"(nearest {sc['nearest_power_of_two']})"
    )
    if not sc["consistent"]:
        print("  !! grid spacing is not a power-of-two pixel step. The vector")
        print("     spacing may not be what the calibration implies -- every")
        print("     length, and so every Strouhal number, scales with this.")

    n_fluid = int(mf.fluid_mask.sum())
    n_body = int((~mf.fluid_mask).sum())
    print(
        f"  mask                : {n_fluid} fluid, {n_body} body "
        f"({100 * n_body / mf.fluid_mask.size:.2f}% blocked)"
    )

    # disc diameter sanity: Re = U D / nu = 33000 at U = 10 -> D ~ 49.5 mm
    span_D = (mf.x.max() - mf.x.min()) / 49.5
    print(f"  FOV streamwise      : {span_D:.1f} D  (D ~ 49.5 mm from Re=33000)")

    # ── snapshots ────────────────────────────────────────────────────────────
    rule("PIV RUNS")
    for run in RUNS:
        # highres and lowres are reported independently: two of the runs ship
        # only one of the two, and failing the whole row on the missing one
        # hides the set that is actually there
        def _count(hr):
            try:
                return list_snapshots(run, highres=hr)
            except FileNotFoundError:
                return []

        hi, lo = _count(True), _count(False)
        if not hi and not lo:
            print(f"  {run:26s} MISSING entirely")
            continue
        with np.load(hi[0]) as z:
            shape = z["u"].shape
            nan = int(np.isnan(z["u"]).sum())
        dur = len(hi) / F_PIV_HZ
        print(
            f"  {run:26s} highres {len(hi):5d} @ {shape}  "
            f"lowres {len(lo):5d}  nan/frame {nan:4d}  {dur:.1f} s"
        )

    print()
    print("  NOTE: highres and lowres counts differ and the pair indices start")
    print("        at different values -- they are NOT row-for-row aligned.")
    print("        Pick one and stay in it.")

    # mask vs NaN agreement, on the baseline run
    hi = list_snapshots(RUNS[0], highres=True)
    with np.load(hi[0]) as z:
        nan_mask = np.isnan(z["u"]).T
    agree = np.array_equal(nan_mask, ~mf.fluid_mask)
    print(f"\n  snapshot NaNs match the mean-field body mask: {agree}")
    if not agree:
        overlap = int((nan_mask & ~mf.fluid_mask).sum())
        print(
            f"  !! {int(nan_mask.sum())} NaN vs {int((~mf.fluid_mask).sum())} "
            f"masked, {overlap} in common. Invalid vectors are not only the"
        )
        print("     body -- decide whether to drop or interpolate the rest.")

    # ── forces ───────────────────────────────────────────────────────────────
    rule("FORCE BALANCE")
    recs = find_force_files()
    if not recs:
        print("  no .dat files matched the expected naming pattern")
        return 1

    sync = [r for r in recs if r.is_sync]
    base = [r for r in recs if r.is_baseline]
    runs_ = [r for r in recs if not r.is_baseline and not r.is_sync]
    print(
        f"  {len(recs)} files: {len(sync)} sync, {len(base)} baseline, "
        f"{len(runs_)} other"
    )

    print("\n  sync files (these pair with the PIV runs):")
    for r in sync:
        print(
            f"    yaw {str(r.yaw_key):12s} t={r.seconds:6d}s  "
            f"{os.path.basename(r.path)}"
        )

    print("\n  baseline files per yaw (need >= 2 for a linear drift fit):")
    by_yaw = {}
    for r in base:
        by_yaw.setdefault(r.yaw_key, []).append(r)
    for yaw, rs in sorted(by_yaw.items(), key=lambda kv: str(kv[0])):
        flag = "" if len(rs) >= 2 else "   !! too few for a drift fit"
        print(f"    yaw {str(yaw):12s} {len(rs)} file(s){flag}")

    # ── the .dat format ──────────────────────────────────────────────────────
    rule("FORCE .dat FORMAT")
    readme = os.path.join(os.path.dirname(sync[0].path), "readme.txt")
    if os.path.exists(readme):
        print(f"  --- {readme} ---")
        with open(readme) as fh:
            print("  " + "\n  ".join(fh.read().splitlines()[:40]))
        print()

    for r in (sync[0], base[0]):
        print(f"  --- {os.path.basename(r.path)} ---")
        try:
            a = read_dat(r.path)
            print(f"    parsed  : {a.shape} (n_channels, n_samples) {a.dtype}")
            print(f"    duration: {a.shape[1] / F_FORCE_HZ:.2f} s at {F_FORCE_HZ:g} Hz")
            print(
                f"    per-channel mean: "
                f"{np.array2string(a.mean(1), precision=3, max_line_width=200)}"
            )
            print(
                f"    per-channel std : "
                f"{np.array2string(a.std(1), precision=3, max_line_width=200)}"
            )
        except Exception as e:
            print(f"    PARSE FAILED: {type(e).__name__}: {e}")
            print(f"    diagnostics: {inspect_dat(r.path)}")
        print()

    # ── pairing feasibility ──────────────────────────────────────────────────
    rule("PIV <-> FORCE PAIRING")
    try:
        F = read_dat(sync[0].path)
        pairs = pair_indices(RUNS[0], highres=True)
        print(
            f"  PIV snapshots      : {len(pairs)}  "
            f"(pair index {pairs[0]} .. {pairs[-1]})"
        )
        print(
            f"  force samples      : {F.shape[1]} at {F_FORCE_HZ:g} Hz "
            f"= {F.shape[1] / F_FORCE_HZ:.2f} s"
        )
        print(f"  force per PIV frame: {FORCE_PER_PIV}")
        print(
            f"  first PIV pair {pairs[0]} -> force sample "
            f"{force_index_for_pair(pairs[0])}"
        )
        print(
            f"  last  PIV pair {pairs[-1]} -> force sample "
            f"{force_index_for_pair(pairs[-1])}"
        )

        need = force_index_for_pair(pairs[-1]) + FORCE_PER_PIV
        if need > F.shape[1]:
            n_ok = sum(
                1
                for p in pairs
                if force_index_for_pair(p) + FORCE_PER_PIV <= F.shape[1]
            )
            print(
                f"  !! needs {need} samples, have {F.shape[1]}. Only {n_ok} of "
                f"{len(pairs)} PIV frames are covered."
            )
            print(f"     -> load_run(..., max_snapshots={n_ok})")
        else:
            print(
                f"  OK: all {len(pairs)} frames covered "
                f"({F.shape[1] - need} samples spare)"
            )

        S = sync_forces(F, pairs[: min(len(pairs), 5900)], method="block")
        print(f"  sync_forces(block) -> {S.shape}")
    except Exception:
        traceback.print_exc(limit=3)

    print("\ndone.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
