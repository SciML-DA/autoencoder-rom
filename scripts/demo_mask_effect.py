#!/usr/bin/env python
"""
demo_mask_effect.py
===================

Why "drop a point if it is invalid in any frame" destroys a wake reconstruction,
on synthetic data where the truth is known.

    python scripts/demo_mask_effect.py

`build_case` used to keep a grid point only if it carried a valid vector in
*every* snapshot. That is the right rule for a solid body, which never moves,
and the wrong one for spurious PIV vectors, which do not appear at random: they
concentrate in high shear, strong out-of-plane motion and poor seeding -- that
is, in the wake. So the rule preferentially deletes the region whose motion the
load cells are measuring, keeps the quiet freestream, and then the estimator is
scored on how well it reconstructs the part of the field nothing can see.

The damage is invisible in the obvious summary statistic. Below, the strict rule
removes ~16% of the grid points and ~99.8% of the wake energy.

Run it with `--real` to print the same arithmetic for the April experiment
numbers from the RDS survey, without needing the data.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.epod import PODLSE, delay_embed, nmse  # noqa: E402


def synthetic(seed: int = 0):
    """A compact sensor-visible 'wake' plus a broad unpredictable 'freestream'."""
    rng = np.random.default_rng(seed)
    nx, ny, nt = 60, 40, 2000
    x = np.linspace(0, 1, nx)[:, None]
    y = np.linspace(0, 1, ny)[None, :]
    t = np.arange(nt)

    wake = np.exp(-((x - 0.30) ** 2 + (y - 0.5) ** 2) / 0.010)
    free = 0.25 * np.sin(2 * np.pi * x) * np.ones_like(y)

    a_w = np.sin(2 * np.pi * t / 40) + 0.2 * rng.standard_normal(nt)
    F = wake[..., None] * a_w + free[..., None] * rng.standard_normal(nt)
    S = (a_w + 0.1 * rng.standard_normal(nt))[None, :]

    # dropout probability tracks the wake, as spurious vectors do
    p = 0.02 * wake / wake.max() + 2e-5
    bad = rng.random((nx, ny, nt)) < p[..., None]
    return np.where(bad, np.nan, F), S, bad, wake, t


def score(Fn, S, mask, t, interp: bool) -> float:
    Q = Fn.reshape(-1, Fn.shape[-1])[mask.ravel()].copy()
    if interp:
        for i in range(Q.shape[0]):
            b = ~np.isfinite(Q[i])
            if b.any():
                Q[i, b] = np.interp(t[b], t[~b], Q[i, ~b])
    else:
        Q = np.nan_to_num(Q)
    tr, te = slice(0, 1500), slice(1600, Q.shape[1])
    Sd = delay_embed(S, 5)
    m = PODLSE(r_field=2, ridge=1e-4).fit(Q[:, tr], Sd[:, tr])
    return nmse(Q[:, te], m.predict(Sd[:, te]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", action="store_true",
                   help="also print the April-experiment arithmetic")
    args = p.parse_args()

    Fn, S, bad, wake, t = synthetic()
    frac_bad = bad.mean(axis=2)
    w = wake.ravel()

    print(f"\n  mean invalid vectors per frame : {100 * bad.mean():.2f}% of the grid\n")
    print(f"  {'mask rule':<28} {'points':>8} {'% grid':>8} "
          f"{'wake energy kept':>18} {'test NMSE':>11}")
    for label, mask, interp in [
        ("strict any-frame (old)", frac_bad == 0, False),
        ("mask_tol 0.02 + interp", frac_bad <= 0.02, True),
        ("keep all, zero-fill", np.ones_like(frac_bad, bool), False),
        ("keep all + interp", np.ones_like(frac_bad, bool), True),
    ]:
        v = score(Fn, S, mask, t, interp)
        kept = (w[mask.ravel()] ** 2).sum() / (w ** 2).sum()
        print(f"  {label:<28} {mask.sum():>8d} {100 * mask.mean():>7.1f}% "
              f"{kept:>18.3f} {v:>11.4f}")

    print("\n  The strict rule keeps most of the grid and almost none of the wake,")
    print("  and the reconstruction goes from working to worse than predicting")
    print("  the mean. Zero-filling is slightly worse than interpolating, which")
    print("  is the difference the old code optimised while the mask rule cost")
    print("  an order of magnitude more.")

    if args.real:
        n_pix, body, per_frame, nt, kept = 15741, 254, 112, 6085, 5821
        print(f"\n  April experiment, 4p5d_10ms_yaw_0_0_0 (from the RDS survey):")
        print(f"    grid                       {n_pix}")
        print(f"    solid body                 {body} ({100 * body / n_pix:.1f}%)")
        print(f"    invalid vectors per frame  {per_frame} ({100 * per_frame / n_pix:.1f}%)")
        print(f"    kept by the strict rule    {kept} ({100 * kept / n_pix:.1f}%)")
        dropped = n_pix - kept - body
        print(f"    dropped and NOT body       {dropped}")
        print(f"    ...those points share {per_frame} invalid vectors per frame,")
        print(f"       so the average one is valid in "
              f"{100 * (1 - per_frame / dropped):.1f}% of frames and is discarded anyway.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
