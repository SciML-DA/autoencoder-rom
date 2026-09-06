#!/usr/bin/env python
"""
diagnose_sensors.py
===================

Why the reconstruction plateaus at NMSE ~0.83 no matter what you throw at it.

    qsub hpc/diagnose.pbs
    python scripts/diagnose_sensors.py --run 4p5d_10ms_yaw_0_0_0

The sweep produced a very specific signature: test NMSE is flat at 0.83-0.85
across latent size 4..128, across linear/MLP/CNN/GRU sensor branches, and across
POD and autoencoder latents -- while the projection floor falls from 0.62 to
0.17. Train NMSE is 0.73-0.79, barely better than test.

Nothing that varies changes the answer, and the model cannot fit its own
training data. That is not overfitting, not a bad basis and not too little
capacity. It is an information ceiling: the load cells do not carry the flow.
This script measures how high that ceiling is and rules the alternatives in or
out, one number at a time.

Six checks, each printing a number and what it means:

  1  mask audit          63% of the PIV field was dropped; the notes said 0.7%
  2  honest observability  per-mode rho^2, fitted on train, measured on test
  3  ceiling              what NMSE those rho^2 imply -- compare to the observed
  4  lag scan             is the PIV/force pairing actually aligned
  5  leak test            how much a random split flatters the same model
  6  force spectrum       is a rig resonance eating the sensor modes
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiments.april_wake.data_preprocessing import add_data_args, load_data, make_split  # noqa: E402
from experiments.april_wake.case_reader import F_FORCE_HZ, F_PIV_HZ, RUNS  # noqa: E402
from field_estimation.epod import PODLSE, delay_embed, nmse, pod  # noqa: E402


def rule(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}", flush=True)


# ── 1. mask ───────────────────────────────────────────────────────────────────


def check_mask(args, case0):
    """How much of the field is thrown away, and how much of that is justified.

    ``wake_experiment.build_case`` uses::

        fluid = ~np.isnan(X).any(axis=(0, 1))

    -- a point is kept only if it is valid in *every one* of the 6085 snapshots.
    That is the right rule for a solid body, which never moves, and the wrong
    one for spurious PIV vectors, which do. If each frame independently drops
    even 0.5% of its vectors, then after 6085 frames almost no point survives
    "valid everywhere", and the mask quietly eats the field.

    The run log reported 9920 of 15741 masked (63%) against ~112 (0.7%) for the
    body itself. This measures where the other 62% comes from and what a
    tolerant threshold would recover.
    """
    rule("1. mask audit")
    nu, nt, nx, ny = case0.X.shape
    n_pix = nx * ny

    # Read the dropout off case.invalid_frac, which build_case records BEFORE
    # anything masks the field. Reading it off case0.X instead measures this
    # script's own input: load_data does `c.X[:, :, ~c.fluid_mask] = np.nan`
    # before returning, so every frame then shows an identical NaN count, the
    # tolerance table comes out flat, and the audit reports "the mask is the
    # body and little else" -- which is how a 63% mask survived being audited.
    if case0.invalid_frac is None:
        print("  case has no invalid_frac -- rerun against a build_case that")
        print("  records it; this audit cannot be trusted without it.")
        return {}
    frac_bad = np.asarray(case0.invalid_frac).ravel()
    ever = frac_bad > 0
    always = frac_bad >= 1.0

    print(f"  grid                      : {nx} x {ny} = {n_pix} points, {nt} frames")
    print(f"  invalid in EVERY frame    : {always.sum():>6d}  ({always.mean():.1%})   <- the solid body")
    print(f"  invalid in ANY frame      : {ever.sum():>6d}  ({ever.mean():.1%})   <- what the pipeline drops")
    print(f"  mean per-frame invalid    : {frac_bad.mean():.2%} of the grid")
    print(f"  worst point               : invalid in {frac_bad.max():.1%} of frames\n")

    print("  if a point were dropped only when invalid in more than <tol> of frames:")
    print(f"  {'tol':>8} {'kept':>8} {'kept %':>9} {'vs current':>12}")
    keep_now = (~ever).sum()
    for tol in (0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.99):
        keep = int((frac_bad <= tol).sum())
        print(f"  {tol:>8.3f} {keep:>8d} {keep / n_pix:>8.1%} {keep / max(keep_now, 1):>11.1f}x")

    print()
    if ever.sum() > always.sum() + 0.05 * n_pix:
        extra = ever.sum() - always.sum()
        print(f"  !! {extra} points ({extra / n_pix:.1%}) are dropped for being invalid")
        print("     in at least one frame while not being body. The body is only")
        print(f"     {always.sum()} points. The reconstruction is being asked to")
        print("     work on a third of the field, and the discarded part includes")
        print("     wake -- which is where the sensor-correlated energy lives.")
        print()
        print("     Fix: --mask-tol 0.02 keeps a point unless it is invalid in")
        print("     more than 2% of frames and interpolates the rest, rather than")
        print("     discarding the point for all time.")
    else:
        print("  The mask is the body and little else; this is not the problem.")
    return dict(always=int(always.sum()), ever=int(ever.sum()), n_pix=n_pix)


# ── 2 & 3. observability and the ceiling ──────────────────────────────────────


def check_observability(Q, S, tr, te, args):
    """Per-mode rho^2 fitted on train and *measured on test*.

    The observability plot in the sweep is computed in-sample. With 12 channels
    x 25 delays = 300 regressors against 4365 training snapshots, a mode with no
    real sensor content still scores rho^2 ~ 300/4365 = 0.07 by chance, and the
    plot's tail sits at 0.10-0.18 -- barely above that. Refitting on train and
    scoring on test separates the modes that are genuinely observable from the
    ones that are fitting noise.
    """
    rule("2. honest observability (fit on train, measured on test)")
    r = args.r_field
    Psi, Sigma, B, qm = pod(Q[:, tr], r=r, subtract_mean=True, method=args.pod_method)
    B_te = Psi.T @ (Q[:, te] - qm)

    Sd = delay_embed(S, args.n_delays, 1, args.delay_ahead)
    Str, Ste = Sd[:, tr], Sd[:, te]
    mu, sd = Str.mean(1, keepdims=True), Str.std(1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    Ctr, Cte = (Str - mu) / sd, (Ste - mu) / sd

    # ridge-regularised least squares per mode, train -> test
    G = Ctr @ Ctr.T
    lam = args.ridge * np.trace(G) / G.shape[0]
    M = np.linalg.solve(G + lam * np.eye(G.shape[0]), Ctr @ B.T).T
    B_hat = M @ Cte

    var_tr = np.einsum("kt,kt->k", B, B) / B.shape[1]
    var_te = np.var(B_te, axis=1)
    resid = np.var(B_te - B_hat, axis=1)
    rho2_te = np.clip(1.0 - resid / np.where(var_te > 0, var_te, 1.0), -1, 1)

    # in-sample, for the comparison that makes the point
    resid_tr = np.var(B - M @ Ctr, axis=1)
    rho2_tr = np.clip(1.0 - resid_tr / np.where(var_tr > 0, var_tr, 1.0), -1, 1)

    # Normalise by the TOTAL fluctuation energy. `Sigma` holds only the r
    # computed singular values, so Sigma**2 / sum(Sigma**2) sums to 1 over the
    # modes computed and silently asserts the rest of the field does not exist.
    # Here that inflated every E_k by ~1.56x and the ceiling with it.
    Qc_tr = Q[:, tr] - qm
    energy = Sigma[:r] ** 2 / float(np.einsum("ij,ij->", Qc_tr, Qc_tr))
    print(f"  {'mode':>5} {'energy':>9} {'rho2 train':>12} {'rho2 test':>11}   verdict")
    for k in range(min(r, args.show_modes)):
        v = "observable" if rho2_te[k] > 0.15 else ("marginal" if rho2_te[k] > 0.05 else "noise")
        print(f"  {k + 1:>5} {energy[k]:>9.4f} {rho2_tr[k]:>12.3f} {rho2_te[k]:>11.3f}   {v}")

    n_obs = int((rho2_te > 0.15).sum())
    print(f"\n  modes with rho2_test > 0.15 : {n_obs} of {r}")
    print(f"  mean rho2_train (all modes) : {rho2_tr.mean():.3f}")
    print(f"  mean rho2_test  (all modes) : {rho2_te.mean():.3f}")
    print(f"  chance level (n_reg/n_train): {Ctr.shape[0] / len(tr):.3f}")
    return rho2_te, energy, Sigma, dict(n_obs=n_obs)


def check_ceiling(rho2_te, energy, observed):
    """What NMSE those rho^2 imply, against what the sweep actually got.

    NMSE_min = 1 - sum_k E_k rho2_k, with E_k the energy fraction of mode k.
    If this lands on the observed plateau, the models are already at the
    ceiling and no amount of architecture will move them -- the honest next step
    is to attack the sensors or the pairing, not the network.
    """
    rule("3. the ceiling")
    contrib = energy * np.clip(rho2_te, 0, 1)
    ceiling = 1.0 - contrib.sum()
    top3 = 1.0 - contrib[:3].sum()
    print(f"  modes analysed carry               : {energy.sum():.4f} of the total fluctuation energy")
    print(f"  energy-weighted explained fraction : {contrib.sum():.4f}")
    print(f"  implied best-possible test NMSE    : {ceiling:.4f}")
    print(f"  ...using only the top 3 modes      : {top3:.4f}")
    print(f"  best NMSE the sweep actually got   : {observed:.4f}")
    print()
    if abs(ceiling - observed) < 0.08:
        print("  These agree. Every model in the sweep is already at the linear")
        print("  information ceiling of this sensor set. Latent size, branch")
        print("  architecture and POD-vs-AE cannot matter, and the sweep showing")
        print("  them not mattering is the expected result, not a bug.")
    else:
        print(f"  Gap of {observed - ceiling:+.3f} between ceiling and achieved.")
        print("  There is headroom the models are not reaching -- that part IS")
        print("  a modelling problem.")
    return ceiling


# ── 4. lag ────────────────────────────────────────────────────────────────────


def check_lag(Q, S, tr, te, args):
    """Slide the force record against the PIV and find where the fit is best.

    ``force_index_for_pair`` derives the offset from the filename; Roman's
    notebook hardcodes 1000. If those disagree the correlation is shifted, and
    ``delay_embed`` only reaches *backwards*, so a force record that needs to
    move the other way is unreachable no matter how long the window.

    A minimum at lag 0 confirms the pairing. A minimum somewhere else is a bug
    worth several tenths of NMSE, and it is the cheapest fix available.
    """
    rule("4. lag scan (PIV/force alignment)")
    r = 8  # leading modes only: they carry the signal and the scan stays fast
    Psi, _, B, qm = pod(Q[:, tr], r=r, subtract_mean=True, method=args.pod_method)

    lags = np.arange(-args.max_lag, args.max_lag + 1, args.lag_step)
    out = []
    for lag in lags:
        Ss = np.roll(S, lag, axis=1)
        Sd = delay_embed(Ss, args.n_delays, 1, args.delay_ahead)
        m = PODLSE(r_field=r, r_sensor=None, ridge=args.ridge, pod_method=args.pod_method).fit(Q[:, tr], Sd[:, tr])
        out.append(nmse(Q[:, te], m.predict(Sd[:, te])))
    out = np.array(out)
    best = int(np.argmin(out))

    print(f"  {'lag':>6} {'ms':>8} {'test NMSE':>11}")
    for lg, v in zip(lags, out):
        mark = "  <- best" if lg == lags[best] else ""
        print(f"  {lg:>6d} {1e3 * lg / F_PIV_HZ:>8.1f} {v:>11.4f}{mark}")
    print(f"\n  best lag        : {lags[best]:+d} samples ({1e3 * lags[best] / F_PIV_HZ:+.1f} ms)")
    print(f"  NMSE at lag 0   : {out[np.where(lags == 0)[0][0]]:.4f}")
    print(f"  NMSE at best lag: {out[best]:.4f}")
    # Judge on effect size, not on argmin. With 1521 test snapshots the scan is
    # noisy at the third decimal, so the best lag is essentially never exactly 0
    # and a bare `if lags[best] != 0` cries wolf on a 0.4% improvement. A genuine
    # pairing error is not subtle -- misaligning a 250 Hz record by even a few
    # samples decorrelates it badly -- so require the gain to be worth acting on.
    at0 = out[np.where(lags == 0)[0][0]]
    gain = at0 - out[best]
    rel = gain / at0 if at0 > 0 else 0.0
    if lags[best] == 0 or rel < 0.02:
        print(
            f"\n  Pairing confirmed: shifting to the best lag buys "
            f"{gain:+.4f} NMSE ({rel:+.1%}), which is noise at this test size."
        )
        print("     The alignment is not the problem.")
    else:
        print(
            f"\n  !! The pairing is off by {lags[best]} PIV samples "
            f"({lags[best] * int(F_FORCE_HZ / F_PIV_HZ)} force samples), "
            f"worth {rel:.1%} NMSE."
        )
        print("     Fix force_index_for_pair before trusting any other number.")
    return lags, out


# ── 5. leak ───────────────────────────────────────────────────────────────────


def check_leak(Q, S, args, rng):
    """The same model on a contiguous split and on a random one.

    At 250 Hz consecutive snapshots are nearly identical, so a randomly held-out
    frame has near-copies of itself in the training set. This is the number to
    quote when comparing against a reference notebook: if the reference used a
    random split, the gap you are chasing may be entirely this.
    """
    rule("5. leak test (contiguous vs random split)")
    n_t = Q.shape[1]
    Sd = delay_embed(S, args.n_delays, 1, args.delay_ahead)
    w = args.n_delays - 1

    n_te = int(round(0.25 * n_t))
    tr_c = np.arange(w, n_t - n_te - args.gap)
    te_c = np.arange(n_t - n_te, n_t)

    perm = rng.permutation(np.arange(w, n_t))
    tr_r, te_r = np.sort(perm[n_te:]), np.sort(perm[:n_te])

    res = {}
    for name, (a, b) in (("contiguous", (tr_c, te_c)), ("random", (tr_r, te_r))):
        m = PODLSE(r_field=args.r_field, r_sensor=None, ridge=args.ridge, pod_method=args.pod_method)
        m.fit(Q[:, a], Sd[:, a])
        res[name] = nmse(Q[:, b], m.predict(Sd[:, b]))
        print(f"  {name:>11} split: test NMSE {res[name]:.4f}  ({len(a)} train / {len(b)} test)")

    print(
        f"\n  random split flatters the same model by "
        f"{res['contiguous'] - res['random']:+.4f} NMSE "
        f"({100 * (1 - res['random'] / res['contiguous']):.0f}% better)"
    )
    if res["random"] < 0.6 * res["contiguous"]:
        print("  That is a large leak. Any reference number obtained from a random")
        print("  split is not comparable to these -- check how the baseline split")
        print("  its data before concluding you are behind it.")
    return res


# ── 6. force spectrum ─────────────────────────────────────────────────────────


def check_force_spectrum(S, out_dir, args):
    """Power spectra of the raw channels, against the shedding frequency.

    The folder readme is explicit that no filtering was applied to remove
    high-energy structural frequencies. A sharp rig resonance will occupy the
    leading sensor POD modes, and the LSE will map rig vibration onto flow
    structures -- spending its limited regressors on something with no fluid
    content.
    """
    rule("6. force spectrum")
    n = S.shape[1]
    f = np.fft.rfftfreq(n, d=1.0 / F_PIV_HZ)
    Sc = S - S.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(Sc, axis=1)) ** 2 / n

    st_f = 0.17 * 10.0 / 0.0495  # St = 0.17, U = 10 m/s, D ~ 49.5 mm
    print(f"  expected shedding  : {st_f:.1f} Hz (St=0.17)")
    print(f"  PIV Nyquist        : {F_PIV_HZ / 2:.0f} Hz\n")
    print(f"  {'ch':>4} {'peak Hz':>9} {'peak/median':>12}   note")
    band = f > 1.0
    for c in range(S.shape[0]):
        pk = f[band][np.argmax(P[c][band])]
        ratio = P[c][band].max() / np.median(P[c][band])
        note = ""
        if ratio > 50 and abs(pk - st_f) > 10:
            note = "sharp peak, not shedding -- likely rig resonance"
        elif abs(pk - st_f) < 10:
            note = "at the shedding frequency"
        print(f"  {c:>4} {pk:>9.1f} {ratio:>12.0f}   {note}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for c in range(S.shape[0]):
        ax.loglog(f[1:], P[c][1:], lw=0.8, alpha=0.75, label=f"ch {c}")
    ax.axvline(st_f, c="k", ls="--", lw=1.2)
    ax.annotate(f"St=0.17 -> {st_f:.0f} Hz", (st_f, ax.get_ylim()[1]), fontsize=8, rotation=90, ha="right", va="top")
    ax.set(xlabel="frequency [Hz]", ylabel="PSD", title="force channels, PIV-rate")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(out_dir, "force_spectrum.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {p}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--r-field", type=int, default=64)
    p.add_argument("--n-delays", type=int, default=25)
    # make_split sizes its warm-up from these two, so they must exist even
    # though this script never sweeps them
    p.add_argument("--delay-stride", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument(
        "--observed", type=float, default=0.8281, help="best test NMSE from the sweep, for the ceiling comparison"
    )
    p.add_argument(
        "--pod-method",
        default="randomized",
        choices=["svd", "snapshot", "randomized", "auto"],
        help="POD algorithm. The lag scan refits once per candidate "
        "lag, and `pod` defaults to a full dense SVD -- which "
        "at the real problem size is 50 s a fit, so 161 lags is "
        "2.2 h and job 3936955 was killed at the walltime doing "
        "exactly that. Randomized is 1.3 s for the same leading "
        "modes and is already what the study and sweep use.",
    )
    p.add_argument("--max-lag", type=int, default=40)
    p.add_argument("--lag-step", type=int, default=5)
    p.add_argument("--show-modes", type=int, default=20)
    p.add_argument("--skip", nargs="*", default=[], choices=["mask", "obs", "lag", "leak", "spectrum"])
    p.add_argument("--out", default="results/diagnosis")
    args = p.parse_args()
    args.delays = [args.n_delays]
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    rule("0. data")
    Q, S, unflat, case0, run_id, cases = load_data(args)
    tr, te = make_split(args, Q.shape[1], run_id, cases)[:2]

    if "mask" not in args.skip:
        check_mask(args, case0)
    if "obs" not in args.skip:
        rho2, energy, Sigma, _ = check_observability(Q, S, tr, te, args)
        check_ceiling(rho2, energy, args.observed)
    if "lag" not in args.skip:
        check_lag(Q, S, tr, te, args)
    if "leak" not in args.skip:
        check_leak(Q, S, args, rng)
    if "spectrum" not in args.skip:
        check_force_spectrum(S, args.out, args)

    rule("done")
    print(f"  plots -> {args.out}/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
