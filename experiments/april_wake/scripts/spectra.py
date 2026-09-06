#!/usr/bin/env python
"""
spectra.py
==========

The frequency-domain view of the sparse-sensor problem: what the load cells
carry, what the flow carries, and at which frequencies the two are linearly
related.

    qsub experiments/april_wake/hpc/spectra.pbs
    python experiments/april_wake/scripts/spectra.py --run 4p5d_10ms_yaw_0_0_0

`diagnose_sensors.py` answers "how much of the field can twelve load cells
explain" with one broadband number, and that number (NMSE ~0.83) is what has
stalled the study. One number cannot distinguish the two situations that
produce it:

  * the sensors are uninformative at every frequency, or
  * the sensors are informative in a narrow band and blind everywhere else,
    and the broadband average dilutes a real signal into nothing.

Those call for opposite next steps -- give up on the sensor set, or restrict the
claim to the band it supports -- so the study cannot move until they are told
apart. Coherence tells them apart, frequency by frequency.

Seven sections:

  1  force spectra, 2500 Hz     the true rig resonances, before decimation
  2  force spectra, PIV rate    what survives the block average, and what
                                aliased into it on the way
  3  field spectra              POD coefficient PSDs: where the flow's energy is
  4  coherence                  gamma^2(f), each channel against each mode
  5  multiple coherence         all twelve channels jointly, per mode. This is
                                per-mode observability resolved in frequency,
                                and its energy-weighted integral is the same
                                linear ceiling diagnose_sensors.py estimates
                                parametrically -- computed a different way, so
                                the two disagreeing is informative
  6  band ceiling               best-possible NMSE against bandwidth: the plot
                                that says whether a band-limited claim is worth
                                making
  7  lag                        cross-correlation and cross-spectral group
                                delay, an estimate of the PIV/force offset that
                                does not depend on refitting an estimator

Everything is computed on the training split. Coherence is a fitted quantity
and a coherence estimated on the test set is not a prediction of anything.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

# this file is experiments/april_wake/scripts/<name>.py, so three levels
# up is the repo root -- which is what makes `experiments` importable.
# `src` needs no insert: the editable install puts it on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.april_wake.data_preprocessing import add_data_args, load_data, make_split  # noqa: E402
from experiments.april_wake.case_reader import (  # noqa: E402
    F_FORCE_HZ,
    F_PIV_HZ,
    RUNS,
    apply_drift_correction,
    baseline_drift_fit,
    find_force_files,
    read_dat,
    yaw_from_run,
)
from field_estimation.epod import pod  # noqa: E402

# Channel names, in the order read_dat returns them: two six-component
# balances. Used for labels only -- nothing keys off them.
CH = [f"{d}{c}" for d in ("d2", "d3") for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]

D_DISC = 0.0495  # porous disc diameter [m]
U_INF = 10.0  # tunnel speed for the 10ms runs [m/s]


def rule(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}", flush=True)


def ch_name(i, n):
    return CH[i] if i < len(CH) and n <= len(CH) else f"ch{i}"


# ── 1. the raw force record ───────────────────────────────────────────────────


def raw_force(run, root, drift_correct=True):
    """The 2500 Hz record for `run`, drift-corrected, before any resampling.

    `build_case` returns only the PIV-rate record, because that is what the
    estimators consume. The full-rate record is what says whether a resonance
    sits above the PIV Nyquist -- and a resonance above Nyquist does not go
    away when you decimate, it folds down onto the band you are fitting in.
    """
    recs = find_force_files(root)
    yaw = yaw_from_run(run)
    match = [r for r in recs if r.is_sync and r.yaw_key == yaw]
    if not match:
        raise FileNotFoundError(f"no sync force file at yaw {yaw}")
    rec = match[0]
    F = read_dat(rec.path)

    if drift_correct:
        base = [r for r in recs if r.is_baseline and r.yaw_key == yaw]
        if len(base) >= 2:
            F = apply_drift_correction(F, rec.seconds, baseline_drift_fit(base)[yaw])
        elif len(base) == 1:
            F = F - np.nanmean(read_dat(base[0].path), axis=1)[:, None]
    return F, os.path.basename(rec.path)


def section_raw_spectrum(F, fname, out, args):
    rule("1. force spectra at the acquisition rate (2500 Hz)")
    st_f = args.strouhal * U_INF / D_DISC
    nper = min(args.nperseg_force, F.shape[1])
    f, P = signal.welch(F - F.mean(1, keepdims=True), fs=F_FORCE_HZ, nperseg=nper, noverlap=nper // 2, axis=1)

    nyq = F_PIV_HZ / 2
    print(f"  file               : {fname}")
    print(f"  record             : {F.shape[1]} samples ({F.shape[1] / F_FORCE_HZ:.1f} s)")
    print(f"  resolution         : {f[1] - f[0]:.2f} Hz ({2 * F.shape[1] // nper - 1} Welch segments)")
    print(f"  expected shedding  : {st_f:.1f} Hz (St={args.strouhal})")
    print(f"  PIV Nyquist        : {nyq:.0f} Hz -- everything above this folds down when the record is decimated\n")

    band = f > 1.0
    above = f > nyq
    print(f"  {'ch':>6} {'peak Hz':>9} {'peak/med':>9} {'% power >Nyq':>13}   note")
    rows = []
    for c in range(F.shape[0]):
        pk = f[band][np.argmax(P[c][band])]
        ratio = P[c][band].max() / np.median(P[c][band])
        frac = P[c][above].sum() / P[c][band].sum()
        note = (
            "at the shedding frequency"
            if abs(pk - st_f) < 5
            else "sharp peak, structural"
            if ratio > 50
            else "broadband"
        )
        if frac > 0.5:
            note += "; MOST of its power aliases"
        print(f"  {ch_name(c, F.shape[0]):>6} {pk:>9.1f} {ratio:>9.0f} {100 * frac:>12.1f}%   {note}")
        rows.append((pk, ratio, frac))

    fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True, sharey=True)
    for c, ax in enumerate(axes.ravel()):
        if c >= F.shape[0]:
            ax.axis("off")
            continue
        ax.loglog(f[1:], P[c][1:], lw=0.8, c="C0")
        ax.axvline(nyq, c="r", ls="-", lw=1.0)
        ax.axvline(st_f, c="k", ls="--", lw=1.0)
        ax.set_title(f"{ch_name(c, F.shape[0])}  peak {rows[c][0]:.0f} Hz", fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("PSD")
    fig.suptitle(
        f"force channels at {F_FORCE_HZ:.0f} Hz   (red = PIV Nyquist {nyq:.0f} Hz, dashed = St={args.strouhal})"
    )
    fig.supxlabel("frequency [Hz]")
    fig.tight_layout()
    _save(fig, out, "force_spectra_raw.png")
    return f, P


# ── 2 & 3. what the estimator actually sees ───────────────────────────────────


def section_piv_rate(S, B, energy, out, args):
    rule("2 & 3. force and field spectra at the PIV rate (250 Hz)")
    st_f = args.strouhal * U_INF / D_DISC
    nper = min(args.nperseg, S.shape[1])
    kw = dict(fs=F_PIV_HZ, nperseg=nper, noverlap=nper // 2, axis=1)
    f, Ps = signal.welch(S - S.mean(1, keepdims=True), **kw)
    _, Pb = signal.welch(B - B.mean(1, keepdims=True), **kw)

    n_seg = 2 * S.shape[1] // nper - 1
    print(f"  resolution         : {f[1] - f[0]:.2f} Hz, {n_seg} Welch segments")
    print(f"  shedding           : {st_f:.1f} Hz     meandering (St~0.075): {0.075 * U_INF / D_DISC:.1f} Hz\n")

    print(f"  {'mode':>5} {'energy':>8} {'peak Hz':>9} {'St':>7}   half the mode's power below")
    for k in range(min(B.shape[0], args.show_modes)):
        p = Pb[k][1:]
        pk = f[1:][np.argmax(p)]
        cum = np.cumsum(p) / p.sum()
        f50 = f[1:][np.searchsorted(cum, 0.5)]
        print(f"  {k + 1:>5} {energy[k]:>8.4f} {pk:>9.2f} {pk * D_DISC / U_INF:>7.3f}   {f50:.1f} Hz")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for c in range(S.shape[0]):
        a1.loglog(f[1:], Ps[c][1:], lw=0.8, alpha=0.8, label=ch_name(c, S.shape[0]))
    a1.set(title="force channels, PIV rate", xlabel="frequency [Hz]", ylabel="PSD")
    a1.legend(fontsize=6, ncol=2)
    for k in range(min(B.shape[0], 8)):
        a2.loglog(f[1:], Pb[k][1:], lw=1.0, label=f"mode {k + 1}")
    a2.set(title="POD coefficients", xlabel="frequency [Hz]", ylabel="PSD")
    a2.legend(fontsize=7, ncol=2)
    for ax in (a1, a2):
        ax.axvline(st_f, c="k", ls="--", lw=1.0)
        ax.axvline(0.075 * U_INF / D_DISC, c="g", ls=":", lw=1.2)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, out, "spectra_piv_rate.png")
    return f, Ps, Pb


# ── 4 & 5. coherence ──────────────────────────────────────────────────────────


def section_coherence(S, B, out, args):
    """gamma^2(f) per channel-mode pair, and the joint multiple coherence.

    Ordinary coherence says whether one channel tracks one mode. Multiple
    coherence is the quantity that matters here: it is the fraction of a mode's
    power at frequency f that *any* linear combination of all twelve channels
    can explain, which is exactly what PODLSE is allowed to do. Integrating it
    against the mode's own spectrum, and summing over modes weighted by energy,
    reproduces the linear NMSE ceiling without fitting anything.

    Both estimators are biased upward, and badly so with twelve inputs and a few
    dozen Welch segments: E[gamma^2] ~ gamma^2 + (1 - gamma^2) p / n_seg for p
    inputs. The correction below is standard (Bendat & Piersol); what matters is
    that the uncorrected numbers are not quotable, and the correction is only
    trustworthy while n_seg is several times p.
    """
    rule("4 & 5. coherence: which frequencies the sensors can see")
    nper = min(args.nperseg, S.shape[1])
    n_seg = 2 * S.shape[1] // nper - 1
    p = S.shape[0]

    print(f"  segments (n_seg)   : {n_seg} of {nper} samples ({f'{F_PIV_HZ / nper:.2f}'} Hz resolution)")
    print(f"  inputs (p)         : {p}")
    print(f"  bias on a null estimate: {p / n_seg:.3f} multiple, {1 / n_seg:.3f} ordinary")
    if n_seg < 3 * p:
        print(f"  !! n_seg < 3p. The bias correction is doing too much work to be")
        print(f"     trusted at face value -- rerun with --nperseg {nper // 2}.")

    kw = dict(fs=F_PIV_HZ, nperseg=nper, noverlap=nper // 2)
    Sc = S - S.mean(1, keepdims=True)
    Bc = B - B.mean(1, keepdims=True)
    n_mode = min(B.shape[0], args.n_modes_coh)

    # cross-spectral matrix of the inputs, once: (n_f, p, p)
    f = signal.csd(Sc[0], Sc[0], **kw)[0]
    Sss = np.empty((len(f), p, p), complex)
    for i in range(p):
        for j in range(i, p):
            c = signal.csd(Sc[i], Sc[j], **kw)[1]
            Sss[:, i, j] = c
            Sss[:, j, i] = np.conj(c)

    ord_coh = np.empty((p, n_mode, len(f)))
    mult_coh = np.empty((n_mode, len(f)))
    Sbb = np.empty((n_mode, len(f)))

    # Tikhonov on the input cross-spectral matrix. Twelve channels off two
    # balances on one rig are not independent, so Sss is near-singular at some
    # frequencies, and an unregularised solve there returns a coherence of 1
    # that is pure numerical noise.
    tr = np.einsum("fii->f", Sss).real / p
    reg = args.csd_ridge * tr[:, None, None] * np.eye(p)[None]

    for k in range(n_mode):
        Sbb[k] = signal.welch(Bc[k], **kw)[1]
        Ssb = np.empty((len(f), p), complex)
        for i in range(p):
            Ssb[:, i] = signal.csd(Sc[i], Bc[k], **kw)[1]
            ord_coh[i, k] = signal.coherence(Sc[i], Bc[k], **kw)[1]
        x = np.linalg.solve(Sss + reg, Ssb[..., None])[..., 0]
        num = np.einsum("fi,fi->f", np.conj(Ssb), x).real
        mult_coh[k] = np.clip(num / np.where(Sbb[k] > 0, Sbb[k], 1), 0, 1)

    def debias(g2, n_in):
        return np.clip(1.0 - (1.0 - g2) * n_seg / max(n_seg - n_in, 1), 0.0, 1.0)

    mult_c = debias(mult_coh, p)
    ord_c = debias(ord_coh, 1)

    # power-weighted average coherence per mode: the fraction of that mode's
    # variance a linear map from the sensors can explain, band by band summed
    w = Sbb / np.where(Sbb.sum(1, keepdims=True) > 0, Sbb.sum(1, keepdims=True), 1)
    per_mode = (mult_c * w).sum(1)

    print(f"\n  {'mode':>5} {'gamma2 peak':>12} {'at Hz':>8} {'power-weighted':>15}   band where gamma2 > 0.2")
    for k in range(n_mode):
        i = int(np.argmax(mult_c[k]))
        hot = f[mult_c[k] > 0.2]
        rng = f"{hot.min():.1f}-{hot.max():.1f} Hz" if hot.size else "(none)"
        print(f"  {k + 1:>5} {mult_c[k][i]:>12.3f} {f[i]:>8.1f} {per_mode[k]:>15.3f}   {rng}")

    best = np.unravel_index(np.argmax(ord_c), ord_c.shape)
    print(
        f"\n  best single channel  : {ch_name(best[0], p)} against mode "
        f"{best[1] + 1}, gamma2 {ord_c[best]:.3f} at {f[best[2]]:.1f} Hz"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    for k in range(min(n_mode, 6)):
        ax.semilogx(f[1:], mult_c[k][1:], lw=1.2, label=f"mode {k + 1}")
    ax.axhline(p / n_seg, c="r", ls=":", lw=1.2, label="chance (uncorrected)")
    ax.set(
        title="multiple coherence: all 12 channels vs each mode",
        xlabel="frequency [Hz]",
        ylabel=r"$\gamma^2$",
        ylim=(0, 1),
    )
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    im = ax.pcolormesh(
        f, np.arange(1, n_mode + 1), mult_c, shading="nearest", cmap="magma", vmin=0, vmax=min(1, mult_c.max() * 1.05)
    )
    ax.set(title="multiple coherence", xlabel="frequency [Hz]", ylabel="POD mode", xscale="log")
    fig.colorbar(im, ax=ax, label=r"$\gamma^2$")

    ax = axes[1, 0]
    im = ax.pcolormesh(f, np.arange(p), ord_c[:, 0, :], shading="nearest", cmap="magma", vmin=0)
    ax.set(title="ordinary coherence, each channel vs mode 1", xlabel="frequency [Hz]", ylabel="channel", xscale="log")
    ax.set_yticks(np.arange(p))
    ax.set_yticklabels([ch_name(c, p) for c in range(p)], fontsize=7)
    fig.colorbar(im, ax=ax, label=r"$\gamma^2$")

    ax = axes[1, 1]
    for k in range(min(n_mode, 6)):
        ax.semilogx(f[1:], Sbb[k][1:] / Sbb[k][1:].max(), lw=1.0, label=f"mode {k + 1}")
    ax.set(
        title="mode PSD (normalised) -- where the energy the coherence\nhas to explain actually is",
        xlabel="frequency [Hz]",
        ylabel="PSD / max",
    )
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    for a in axes.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "coherence.png")
    return f, mult_c, Sbb, ord_c, per_mode


# ── 6. the band ceiling ───────────────────────────────────────────────────────


def section_band_ceiling(f, mult_c, Sbb, energy, observed, out):
    """Best-possible NMSE against the bandwidth you are willing to claim.

    The broadband ceiling is one point on this curve -- the right-hand end. If
    the curve is flat, the sensors really are uninformative and the sensible
    outcome of the project is that measurement, stated cleanly. If it drops
    sharply at low frequency, then a low-pass target is a reconstruction the
    sensors can actually support, and reporting it is not moving the goalposts
    as long as the filter is stated: "the sensor-observable component of the
    wake" is a different and defensible claim from "the wake".
    """
    rule("6. the ceiling as a function of bandwidth")
    n_mode = mult_c.shape[0]

    # Weight each analysed mode by its share of the WHOLE field's energy, not by
    # its share of the analysed modes'. Normalising within the analysed set
    # silently asserts that the modes beyond it do not exist, and here they
    # carry ~35% of the resolved energy and are pure noise -- which turns a
    # ceiling of 0.60 into one of 0.38. The unanalysed remainder is added back
    # below as energy no linear map explains, which is what the per-mode rho^2
    # in diagnose_sensors.py says it is.
    E = np.asarray(energy, float)[:n_mode]
    tail = max(float(1.0 - E.sum()), 0.0)

    # Sbb[k] integrates to var(b_k), which is already proportional to E_k, so
    # weighting the raw PSD by E_k counts the energy twice and produces a
    # ceiling below the tail it just declared unexplained. Normalise each mode's
    # spectrum to a distribution over frequency first; E_k then carries the
    # energy exactly once.
    tot_k = Sbb.sum(axis=1, keepdims=True)
    P = Sbb / np.where(tot_k > 0, tot_k, 1.0)  # each row sums to 1

    unexp = np.cumsum(P * (1 - mult_c) * E[:, None], axis=1).sum(0) + tail
    total = np.cumsum(P * E[:, None], axis=1).sum(0) + tail
    nmse_band = unexp / np.where(total > 0, total, 1)

    # what fraction of the field's total energy sits below each cutoff. The
    # tail is in neither numerator nor denominator here: it is energy the
    # analysed modes do not describe at any frequency.
    covered = (np.cumsum(P * E[:, None], axis=1).sum(0)) / max(E.sum(), 1e-30)

    print(f"  {'cutoff Hz':>10} {'field energy kept':>19} {'NMSE floor in band':>20}")
    for fc in [2, 5, 10, 15, 20, 25, 30, 40, 60, 90, 125]:
        i = int(np.searchsorted(f, fc))
        if i >= len(f):
            continue
        print(f"  {f[i]:>10.1f} {100 * covered[i]:>18.1f}% {nmse_band[i]:>20.3f}")
    print(f"\n  analysed modes 1..{n_mode} carry           : {E.sum():.3f} of the energy")
    print(f"  the remaining modes                : {tail:.3f}, counted as unexplained")
    print(f"  broadband ceiling (this method)   : {nmse_band[-1]:.4f}")
    print("  NOTE: coherence is fitted and evaluated on the SAME split, so this")
    print("  is an in-sample bound. The out-of-sample equivalent is the")
    print("  energy-weighted rho2_test in diagnose_sensors.py, which is higher.")
    if observed is not None:
        print(f"  best NMSE the sweep actually got   : {observed:.4f}")
        gap = observed - nmse_band[-1]
        print(f"  gap                                : {gap:+.4f}")
        if gap > 0.05:
            print("\n  The estimator is leaving headroom the sensors do carry.")
            print("  That part is a modelling problem, not an information limit.")
        elif gap > -0.05:
            print("\n  The fitted estimator is at the non-parametric ceiling.")
            print("  Architecture cannot help; the sensors or the pairing can.")
        else:
            print("\n  The sweep beat this ceiling, which means one of the two is")
            print("  mis-estimated -- most likely the coherence bias correction.")
            print("  Rerun with a smaller --nperseg before quoting either.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.semilogx(f[1:], nmse_band[1:], lw=1.6, c="C3")
    if observed is not None:
        a1.axhline(observed, c="k", ls="--", lw=1.2, label="best sweep NMSE")
        a1.legend(fontsize=8)
    a1.set(
        title="best-possible NMSE for a target low-passed at $f_c$",
        xlabel="cutoff $f_c$ [Hz]",
        ylabel="NMSE floor",
        ylim=(0, 1),
    )
    a2.semilogx(f[1:], 100 * covered[1:], lw=1.6, c="C0")
    a2.set(title="fraction of resolved field energy below $f_c$", xlabel="cutoff $f_c$ [Hz]", ylabel="% of energy")
    for a in (a1, a2):
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, out, "band_ceiling.png")
    return nmse_band, covered


# ── 7. lag ────────────────────────────────────────────────────────────────────


def section_lag(S, B, f, mult_c, out, args):
    """Two estimates of the PIV/force offset that do not refit an estimator.

    The lag scan in `diagnose_sensors.py` refits PODLSE at every candidate lag,
    which is honest but slow and confounds the offset with the window length --
    its optimum landed at -25 samples with a 25-sample causal window, which is
    exactly where an unreachable offset and a resolved one look the same.

    Cross-correlation is direct. Cross-spectral group delay is better still:
    the phase of the cross-spectrum is linear in frequency with slope -2*pi*tau
    for a pure delay, so fitting that slope over the band where the coherence is
    high gives tau without sliding anything.
    """
    rule("7. PIV/force offset, measured two ways")
    Sc = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-30)
    Bc = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-30)
    n_t = S.shape[1]
    L = args.max_lag
    lags = np.arange(-L, L + 1)

    n_mode = min(B.shape[0], 3)
    xc = np.empty((S.shape[0], n_mode, len(lags)))
    for c in range(S.shape[0]):
        for k in range(n_mode):
            full = np.correlate(Bc[k], Sc[c], mode="full") / n_t
            mid = len(full) // 2
            xc[c, k] = full[mid - L : mid + L + 1]

    # xc[c, k, i] peaks where B[t] matches S[t - lags[i]]; the sign convention
    # matches --force-lag, so the argmax is the value to pass straight through.
    flat = np.abs(xc).max(axis=(0, 1))
    lag_xc = int(lags[np.argmax(flat)])
    c_best, k_best, i_best = np.unravel_index(np.argmax(np.abs(xc)), xc.shape)

    print(f"  cross-correlation, strongest pair : {ch_name(c_best, S.shape[0])} vs mode {k_best + 1}")
    print(
        f"    peak |r|        : {abs(xc[c_best, k_best, i_best]):.3f} at lag "
        f"{lags[i_best]:+d} ({1e3 * lags[i_best] / F_PIV_HZ:+.1f} ms)"
    )
    print(f"    argmax over all channels and modes: {lag_xc:+d} samples ({1e3 * lag_xc / F_PIV_HZ:+.1f} ms)")
    print(f"    |r| at lag 0    : {abs(xc[:, :, L]).max():.3f}")

    # group delay, over the frequencies where the joint coherence is worth using
    hot = mult_c[0] > args.coh_floor
    tau = np.nan
    if hot.sum() >= 4:
        kw = dict(fs=F_PIV_HZ, nperseg=min(args.nperseg, n_t), noverlap=min(args.nperseg, n_t) // 2)
        ph = np.unwrap(np.angle(signal.csd(Sc[c_best], Bc[k_best], **kw)[1]))
        A = np.vstack([f[hot], np.ones(hot.sum())]).T
        slope = np.linalg.lstsq(A, ph[hot], rcond=None)[0][0]
        tau = -slope / (2 * np.pi)
        print(f"\n  cross-spectral group delay ({hot.sum()} bins with gamma2 > {args.coh_floor}):")
        print(f"    tau             : {1e3 * tau:+.1f} ms ({tau * F_PIV_HZ:+.1f} PIV samples)")
    else:
        print(f"\n  group delay: skipped, fewer than 4 bins above gamma2 = {args.coh_floor}")

    print(
        f"\n  A convection time over one disc diameter is "
        f"{1e3 * D_DISC / U_INF:.1f} ms; the PIV window is several diameters"
    )
    print("  downstream, so a delay of a few ms is physics and a delay of tens")
    print("  of ms is an acquisition offset. Pass the result as --force-lag.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for c in range(S.shape[0]):
        a1.plot(1e3 * lags / F_PIV_HZ, xc[c, 0], lw=0.8, alpha=0.75, label=ch_name(c, S.shape[0]))
    a1.axvline(0, c="k", lw=1.0)
    a1.axvline(1e3 * lag_xc / F_PIV_HZ, c="r", ls="--", lw=1.2, label=f"argmax {lag_xc:+d}")
    a1.set(title="cross-correlation with POD mode 1", xlabel="lag [ms]", ylabel="r")
    a1.legend(fontsize=6, ncol=2)
    a2.plot(1e3 * lags / F_PIV_HZ, flat, lw=1.4, c="C3")
    a2.axvline(1e3 * lag_xc / F_PIV_HZ, c="r", ls="--", lw=1.2)
    a2.set(title="max |r| over all channels and the leading 3 modes", xlabel="lag [ms]", ylabel="max |r|")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "lag.png")
    return lag_xc, tau


# ── where the sensors can see, in space ───────────────────────────────────────


def section_spatial(Psi, mult_c, Sbb, unflat, out):
    """Field energy, and the part of it the sensors explain, on the grid.

    Reconstructed from the POD basis rather than by transforming the field
    directly: mode k contributes Psi_k^2 times its own power everywhere, so the
    sensor-explainable map is the same sum weighted by that mode's
    power-weighted coherence. It is cheap and it answers the question a
    reviewer asks first -- not "how much", but "where".
    """
    rule("where in the field the sensors can see")
    n_mode = mult_c.shape[0]
    w = Sbb.sum(1)
    expl = (Sbb * mult_c).sum(1)
    tot_map = (Psi[:, :n_mode] ** 2) @ w
    obs_map = (Psi[:, :n_mode] ** 2) @ expl

    frac = obs_map.sum() / tot_map.sum()
    print(f"  fraction of the leading-{n_mode} energy the sensors explain: {frac:.3f}")

    G = unflat(np.stack([tot_map, obs_map, obs_map / np.where(tot_map > 0, tot_map, 1)], 1))
    titles = ["field energy (modes 1..%d)" % n_mode, "sensor-explainable energy", "ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for j in range(3):
        for u in range(min(2, G.shape[0])):
            ax = axes[u, j]
            im = ax.pcolormesh(G[u, j].T, cmap="viridis" if j < 2 else "magma", shading="auto")
            ax.set_title(f"{titles[j]}  ({'u' if u == 0 else 'v'})", fontsize=9)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    _save(fig, out, "spatial_observability.png")
    return frac


# ── plumbing ──────────────────────────────────────────────────────────────────


def _save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    # make_split reads these; spectra.py does not sweep them
    p.add_argument("--delays", type=int, nargs="+", default=[1])
    p.add_argument("--delay-stride", type=int, default=1)

    s = p.add_argument_group("spectra")
    s.add_argument("--r-field", type=int, default=16, help="POD modes to analyse; the coherence of mode 40 is noise")
    s.add_argument("--n-modes-coh", type=int, default=8)
    s.add_argument("--show-modes", type=int, default=8)
    s.add_argument(
        "--nperseg",
        type=int,
        default=256,
        help="Welch segment at the PIV rate. Shorter means more "
        "segments and less coherence bias, at coarser "
        "resolution -- the trade that decides whether the "
        "multiple coherence is quotable at all",
    )
    s.add_argument("--nperseg-force", type=int, default=8192, help="Welch segment at 2500 Hz")
    s.add_argument("--csd-ridge", type=float, default=1e-6, help="Tikhonov on the input cross-spectral matrix")
    s.add_argument(
        "--coh-floor", type=float, default=0.2, help="coherence above which a bin is used for the group delay"
    )
    s.add_argument("--pod-method", default="randomized", choices=["svd", "snapshot", "randomized", "auto"])
    s.add_argument("--strouhal", type=float, default=0.17)
    s.add_argument("--max-lag", type=int, default=80, help="PIV samples")
    s.add_argument(
        "--observed", type=float, default=0.8281, help="best NMSE from the sweep, for the ceiling comparison"
    )
    s.add_argument("--skip", nargs="*", default=[], choices=["raw", "piv", "coh", "band", "lag", "spatial"])
    s.add_argument("--out", default="results/spectra")
    args = p.parse_args()

    out = os.path.join(args.out, args.run)
    os.makedirs(out, exist_ok=True)

    rule("0. data")
    Q, S, unflat, case0, run_id, cases = load_data(args)
    tr, te = make_split(args, Q.shape[1], run_id, cases)[:2]

    Psi, Sigma, B, qm = pod(Q[:, tr], r=args.r_field, subtract_mean=True, method=args.pod_method)
    # Normalise by the TOTAL fluctuation energy, not by the truncated basis's.
    # `Sigma` holds only r_field singular values, so Sigma**2 / sum(Sigma**2)
    # sums to 1 over the modes computed and silently asserts the rest of the
    # field does not exist -- which turns an honest ceiling of ~0.67 into 0.38.
    Qc = Q[:, tr] - qm
    total_energy = float(np.einsum("ij,ij->", Qc, Qc))
    energy = Sigma[: args.r_field] ** 2 / total_energy
    print(f"  POD r={args.r_field} on the training split, {100 * energy.sum():.1f}% of the total fluctuation energy")

    Str = S[:, tr]
    res = {}

    if "raw" not in args.skip:
        F, fname = raw_force(args.run, os.environ.get("RDS_ROOT"), drift_correct=not args.no_drift)
        section_raw_spectrum(F, fname, out, args)
    if "piv" not in args.skip:
        section_piv_rate(Str, B, energy, out, args)

    f = mult_c = Sbb = None
    if "coh" not in args.skip:
        f, mult_c, Sbb, ord_c, per_mode = section_coherence(Str, B, out, args)
        res.update(f=f, mult_coh=mult_c, mode_psd=Sbb, ord_coh=ord_c, per_mode=per_mode)
        if "band" not in args.skip:
            nmse_band, covered = section_band_ceiling(f, mult_c, Sbb, energy, args.observed, out)
            res.update(nmse_band=nmse_band, covered=covered)
        if "lag" not in args.skip:
            lag_xc, tau = section_lag(Str, B, f, mult_c, out, args)
            res.update(lag_xcorr=lag_xc, group_delay_s=tau)
        if "spatial" not in args.skip:
            res["explained_fraction"] = section_spatial(Psi, mult_c, Sbb, unflat, out)

    np.savez_compressed(os.path.join(out, "spectra.npz"), energy=energy, **res)
    rule("done")
    print(f"  plots and spectra.npz -> {out}/")
    if "lag" not in args.skip and "coh" not in args.skip:
        print(f"\n  next: rerun the baseline with --force-lag {res['lag_xcorr']:+d}")
        print("        and compare against the lag-0 number before anything else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
