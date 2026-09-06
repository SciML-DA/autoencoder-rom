#!/usr/bin/env python
"""
verify_reconstruction.py
========================

Verification suite for the sparse-sensor reconstruction stack: ``field_estimation/epod.py``
(POD, extended POD, POD-LSE), ``field_estimation/branched_ae.py`` (the two-branch
autoencoder and the latent forecaster), and the ``Case`` machinery in
``datasets/wake_experiment.py``.

Self-contained: no RDS, no network, no ``wake_synthetic``. The field is an
exactly rank-r expansion built here in forty lines, which turns "did my POD
work?" into a machine-precision assertion instead of a judgement call, and the
data-layer checks run against an RDS-shaped directory written into a temp dir.

    python experiments/april_wake/scripts/verify_reconstruction.py           # all checks
    python experiments/april_wake/scripts/verify_reconstruction.py -k epod   # only matching names
    python experiments/april_wake/scripts/verify_reconstruction.py -v        # show every number
    python experiments/april_wake/scripts/verify_reconstruction.py --slow    # + the end-to-end study run

The checks that matter most, in order
-------------------------------------
``lse_exact_recovery``      With linear sensors and no noise POD-LSE is not an
                            approximation, it is the exact solution. If this is
                            above 1e-10 the bug is in the linear algebra and no
                            hyperparameter will fix it.
``pod_recovers_known_rank`` A transposed data matrix decomposes happily and
                            reconstructs happily -- it just silently has the
                            wrong rank. Cheapest guard against the most common
                            bug in the subject.
``no_leakage``              A contiguous split must score *worse* than a random
                            one. If they match, the random split is holding
                            nothing out and every number from it is meaningless.
``delay_embed_raises_rank`` The claim the whole delay machinery rests on: a
                            linear map from n instantaneous channels reaches an
                            n-dimensional subspace, and lags raise it.
``epod_equals_podlse``      Extended POD and POD-LSE with an untruncated field
                            basis are the same estimator by two routes. If they
                            disagree, one of the two is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

# this file is experiments/april_wake/scripts/<name>.py, so three levels
# up is the repo root -- which is what makes `experiments` importable.
# `src` needs no insert: the editable install puts it on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.april_wake import case_reader as we  # noqa: E402
from field_estimation.branched_ae import (  # noqa: E402
    BranchedAE,
    LatentForecaster,
    LinearLatent,
    TorchLatent,
    sensor_windows,
)
from field_estimation.epod import (  # noqa: E402
    PODLSE,
    ExtendedPOD,
    blocked_folds,
    delay_embed,
    extended_pod,
    lse_map,
    mode_observability,
    nmse,
    pod,
    projection_floor,
    split_train_test,
)

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
    return ok


# ── the fixture ───────────────────────────────────────────────────────────────


def toy(
    n_t=1200,
    n_x=24,
    n_y=16,
    rank=6,
    n_sensors=12,
    f_sample=250.0,
    response="linear",
    snr_db=float("inf"),
    seed=0,
    n_mask=9,
):
    """An exactly rank-``r`` velocity field plus synchronised load cells.

    The field is

        u(x, y, t) = u_bar(x, y) + sum_k a_k(t) psi_k(x, y)

    with orthonormalised random spatial modes and narrowband oscillators at
    distinct frequencies. Because it is built at a known rank, the POD spectrum
    has exactly ``rank`` non-zero singular values and every rank-based check
    below is an equality, not a tolerance judgement.

    ``response="quadratic"`` makes the loads go as the square of the modal
    amplitudes, which is what a load cell actually measures -- force is a
    quadratic functional of velocity. That puts an irreducible floor under any
    *linear* estimator, and the gap between the linear floor and zero is exactly
    the headroom the two-branch autoencoder is trying to claim.

    float64 throughout: quantising to float32 leaves singular values around
    1e-4, which swamps the 1e-10 an exact reconstruction should reach, and the
    check would then be measuring the storage format.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) / f_sample

    fluid = np.ones((n_x, n_y), bool)
    if n_mask:  # a solid body, as disc 2 is in the real field of view
        cx, cy = n_x // 3, n_y // 2
        w = int(round(np.sqrt(n_mask) / 2))
        fluid[cx - w : cx + w + 1, cy - w : cy + w + 1] = False
    n_f = int(fluid.sum())

    Psi, _ = np.linalg.qr(rng.standard_normal((2 * n_f, rank)))
    freqs = 3.0 + 2.7 * np.arange(rank)  # distinct, all well under Nyquist
    A = np.stack(
        [
            np.sin(2 * np.pi * f * t + p) * amp
            for f, p, amp in zip(freqs, rng.uniform(0, 2 * np.pi, rank), np.linspace(1.0, 0.25, rank))
        ]
    )
    q_mean = rng.standard_normal((2 * n_f, 1)) * 2.0 + 10.0
    Q = Psi @ A + q_mean  # (2*n_f, n_t), the repo's masked flat layout

    M = rng.standard_normal((n_sensors, rank))
    S = M @ A
    if response == "quadratic":
        S = S + 0.5 * (M @ (A**2))
    elif response != "linear":
        raise ValueError(response)
    if np.isfinite(snr_db):
        p_sig = np.mean(S**2, axis=1, keepdims=True)
        S = S + rng.standard_normal(S.shape) * np.sqrt(p_sig / 10 ** (snr_db / 10))

    return dict(
        Q=Q, S=S, A=A, M=M, Psi=Psi, q_mean=q_mean, fluid=fluid, rank=rank, n_x=n_x, n_y=n_y, t=t, f_sample=f_sample
    )


def _grid(c, Q=None):
    """Flat (2*n_f, n_t) -> (2, n_t, n_x, n_y) with NaN holes, repo convention."""
    Q = c["Q"] if Q is None else Q
    n_t = Q.shape[1]
    out = np.full((2, n_t, c["n_x"] * c["n_y"]), np.nan)
    A = Q.reshape(-1, 2, n_t).transpose(1, 2, 0)
    out[:, :, c["fluid"].ravel()] = A
    return out.reshape(2, n_t, c["n_x"], c["n_y"])


# ── 1. POD ────────────────────────────────────────────────────────────────────


@check
def pod_recovers_known_rank(v):
    """Exactly `rank` singular values above the numerical floor.

    A transposed data matrix still decomposes and still reconstructs; it just
    silently has the wrong rank, and nothing downstream complains.
    """
    c = toy()
    _, Sigma, _, _ = pod(c["Q"], subtract_mean=True)
    thresh = Sigma[0] * 1e-10
    n = int((Sigma > thresh).sum())
    if v:
        print(f"      sigma[:8] = {np.array2string(Sigma[:8], precision=3)}")
    return _report("pod_recovers_known_rank", n == c["rank"], f"{n} modes above threshold, expected {c['rank']}")


@check
def pod_orthonormal_and_exact(v):
    """Psi^T Psi = I and Psi B reproduces the data. Also POD's cycle-consistency."""
    c = toy()
    Q = c["Q"] - c["Q"].mean(1, keepdims=True)
    Psi, Sigma, B, _ = pod(Q)
    r = c["rank"]
    orth = np.abs(Psi[:, :r].T @ Psi[:, :r] - np.eye(r)).max()
    recon = np.abs(Psi @ B - Q).max() / np.abs(Q).max()
    cycle = np.abs(Psi[:, :r].T @ (Psi[:, :r] @ B[:r]) - B[:r]).max() / np.abs(B).max()
    ok = orth < 1e-12 and recon < 1e-12 and cycle < 1e-12
    return _report("pod_orthonormal_and_exact", ok, f"orth {orth:.1e}, recon {recon:.1e}, cycle {cycle:.1e}")


@check
def pod_methods_agree(v):
    """svd / snapshot / randomized return the same leading modes.

    Not the same *signs* -- an SVD is only defined up to a sign per mode -- so
    this compares singular values and the subspace, which is what any downstream
    use actually depends on.
    """
    c = toy()
    r = c["rank"]
    Ps, Ss, _, _ = pod(c["Q"], r, subtract_mean=True, method="svd")
    Pn, Sn, _, _ = pod(c["Q"], r, subtract_mean=True, method="snapshot")
    Pr, Sr, _, _ = pod(c["Q"], r, subtract_mean=True, method="randomized", n_iter=6)
    d_sig = max(np.abs(Ss - Sn).max(), np.abs(Ss - Sr).max()) / Ss[0]
    # subspace distance: ||P_a P_a^T - P_b P_b^T|| is sign- and rotation-blind
    d_sub = max(np.abs(Ps @ Ps.T - P.dot(P.T)).max() for P in (Pn, Pr))
    ok = d_sig < 1e-8 and d_sub < 1e-8
    return _report("pod_methods_agree", ok, f"sigma {d_sig:.1e}, subspace {d_sub:.1e}")


# ── 2. the linear estimators ──────────────────────────────────────────────────


@check
def lse_exact_recovery(v):
    """Linear sensors, no noise: POD-LSE is exact, not approximate."""
    c = toy(response="linear")
    m = PODLSE(r_field=c["rank"], r_sensor=None).fit(c["Q"], c["S"])
    e = m.score(c["Q"], c["S"])
    return _report("lse_exact_recovery", e < 1e-10, f"nmse {e:.2e} (need < 1e-10)")


@check
def lse_map_solves_least_squares(v):
    """The map agrees with an independent lstsq and its residual is orthogonal."""
    c = toy()
    # r_sensor == the true rank: ask for more and C C^T is singular, at which
    # point "the" least-squares solution is only defined up to its null space
    # and comparing two solvers compares their null-space conventions
    Psi, _, B, _ = pod(c["Q"], c["rank"], subtract_mean=True)
    _, _, C, _ = pod(c["S"] - c["S"].mean(1, keepdims=True), c["rank"])
    M = lse_map(B, C)
    M_ref = np.linalg.lstsq(C.T, B.T, rcond=None)[0].T
    d = np.abs(M - M_ref).max() / np.abs(M_ref).max()
    orth = np.abs((B - M @ C) @ C.T).max() / np.abs(B @ C.T).max()
    ok = d < 1e-8 and orth < 1e-8
    return _report("lse_map_solves_least_squares", ok, f"vs lstsq {d:.1e}, residual orthogonality {orth:.1e}")


@check
def epod_equals_podlse(v):
    """Extended POD == POD-LSE with an untruncated field basis, at any ridge."""
    c = toy(response="quadratic", snr_db=30)
    worst = 0.0
    for ridge in (0.0, 1e-4, 1e-1):
        a = PODLSE(r_field=None, r_sensor=None, ridge=ridge).fit(c["Q"], c["S"])
        b = ExtendedPOD(r_sensor=None, ridge=ridge).fit(c["Q"], c["S"])
        d = np.abs(a.predict(c["S"]) - b.predict(c["S"])).max() / np.abs(c["Q"]).max()
        worst = max(worst, d)
    return _report("epod_equals_podlse", worst < 1e-10, f"max |diff| {worst:.2e}")


@check
def epod_is_borees_formula(v):
    """Psi_ext[:, k] == Q_c c_k / ||c_k||^2, computed directly.

    The estimator gets there through a least-squares solve; Borée's paper writes
    it as a per-mode division. They agree only because the sensor POD gives
    orthogonal coefficient rows, which is worth asserting rather than assuming.
    """
    c = toy()
    m = ExtendedPOD(r_sensor=6, ridge=0.0).fit(c["Q"], c["S"])
    Qc = c["Q"] - m.q_mean
    direct = np.stack([Qc @ ck / (ck @ ck) for ck in m.C], axis=1)
    d = np.abs(direct - m.Psi_ext).max() / np.abs(m.Psi_ext).max()
    return _report("epod_is_borees_formula", d < 1e-10, f"max rel diff {d:.2e}")


@check
def extended_pod_recovers_sensor_map(v):
    """Forward direction: the extended modes rebuild the correlated sensor part.

    ``S`` is centred first, and that is not a detail. The extended modes are
    built from mean-subtracted field coefficients, so they span the fluctuation
    directions only -- the sensor mean is orthogonal to all of them and no
    number of modes will reproduce it. Comparing against an uncentred ``S``
    leaves exactly that mean as a residual, which looks like a 1% modelling
    error and is really a missing constant.
    """
    c = toy(response="linear")
    Psi, _, B, _ = pod(c["Q"], c["rank"], subtract_mean=True)
    Sc = c["S"] - c["S"].mean(1, keepdims=True)
    P = extended_pod(B, Sc)
    d = np.abs(P @ B - Sc).max() / np.abs(Sc).max()
    return _report("extended_pod_recovers_sensor_map", d < 1e-10, f"rel err {d:.2e}")


@check
def ridge_limits_are_sane(v):
    """ridge -> 0 recovers the unregularised map; ridge -> inf drives it to zero."""
    c = toy()
    Psi, _, B, _ = pod(c["Q"], c["rank"], subtract_mean=True)
    _, _, C, _ = pod(c["S"] - c["S"].mean(1, keepdims=True), c["rank"])
    M0 = lse_map(B, C, 0.0)
    small = np.abs(lse_map(B, C, 1e-14 * np.trace(C @ C.T)) - M0).max() / np.abs(M0).max()
    big = np.abs(lse_map(B, C, 1e14 * np.trace(C @ C.T))).max() / np.abs(M0).max()
    ok = small < 1e-6 and big < 1e-10
    return _report("ridge_limits_are_sane", ok, f"small {small:.1e}, large {big:.1e}")


@check
def prediction_equals_encode_then_expand(v):
    """predict() is exactly Psi @ encode() + mean. No hidden extra step."""
    c = toy()
    m = PODLSE(r_field=c["rank"], r_sensor=8, ridge=1e-6).fit(c["Q"], c["S"])
    d = np.abs(m.predict(c["S"]) - (m.Psi @ m.encode(c["S"]) + m.q_mean)).max()
    return _report("prediction_equals_encode_then_expand", d < 1e-12, f"max |diff| {d:.1e}")


@check
def standardisation_handles_mixed_units(v):
    """Scaling one channel by 1e6 does not change the answer.

    Six-component balances mix newtons and newton-metres. Without per-channel
    standardisation the sensor POD ranks channels by unit magnitude.
    """
    c = toy()
    S2 = c["S"].copy()
    S2[0] *= 1e6
    a = PODLSE(r_field=c["rank"], r_sensor=8, ridge=1e-8).fit(c["Q"], c["S"])
    b = PODLSE(r_field=c["rank"], r_sensor=8, ridge=1e-8).fit(c["Q"], S2)
    d = abs(a.score(c["Q"], c["S"]) - b.score(c["Q"], S2))
    return _report("standardisation_handles_mixed_units", d < 1e-8, f"|d nmse| {d:.1e}")


@check
def error_grows_with_sensor_noise(v):
    """Error increases monotonically as SNR falls. If not, you are fitting noise."""
    errs = []
    for snr in (60, 40, 20, 10):
        c = toy(response="linear", snr_db=snr, seed=1)
        tr, te = split_train_test(c["Q"].shape[1], 0.25, gap=40)
        m = PODLSE(r_field=c["rank"], r_sensor=8, ridge=1e-4).fit(c["Q"][:, tr], c["S"][:, tr])
        errs.append(m.score(c["Q"][:, te], c["S"][:, te]))
    ok = all(b > a for a, b in zip(errs, errs[1:]))
    return _report("error_grows_with_sensor_noise", ok, " < ".join(f"{e:.2e}" for e in errs))


@check
def linear_floor_under_nonlinearity(v):
    """Quadratic sensors put an irreducible floor under any linear estimator.

    The gap between the linear and quadratic columns is the headroom the
    two-branch autoencoder is trying to claim. If it is not there, the
    nonlinear model has nothing to win and a reported win is a leak.
    """
    out = {}
    for resp in ("linear", "quadratic"):
        c = toy(response=resp, seed=2)
        tr, te = split_train_test(c["Q"].shape[1], 0.25, gap=40)
        m = PODLSE(r_field=c["rank"], r_sensor=None).fit(c["Q"][:, tr], c["S"][:, tr])
        out[resp] = m.score(c["Q"][:, te], c["S"][:, te])
    ok = out["linear"] < 1e-10 < out["quadratic"]
    return _report(
        "linear_floor_under_nonlinearity", ok, f"linear {out['linear']:.1e}, quadratic {out['quadratic']:.1e}"
    )


# ── 3. observability, floors and the rank ceiling ─────────────────────────────


@check
def observability_is_one_for_linear_sensors(v):
    """rho^2 = 1 for every mode the sensors span, and < 1 for one they do not."""
    c = toy(response="linear", n_sensors=12)
    Psi, _, B, _ = pod(c["Q"], c["rank"], subtract_mean=True)
    _, _, C, _ = pod(c["S"] - c["S"].mean(1, keepdims=True), c["rank"])
    full = mode_observability(B, C)["rho2"]
    # drop a sensor mode: the field is rank 6, so 3 sensor modes cannot span it
    partial = mode_observability(B, pod(c["S"] - c["S"].mean(1, keepdims=True), 3)[2])["rho2"]
    if v:
        print(f"      full={np.round(full, 4)}  partial={np.round(partial, 4)}")
    ok = np.abs(full - 1).max() < 1e-8 and partial.sum() < c["rank"] - 0.5
    return _report(
        "observability_is_one_for_linear_sensors",
        ok,
        f"full min {full.min():.6f}, partial sum {partial.sum():.2f}/{c['rank']}",
    )


@check
def projection_floor_bounds_every_estimator(v):
    """No estimator using a basis can score below that basis's truncation error."""
    c = toy(response="quadratic", seed=3)
    tr, te = split_train_test(c["Q"].shape[1], 0.25, gap=40)
    r = 3
    Psi, _, _, qm = pod(c["Q"][:, tr], r, subtract_mean=True)
    floor = projection_floor(c["Q"][:, te], Psi, qm)
    m = PODLSE(r_field=r, r_sensor=None).fit(c["Q"][:, tr], c["S"][:, tr])
    got = m.score(c["Q"][:, te], c["S"][:, te])
    ok = got >= floor * (1 - 1e-9)
    return _report("projection_floor_bounds_every_estimator", ok, f"floor {floor:.4f} <= score {got:.4f}")


@check
def delay_embed_is_causal(v):
    """Block i is the record delayed by i*stride, zero-padded, no future leakage."""
    c = toy(n_t=60, n_sensors=4)
    S = c["S"]
    D = delay_embed(S, 5, 2)
    n_s = S.shape[0]
    ok = True
    for i in range(5):
        lag = 2 * i
        ok &= np.allclose(D[i * n_s : (i + 1) * n_s, lag:], S[:, : S.shape[1] - lag])
        ok &= np.all(D[i * n_s : (i + 1) * n_s, :lag] == 0)
    ok &= np.allclose(delay_embed(S, 1), S)
    return _report("delay_embed_is_causal", bool(ok), "blocks, padding and identity")


@check
def delay_embed_raises_rank(v):
    """Few channels + lags beats few channels alone. The premise of the module.

    Three instantaneous channels can only address a three-dimensional subspace
    of a rank-six field under any linear map, however many field modes you keep.
    Lagged copies raise that ceiling, and the score has to fall accordingly.
    """
    c = toy(response="linear", n_sensors=3, rank=6, seed=4)
    tr, te = split_train_test(c["Q"].shape[1], 0.25, gap=60, warmup=20)
    inst = PODLSE(r_field=6, r_sensor=None).fit(c["Q"][:, tr], c["S"][:, tr])
    e0 = inst.score(c["Q"][:, te], c["S"][:, te])
    Sd = delay_embed(c["S"], 21)
    lag = PODLSE(r_field=6, r_sensor=None, ridge=1e-8).fit(c["Q"][:, tr], Sd[:, tr])
    e1 = lag.score(c["Q"][:, te], Sd[:, te])
    if v:
        print(f"      instantaneous {e0:.3e} -> delayed {e1:.3e}")
    return _report("delay_embed_raises_rank", e1 < e0 / 10, f"{e0:.2e} -> {e1:.2e} with 21 lags")


@check
def no_leakage(v):
    """A contiguous split must score worse than a random one.

    Time-resolved PIV oversamples a wake, so a randomly held-out snapshot has
    near-copies of itself in the training set and every method scores
    brilliantly. If the two agree, the random split is holding nothing out.
    """
    c = toy(response="quadratic", n_t=1500, seed=5)
    n = c["Q"].shape[1]
    tr, te = split_train_test(n, 0.25, gap=60)
    m = PODLSE(r_field=c["rank"], r_sensor=None, ridge=1e-6).fit(c["Q"][:, tr], c["S"][:, tr])
    contig = m.score(c["Q"][:, te], c["S"][:, te])

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    rtr, rte = perm[: len(tr)], perm[len(tr) :][: len(te)]
    mr = PODLSE(r_field=c["rank"], r_sensor=None, ridge=1e-6).fit(c["Q"][:, rtr], c["S"][:, rtr])
    rand = mr.score(c["Q"][:, rte], c["S"][:, rte])
    if v:
        print(f"      contiguous {contig:.4e}  random {rand:.4e}")
    return _report("no_leakage", contig > rand, f"contiguous {contig:.3e} > random {rand:.3e}")


@check
def blocked_folds_respect_the_gap(v):
    """Folds partition the validation side and keep a guard band on both edges."""
    idx = np.arange(500)
    seen, ok = [], True
    for tr, va in blocked_folds(idx, k=5, gap=10):
        seen.append(va)
        ok &= len(np.intersect1d(tr, va)) == 0
        ok &= np.min(np.abs(tr[:, None] - va[None, :])) > 10 if len(tr) else True
    ok &= np.array_equal(np.sort(np.concatenate(seen)), idx)
    return _report("blocked_folds_respect_the_gap", bool(ok), f"{len(seen)} folds, no overlap, gap honoured")


# ── 4. the two-branch autoencoder ─────────────────────────────────────────────


@check
def sensor_windows_match_delay_embed(v):
    """The networks and the linear estimators see byte-identical observations."""
    c = toy(n_t=80, n_sensors=5)
    L, h = 6, 2
    W = sensor_windows(c["S"], L, h)
    D = delay_embed(c["S"], L, h)
    back = W[:, ::-1, :].transpose(1, 2, 0).reshape(L * c["S"].shape[0], -1)
    ok = np.array_equal(back, D) and np.allclose(W[:, -1, :].T, c["S"])
    return _report("sensor_windows_match_delay_embed", ok, f"{W.shape} windows, newest slice == S")


@check
def linear_latent_roundtrips(v):
    """LinearLatent encode/decode is exact on its own span, and torch == numpy."""
    import torch

    c = toy()
    Psi, _, _, qm = pod(c["Q"], c["rank"], subtract_mean=True)
    lat = LinearLatent(Psi, qm, device="cpu")
    Z = lat.encode(c["Q"])
    rt = nmse(c["Q"], lat.decode(Z))
    Zt = torch.as_tensor(Z.T, dtype=torch.float32)
    d = np.abs(lat.decode_torch(Zt).numpy().T - (lat.decode(Z) - qm)).max()
    d /= np.abs(c["Q"]).max()
    ok = rt < 1e-20 and d < 1e-5  # float32 on the torch side
    return _report("linear_latent_roundtrips", ok, f"round-trip nmse {rt:.1e}, torch vs numpy {d:.1e}")


@check
def branched_linear_matches_podlse(v):
    """branch='linear' converges to the closed-form LSE solution.

    Same objective, different solver. They will not agree to machine precision
    -- one is a QR solve and the other is Adam -- but a gap of more than a few
    percent means the two are not solving the same problem, which is a bug in
    the preprocessing, not a training-length issue.
    """
    c = toy(response="quadratic", seed=6)
    tr, te = split_train_test(c["Q"].shape[1], 0.25, gap=40, warmup=0)
    Psi, _, _, qm = pod(c["Q"][:, tr], c["rank"], subtract_mean=True)
    closed = PODLSE(r_field=c["rank"], r_sensor=None).fit(c["Q"][:, tr], c["S"][:, tr])
    e_closed = closed.score(c["Q"][:, te], c["S"][:, te])

    m = BranchedAE(
        LinearLatent(Psi, qm, device="cpu"),
        branch="linear",
        n_delays=1,
        n_epochs=600,
        patience=120,
        learning_rate=5e-3,
        batch_size=256,
        device="cpu",
        seed=0,
    ).fit(c["Q"], c["S"], tr)

    # On the training block the closed form is the exact minimiser, so no
    # gradient-descent run can beat it -- that is the sharp assertion, and it
    # fires on any preprocessing mismatch between the two paths.
    tr_closed = closed.score(c["Q"][:, tr], c["S"][:, tr])
    tr_learned = m.score(c["Q"], c["S"], tr)
    # On *test* they are allowed to differ, and the learned one is often
    # slightly better: early stopping on a held-out tail regularises, which the
    # closed-form solve has no equivalent of. A factor of two either way means
    # they are still solving the same problem; more than that does not.
    e_learned = m.score(c["Q"], c["S"], te)
    ratio = e_learned / e_closed
    if v:
        print(f"      train: closed {tr_closed:.5f} learned {tr_learned:.5f}")
        print(f"      test : closed {e_closed:.5f} learned {e_learned:.5f}")
    ok = tr_closed <= tr_learned * (1 + 1e-6) and 0.5 < ratio < 2.0
    return _report(
        "branched_linear_matches_podlse", ok, f"train {tr_closed:.4f}<={tr_learned:.4f}, test ratio {ratio:.2f}"
    )


@check
def branch_beats_the_linear_floor(v):
    """A nonlinear G claims the headroom a quadratic sensor response leaves.

    This is the whole thesis of the project in one assertion. If it fails on
    synthetic data where the nonlinearity is known and exact, it will not
    succeed on the experiment.
    """
    c = toy(response="quadratic", n_t=2000, seed=7)
    tr, te = split_train_test(c["Q"].shape[1], 0.25, gap=60, warmup=0)
    Psi, _, _, qm = pod(c["Q"][:, tr], c["rank"], subtract_mean=True)
    lat = LinearLatent(Psi, qm, device="cpu")
    closed = PODLSE(r_field=c["rank"], r_sensor=None).fit(c["Q"][:, tr], c["S"][:, tr])
    e_lin = closed.score(c["Q"][:, te], c["S"][:, te])
    m = BranchedAE(
        lat, branch="mlp", n_delays=1, hidden=(64, 64), n_epochs=400, patience=80, batch_size=128, device="cpu", seed=0
    ).fit(c["Q"], c["S"], tr)
    e_nl = m.score(c["Q"], c["S"], te)
    if v:
        print(f"      linear floor {e_lin:.5f}  mlp {e_nl:.5f}")
    return _report("branch_beats_the_linear_floor", e_nl < e_lin / 2, f"{e_lin:.4f} -> {e_nl:.4f}")


@check
def warmup_is_enforced(v):
    """fit() refuses train indices whose window reaches before the record."""
    c = toy(n_t=300)
    Psi, _, _, qm = pod(c["Q"], 4, subtract_mean=True)
    lat = LinearLatent(Psi, qm, device="cpu")
    try:
        BranchedAE(lat, n_delays=10, device="cpu").fit(c["Q"], c["S"], np.arange(0, 200))
    except ValueError as e:
        return _report("warmup_is_enforced", "history" in str(e), "raised on short history")
    return _report("warmup_is_enforced", False, "accepted a contaminated window")


@check
def torch_latent_wraps_an_autoencoder(v):
    """TorchLatent(AE) decodes to the same field the projector does."""
    from models.data_driven.autoencoders import AE

    c = toy(n_t=400, n_x=16, n_y=12, rank=4)
    X = _grid(c)
    ae = AE(n_latent=4, layer_dims=(64, 32), n_epochs=40, batch_size=32, device="cpu", seed=0).fit(X)
    lat = TorchLatent(ae, device="cpu")
    Z = lat.encode(X)
    a = lat.decode(Z)
    F = lat.field_target(c["Q"])
    import torch

    with torch.no_grad():
        b = lat.decode_torch(torch.as_tensor(Z.T, dtype=torch.float32)).numpy()
    scale = np.asarray(ae._scale).ravel()
    d = np.abs(b.T * scale[:, None] + ae.Q_mean - a).max() / np.abs(a).max()
    ok = d < 1e-5 and F.shape == (c["Q"].shape[1], c["Q"].shape[0]) and Z.shape[0] == 4
    return _report("torch_latent_wraps_an_autoencoder", ok, f"decode_torch vs decode {d:.1e}, latent {Z.shape}")


@check
def forecaster_beats_persistence(v):
    """Closed-loop rollout beats holding the last state, out to a real horizon.

    Persistence is the honest null model for a forecaster: at 250 Hz the state
    barely moves between samples, so anything that fails to beat it has learned
    the identity and nothing more.
    """
    c = toy(n_t=3000, rank=4, seed=8)
    Psi, _, B, _ = pod(c["Q"], 4, subtract_mean=True)
    tr = np.arange(2200)
    f = LatentForecaster(
        n_latent=4, n_delays=20, n_unroll=5, hidden=48, n_epochs=120, patience=25, device="cpu", seed=0
    ).fit(B, tr)
    h = 40
    errs, pers = [], []
    for t0 in range(2400, 2900, 50):
        p = f.rollout(B[:, t0 - 20 : t0], h)
        truth = B[:, t0 : t0 + h]
        errs.append(nmse(truth, p))
        pers.append(nmse(truth, np.repeat(B[:, t0 - 1 : t0], h, axis=1)))
    e, pe = float(np.mean(errs)), float(np.mean(pers))
    if v:
        print(f"      gru {e:.4f}  persistence {pe:.4f}")
    return _report("forecaster_beats_persistence", e < pe, f"gru {e:.4f} < persistence {pe:.4f} at h={h}")


# ── 5. the data layer ─────────────────────────────────────────────────────────


def _write_fixture(root, n_t=240, n_x=24, n_y=16, seed=0):
    """An RDS-shaped run directory: snapshots, mean field, forces, baselines.

    Exercises the readers against the real conventions -- (Ny, Nx) storage, the
    v sign flip, PIV pair indices starting at 201, MATLAB column-major .dat
    files at 2500 Hz, and the tunnel-off baseline files the drift fit needs.
    """
    c = toy(n_t=n_t, n_x=n_x, n_y=n_y, seed=seed)
    G = _grid(c)  # (2, n_t, n_x, n_y)
    run = "4p5d_10ms_yaw_0_0_0"
    snaps = os.path.join(root, run, "piv_snapshots_highres")
    os.makedirs(snaps)
    for i in range(n_t):
        a = 201 + 2 * i
        np.savez(
            os.path.join(snaps, f"PIV_PAIR_{a:06d}-{a + 1:06d}.npz"), u=G[0, i].T, v=(G[1, i] / we.V_SIGN).T
        )  # stored (Ny, Nx)

    X, Y = np.meshgrid(np.arange(n_x) * 2.0, np.arange(n_y) * 2.0)  # (Ny, Nx)
    np.savez(
        os.path.join(root, we.MEANFIELD),
        X=X,
        Y=Y,
        u_bar=np.nanmean(G[0], 0).T,
        v_bar=np.nanmean(G[1], 0).T,
        Rxx=np.nanvar(G[0], 0).T,
        Rxy=np.zeros((n_y, n_x)),
        Ryy=np.nanvar(G[1], 0).T,
        mask=c["fluid"].T,
    )

    # forces at 2500 Hz: PIV frame j lands on force sample 10j, and frame i has
    # pair 201+2i -> j = 100+i, so the record must reach 10*(100+n_t)
    fd = os.path.join(root, "synced_forces", "2026-04-01")
    os.makedirs(fd)
    n_f = we.FORCE_PER_PIV * (100 + n_t) + we.FORCE_PER_PIV
    F = np.zeros((we.N_FORCE_CHANNELS, n_f))
    for i in range(n_t):
        F[:, we.FORCE_PER_PIV * (100 + i) : we.FORCE_PER_PIV * (101 + i)] = c["S"][:, i : i + 1]
    F.astype("<f8").T.ravel(order="C").tofile(  # column-major on disk
        os.path.join(fd, "10ms_0_0_sync(12-00-00).dat")
    )
    for hh, off in ((11, 1.0), (13, 3.0)):  # two tunnel-off files -> linear drift
        np.full((we.N_FORCE_CHANNELS, 100), off).astype("<f8").T.ravel(order="C").tofile(
            os.path.join(fd, f"baseline(0{hh}-00-00).dat" if hh < 10 else f"baseline({hh}-00-00).dat")
        )
    return run, c


@check
def case_flat_roundtrips_with_holes(v):
    """flat -> unflat is lossless and puts the NaN holes back where they were."""
    c = toy(n_t=50)
    G = _grid(c)
    case = we.Case(run="toy", X=G, S=c["S"], pairs=list(range(50)), fluid_mask=c["fluid"])
    Q = case.flat()
    back = case.unflat(Q)
    same = np.array_equal(np.isnan(back), np.isnan(G))
    d = np.nanmax(np.abs(back - G))
    ok = Q.shape == c["Q"].shape and same and d < 1e-12
    return _report("case_flat_roundtrips_with_holes", ok, f"{Q.shape}, holes preserved, max |diff| {d:.1e}")


@check
def build_case_reads_an_rds_shaped_run(v):
    """The full reader path: snapshots, transpose, v sign, pairing, drift, sync."""
    root = tempfile.mkdtemp()
    try:
        run, c = _write_fixture(root, n_t=120)
        case = we.build_case(run, root=root, verbose=v)
        Q = case.flat()
        # the sensor record is constant over each PIV frame's 10 force samples,
        # so any resampling method must return exactly S -- up to the drift
        d_s = np.abs(case.S - (c["S"] - 2.0)).max() / np.abs(c["S"]).max()
        d_q = np.abs(Q - c["Q"]).max() / np.abs(c["Q"]).max()
        ok = case.n_t == 120 and case.drift_applied and d_s < 1e-10 and d_q < 1e-5  # float32 snapshots on disk
        if v:
            print(f"      Q {Q.shape} d_q={d_q:.1e}  S {case.S.shape} d_s={d_s:.1e}")
        return _report(
            "build_case_reads_an_rds_shaped_run", ok, f"field {d_q:.1e}, sensors {d_s:.1e}, drift {case.drift_applied}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


@check
def force_sync_picks_the_right_samples(v):
    """PIV frame i must land on force sample 10*(100+i), derived not hardcoded.

    Roman's notebook hardcodes the offset of 1000, which is right only for a run
    starting at pair 201. A ramp in the record makes an off-by-one visible;
    without it, a shifted pairing looks like a slightly worse model.
    """
    n_ch, n_f = 4, 5000
    F = np.tile(np.arange(n_f, dtype=float), (n_ch, 1))
    pairs = [201 + 2 * i for i in range(50)]
    got = we.sync_forces(F, pairs, method="decimate")[0]
    want = np.array([we.FORCE_PER_PIV * (100 + i) for i in range(50)], float)
    ok = np.array_equal(got, want)
    blk = we.sync_forces(F, pairs, method="block")[0]
    ok &= np.allclose(blk, want + (we.FORCE_PER_PIV - 1) / 2)
    return _report("force_sync_picks_the_right_samples", bool(ok), "decimate and block agree with the derived offset")


@check
def drift_correction_removes_the_ramp(v):
    """The tunnel-off drift fit reproduces and removes a known linear ramp."""
    root = tempfile.mkdtemp()
    try:
        fd = os.path.join(root, "synced_forces", "d")
        os.makedirs(fd)
        secs = [3600, 7200, 10800]
        for s, off in zip(secs, [1.0, 2.0, 3.0]):  # exactly linear in wall time
            h = s // 3600
            np.full((we.N_FORCE_CHANNELS, 50), off).astype("<f8").T.ravel(order="C").tofile(
                os.path.join(fd, f"baseline_0_0(0{h}-00-00).dat")
            )
        recs = we.find_force_files(root)
        fits = we.baseline_drift_fit(recs, degree=1)
        data = np.full((we.N_FORCE_CHANNELS, 10), 5.0)
        out = we.apply_drift_correction(data, 7200, fits[(0, 0)])
        d = np.abs(out - 3.0).max()  # 5.0 measured minus the fitted 2.0 zero
        return _report("drift_correction_removes_the_ramp", d < 1e-9, f"residual {d:.1e} after removing a known ramp")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@check
def study_script_runs_end_to_end(v):
    """`sparse_sensor_study.py --quick` completes on a fixture and writes a CSV.

    Slow (a minute or two) and gated behind --slow, but it is the only check
    that covers the driver: argument plumbing, the split, the plotting, the CSV
    schema. Everything else here tests a function in isolation.
    """
    if not _SLOW:
        return _report("study_script_runs_end_to_end", True, "skipped (pass --slow)")
    root = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    try:
        run, _ = _write_fixture(root, n_t=400)
        env = {**os.environ, "RDS_ROOT": root, "MPLBACKEND": "Agg"}
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "sparse_sensor_study.py"),
            "--quick",
            "--run",
            run,
            "--n",
            "400",
            "--r-field",
            "6",
            "--out",
            out,
            "--tag",
            "verify",
            "--frames",
            "10",
            "--device",
            "cpu",
        ]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
        csvp = os.path.join(out, "verify", "results.csv")
        ok = p.returncode == 0 and os.path.exists(csvp)
        if v or not ok:
            print(p.stdout[-3000:])
            print(p.stderr[-3000:])
        n_rows = sum(1 for _ in open(csvp)) - 1 if os.path.exists(csvp) else 0
        return _report("study_script_runs_end_to_end", ok, f"rc={p.returncode}, {n_rows} rows in results.csv")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


@check
def sweep_script_runs_and_resumes(v):
    """`sparse_sensor_sweep.py --quick` completes, then re-runs adding no rows.

    Resumption is the property the whole cluster workflow rests on -- a 24 h job
    that hits walltime is expected to be resubmitted and continue. It is also
    silently easy to break: the key is built from the requested config and
    compared against a CSV round trip, so any type or None/"" mismatch makes
    every restart refit everything and double the file. This catches that.
    """
    if not _SLOW:
        return _report("sweep_script_runs_and_resumes", True, "skipped (pass --slow)")
    root = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    try:
        run, _ = _write_fixture(root, n_t=400)
        env = {**os.environ, "RDS_ROOT": root, "MPLBACKEND": "Agg"}
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "sparse_sensor_sweep.py"),
            "--quick",
            "--run",
            run,
            "--n",
            "400",
            "--stages",
            "A",
            "B",
            "--latents-sweep",
            "4",
            "8",
            "--delays-sweep",
            "1",
            "5",
            "--latents",
            "pod",
            "--branches",
            "mlp",
            "--out",
            out,
            "--tag",
            "verify",
            "--device",
            "cpu",
            "--video-frames",
            "8",
        ]
        p1 = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        csvp = os.path.join(out, "verify", "results.csv")
        n1 = sum(1 for _ in open(csvp)) - 1 if os.path.exists(csvp) else -1

        p2 = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        n2 = sum(1 for _ in open(csvp)) - 1 if os.path.exists(csvp) else -1
        pack = os.path.exists(os.path.join(out, "verify", "video_pack.npz"))

        ok = p1.returncode == 0 and p2.returncode == 0 and n1 > 0 and n1 == n2 and pack and "0 to run" in p2.stdout
        if v or not ok:
            print(p1.stdout[-2000:], p1.stderr[-2000:])
            print(p2.stdout[-2000:], p2.stderr[-2000:])
        return _report("sweep_script_runs_and_resumes", ok, f"{n1} rows, {n2} after resume, video pack {pack}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


@check
def video_pack_renders(v):
    """A pack round-trips through `make_reconstruction_video.py` in both views.

    The renderer imports nothing from ``src/`` on purpose -- it has to run on a
    laptop with no torch -- so nothing else in this suite would notice if the
    pack format and the reader drifted apart.
    """
    if not _SLOW:
        return _report("video_pack_renders", True, "skipped (pass --slow)")
    d = tempfile.mkdtemp()
    try:
        c = toy(n_t=12, n_x=16, n_y=12, rank=4)
        G = _grid(c).astype(np.float32)
        np.savez_compressed(
            os.path.join(d, "video_pack.npz"),
            truth=G,
            pred_0=G * 0.9,
            x=np.arange(16, dtype=float),
            y=np.arange(12, dtype=float),
            meta=json.dumps(
                {"dt": 0.004, "run": "toy", "labels": {"pred_0": {"label": "toy", "nmse": 0.01, "pinned": True}}}
            ),
        )
        script = os.path.join(os.path.dirname(__file__), "make_reconstruction_video.py")
        env = {**os.environ, "MPLBACKEND": "Agg"}
        made = []
        for extra in (
            ["--format", "gif"],
            ["--format", "gif", "--view", "total"],
            ["--format", "gif", "--side-by-side"],
        ):
            r = subprocess.run(
                [sys.executable, script, os.path.join(d, "video_pack.npz"), "--out", d, "--fps", "5"] + extra,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode != 0:
                if v:
                    print(r.stdout, r.stderr)
                return _report("video_pack_renders", False, f"rc={r.returncode} on {extra}")
            made += [f for f in os.listdir(d) if f.endswith(".gif")]
        return _report("video_pack_renders", len(set(made)) >= 3, f"{len(set(made))} animations from one pack")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── runner ────────────────────────────────────────────────────────────────────

_SLOW = False


def main():
    global _SLOW
    ap = argparse.ArgumentParser(description="sparse-sensor reconstruction checks")
    ap.add_argument("-k", metavar="PATTERN", help="only checks whose name matches")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--slow", action="store_true", help="include the end-to-end run")
    args = ap.parse_args()
    _SLOW = args.slow

    checks = [c for c in CHECKS if not args.k or args.k in c.__name__]
    if not checks:
        print(f"no checks match {args.k!r}")
        return 2

    print(f"\nsparse-sensor reconstruction -- {len(checks)} checks\n")
    results = []
    for c in checks:
        try:
            results.append(bool(c(args.verbose)))
        except Exception:
            results.append(False)
            print(f"  [FAIL] {c.__name__:38s} raised:")
            traceback.print_exc(limit=4)

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} passed\n")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
