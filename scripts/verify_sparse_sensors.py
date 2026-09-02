#!/usr/bin/env python
"""
verify_sparse_sensors.py
========================

Verification suite for sparse-sensor reconstruction, runnable with no network
and no experimental data.

Run it::

    python scripts/verify_sparse_sensors.py            # all checks
    python scripts/verify_sparse_sensors.py -k lse     # only matching names
    python scripts/verify_sparse_sensors.py -v         # show every number

Each check is a property that must hold for *any* correct implementation, so
you can point them at your own code by swapping the estimator in ``ESTIMATOR``
below. They are ordered from "your linear algebra is wrong" to "your method
does not generalise", which is roughly the order you want to find out.

The important ones
------------------
``lse_exact_recovery`` is the check that matters most. When the sensor map is
genuinely linear and noise-free, POD-LSE is not an approximation -- it is the
exact solution, and it must come back at round-off. If that check fails, the
bug is in your algebra, and no amount of tuning will fix it. Everything
downstream is only meaningful once it passes.

``no_leakage`` is the one people fail silently. Time-resolved PIV at 1 kHz
oversamples a wake wildly, so a *random* train/test split puts near-duplicate
snapshots on both sides and every method scores brilliantly. The check fits on
a contiguous block and scores on a disjoint later block.

``linear_floor_under_nonlinearity`` encodes the claim from Novoa's notes that
motivates the whole nonlinear branch: when the sensors respond quadratically
to the flow -- which load cells do -- a linear estimator has an error floor
that more modes and more regularisation tuning cannot get under.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.wake_synthetic import WakeConfig, generate  # noqa: E402
from tools.epod import (  # noqa: E402
    PODLSE,
    extended_pod,
    lse_map,
    nmse,
    pod,
    split_train_test,
)

# Swap this for your own class to run the same checks against it. It needs
# .fit(Q, S) / .predict(S) / .encode(S), with columns as snapshots.
ESTIMATOR = PODLSE

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ── helpers ───────────────────────────────────────────────────────────────────


def _case(**kw):
    """A small, fast synthetic run. Overrides go straight to WakeConfig.

    float64 and a 3-second record, both deliberately. In float32 the rank and
    exact-recovery checks measure the storage format rather than the method:
    quantising a 10 m/s field to float32 leaves singular values around 5e-4,
    which swamps the 1e-10 an exact reconstruction should reach. And at
    f_sample=1000 the default 600 samples cover under two periods of the
    meandering mode, so a train block and a test block have genuinely
    different statistics and every score looks terrible for the wrong reason.
    400 Hz x 1200 samples is 3 s, about nine meander periods, with the
    shedding still sampled ~12 times per period.
    """
    base: dict = dict(
        n_t=1200,
        f_sample=400.0,
        n_x=64,
        n_y=32,
        seed=0,
        piv_snr_db=float("inf"),
        dtype="float64",
    )
    base.update(kw)
    return generate(WakeConfig(**base))


def _flat(case):
    """(Q, S) with columns as snapshots, ready for POD."""
    return case.flat(subtract_mean=False), case.S.T.astype(np.float64)


def _report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")
    return ok


# ── 1. the decomposition itself ───────────────────────────────────────────────


@check
def pod_orthonormal_and_exact(v):
    """Psi has orthonormal columns and Psi B reproduces the data."""
    case = _case()
    Q = case.flat(subtract_mean=True)
    Psi, Sigma, B, _ = pod(Q)

    r = min(Psi.shape[1], 40)
    orth = np.abs(Psi[:, :r].T @ Psi[:, :r] - np.eye(r)).max()
    recon = np.abs(Psi @ B - Q).max() / np.abs(Q).max()
    # in float64: summing ~5M float32 squares loses ~2e-8 on its own, which
    # would fail the 1e-12 threshold below for reasons unrelated to the POD
    total = np.sum(Q.astype(np.float64) ** 2)
    energy = abs(np.sum(Sigma**2) - total) / total
    monotone = bool(np.all(np.diff(Sigma) <= 1e-9 * Sigma[0]))

    if v:
        print(f"      orth={orth:.2e} recon={recon:.2e} energy={energy:.2e}")
    ok = orth < 1e-10 and recon < 1e-10 and energy < 1e-12 and monotone
    return _report(
        "pod_orthonormal_and_exact",
        ok,
        f"orth {orth:.1e}, recon {recon:.1e}, energy {energy:.1e}",
    )


@check
def pod_recovers_known_rank(v):
    """A field built as rank r must have exactly r non-zero singular values.

    This is the check that tells you your data matrix is oriented correctly.
    A transposed Q still decomposes happily and still reconstructs -- it just
    silently has the wrong rank, which nothing downstream will complain about.
    """
    case = _case()
    Q = case.flat(subtract_mean=True)
    _, Sigma, _, _ = pod(Q)

    tol = Sigma[0] * max(Q.shape) * np.finfo(np.float64).eps
    found = int((Sigma > tol).sum())
    expect = case.truth.rank

    if v:
        print(f"      sigma[{expect-1}]={Sigma[expect-1]:.3e} "
              f"sigma[{expect}]={Sigma[expect]:.3e} gap="
              f"{Sigma[expect-1]/max(Sigma[expect],1e-300):.1e}")
    ok = found == expect
    return _report(
        "pod_recovers_known_rank", ok, f"found {found}, built {expect}"
    )


@check
def extended_pod_recovers_sensor_map(v):
    """EPOD modes reproduce the sensor signal's correlated part.

    With a linear noiseless sensor map, *all* of the sensor signal is
    correlated with the field, so the extended modes must reconstruct S
    exactly from the field coefficients.
    """
    case = _case(sensor_response="linear", sensor_snr_db=float("inf"))
    Q, S = _flat(case)
    Qc = Q - Q.mean(axis=1, keepdims=True)
    Sc = S - S.mean(axis=1, keepdims=True)

    _, _, B, _ = pod(Qc)
    r = case.truth.rank
    B = B[:r]
    psi_ext = extended_pod(B, Sc)  # (N_s, r)
    err = np.abs(psi_ext @ B - Sc).max() / np.abs(Sc).max()

    if v:
        print(f"      extended modes {psi_ext.shape}, rel err {err:.2e}")
    ok = err < 1e-8
    return _report("extended_pod_recovers_sensor_map", ok, f"rel err {err:.1e}")


# ── 2. the linear map ─────────────────────────────────────────────────────────


@check
def lse_map_solves_least_squares(v):
    """lse_map must agree with an independent lstsq solve."""
    rng = np.random.default_rng(0)
    B = rng.standard_normal((7, 300))
    C = rng.standard_normal((5, 300))

    M = lse_map(B, C, ridge=0.0)
    M_ref = np.linalg.lstsq(C.T, B.T, rcond=None)[0].T
    err = np.abs(M - M_ref).max() / np.abs(M_ref).max()

    # and the residual must be orthogonal to the regressors: the normal equations
    resid = B - M @ C
    orth = np.abs(resid @ C.T).max() / (np.abs(B).max() * np.abs(C).max())

    if v:
        print(f"      vs lstsq {err:.2e}, normal-equation residual {orth:.2e}")
    ok = err < 1e-9 and orth < 1e-10
    return _report(
        "lse_map_solves_least_squares", ok, f"vs lstsq {err:.1e}, normal {orth:.1e}"
    )


@check
def ridge_limits_are_sane(v):
    """ridge -> 0 recovers the unregularised map; ridge -> inf drives M to 0."""
    rng = np.random.default_rng(1)
    B = rng.standard_normal((4, 200))
    C = rng.standard_normal((4, 200))

    M0 = lse_map(B, C, 0.0)
    M_small = lse_map(B, C, 1e-12 * np.trace(C @ C.T))
    M_huge = lse_map(B, C, 1e12 * np.trace(C @ C.T))

    near = np.abs(M_small - M0).max() / np.abs(M0).max()
    shrunk = np.abs(M_huge).max() / np.abs(M0).max()

    if v:
        print(f"      |M(1e-12)-M(0)|={near:.2e}  |M(1e12)|/|M(0)|={shrunk:.2e}")
    ok = near < 1e-6 and shrunk < 1e-9
    return _report("ridge_limits_are_sane", ok, f"near {near:.1e}, shrunk {shrunk:.1e}")


@check
def lse_exact_recovery(v):
    """THE check: linear noiseless sensors => reconstruction at round-off.

    With sensor_response="linear" and no noise, s(t) = M_true a(t) exactly and
    the field is exactly rank r. POD-LSE is then not an approximation but the
    exact inverse, so anything above ~1e-10 here is a bug in the algebra, not
    a modelling limitation.
    """
    case = _case(sensor_response="linear", sensor_snr_db=float("inf"))
    Q, S = _flat(case)
    r = case.truth.rank

    model = ESTIMATOR(r_field=r, r_sensor=r, ridge=0.0).fit(Q, S)
    err = model.score(Q, S)

    if v:
        rank_s = np.linalg.matrix_rank(S - S.mean(axis=1, keepdims=True))
        print(f"      field rank {r}, sensor rank {rank_s}, nmse {err:.3e}")
    ok = err < 1e-10
    return _report("lse_exact_recovery", ok, f"nmse {err:.2e} (want < 1e-10)")


@check
def prediction_equals_encode_then_expand(v):
    """predict(S) must equal Psi @ encode(S) + mean, exactly."""
    case = _case(sensor_response="linear")
    Q, S = _flat(case)
    m = ESTIMATOR(r_field=12, r_sensor=10, ridge=1e-8).fit(Q, S)

    direct = m.predict(S)
    staged = m.Psi @ m.encode(S) + m.q_mean
    err = np.abs(direct - staged).max() / np.abs(direct).max()

    if v:
        print(f"      max abs discrepancy {err:.2e}")
    ok = err < 1e-12
    return _report("prediction_equals_encode_then_expand", ok, f"rel {err:.1e}")


# ── 3. statistical behaviour ──────────────────────────────────────────────────


@check
def beats_the_mean(v):
    """Any working method scores nmse < 1 -- i.e. beats predicting the mean."""
    case = _case(sensor_response="quadratic", sensor_snr_db=40.0)
    Q, S = _flat(case)
    tr, te = split_train_test(Q.shape[1], 0.25, gap=50)

    m = ESTIMATOR(r_field=20, r_sensor=15, ridge=1e-6).fit(Q[:, tr], S[:, tr])
    err = m.score(Q[:, te], S[:, te])

    baseline = nmse(Q[:, te], np.repeat(Q[:, tr].mean(1, keepdims=True), te.size, 1))
    if v:
        print(f"      model {err:.4f} vs mean-predictor {baseline:.4f}")
    ok = err < 0.95 * baseline
    return _report("beats_the_mean", ok, f"nmse {err:.3f} vs mean {baseline:.3f}")


@check
def error_grows_with_sensor_noise(v):
    """Reconstruction error must increase monotonically with sensor noise."""
    errs = []
    for snr in (60.0, 40.0, 20.0, 10.0, 0.0):
        case = _case(sensor_response="linear", sensor_snr_db=snr)
        Q, S = _flat(case)
        tr, te = split_train_test(Q.shape[1], 0.25, gap=50)
        m = ESTIMATOR(r_field=15, r_sensor=12, ridge=1e-4).fit(Q[:, tr], S[:, tr])
        errs.append(m.score(Q[:, te], S[:, te]))

    if v:
        print("      " + "  ".join(f"{e:.4f}" for e in errs))
    # allow a small non-monotone wobble from the finite record
    ok = all(b > a - 0.02 for a, b in zip(errs, errs[1:])) and errs[-1] > errs[0]
    return _report(
        "error_grows_with_sensor_noise", ok, f"{errs[0]:.3f} -> {errs[-1]:.3f}"
    )


@check
def standardisation_handles_mixed_units(v):
    """Scaling one channel by 1e6 must not change the answer.

    Forces are newtons and moments are newton-metres; if a channel's units
    change the reconstruction, the sensor POD is ranking channels by unit
    magnitude rather than by information content.
    """
    case = _case(sensor_response="quadratic")
    Q, S = _flat(case)
    tr, te = split_train_test(Q.shape[1], 0.25, gap=50)

    S_scaled = S.copy()
    S_scaled[3:6] *= 1e6  # the moment channels of disc 0

    a = ESTIMATOR(r_field=15, r_sensor=12, ridge=1e-6, standardise_sensors=True)
    b = ESTIMATOR(r_field=15, r_sensor=12, ridge=1e-6, standardise_sensors=True)
    e1 = a.fit(Q[:, tr], S[:, tr]).score(Q[:, te], S[:, te])
    e2 = b.fit(Q[:, tr], S_scaled[:, tr]).score(Q[:, te], S_scaled[:, te])

    if v:
        print(f"      original {e1:.5f}, rescaled {e2:.5f}")
    ok = abs(e1 - e2) < 1e-3 * max(e1, 1e-9)
    return _report(
        "standardisation_handles_mixed_units", ok, f"{e1:.4f} vs {e2:.4f}"
    )


@check
def no_leakage(v):
    """A contiguous split must score worse than a random one.

    If these two come out the same, the random split is not actually holding
    anything out and any score you quote from it is meaningless.
    """
    case = _case(n_t=800, sensor_response="quadratic", sensor_snr_db=30.0)
    Q, S = _flat(case)
    n = Q.shape[1]

    tr, te = split_train_test(n, 0.25, gap=60)
    e_contig = (
        ESTIMATOR(r_field=20, r_sensor=15, ridge=1e-6)
        .fit(Q[:, tr], S[:, tr])
        .score(Q[:, te], S[:, te])
    )

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    rtr, rte = perm[: int(0.75 * n)], perm[int(0.75 * n) :]
    e_random = (
        ESTIMATOR(r_field=20, r_sensor=15, ridge=1e-6)
        .fit(Q[:, rtr], S[:, rtr])
        .score(Q[:, rte], S[:, rte])
    )

    if v:
        print(f"      contiguous {e_contig:.4f}, random(leaky) {e_random:.4f}")
    ok = e_contig > e_random
    return _report(
        "no_leakage", ok, f"contiguous {e_contig:.3f} > random {e_random:.3f}"
    )


@check
def linear_floor_under_nonlinearity(v):
    """Quadratic sensors put a floor under any linear estimator.

    Same field, same noise, same modes -- only the sensor response differs.
    The linear case should reach round-off; the quadratic one should not, no
    matter how many modes it is given. That gap is the headroom the two-branch
    autoencoder is meant to claim.
    """
    errs = {}
    for resp in ("linear", "quadratic"):
        case = _case(sensor_response=resp, sensor_snr_db=float("inf"))
        Q, S = _flat(case)
        r = case.truth.rank
        best = min(
            ESTIMATOR(r_field=r, r_sensor=k, ridge=0.0).fit(Q, S).score(Q, S)
            for k in (r // 2, r, min(S.shape[0], 18))
        )
        errs[resp] = best

    if v:
        print(f"      linear {errs['linear']:.3e}, quadratic {errs['quadratic']:.3e}")
    ok = errs["linear"] < 1e-10 and errs["quadratic"] > 1e3 * max(
        errs["linear"], 1e-16
    )
    return _report(
        "linear_floor_under_nonlinearity",
        ok,
        f"linear {errs['linear']:.1e}, quadratic {errs['quadratic']:.1e}",
    )


# ── 4. plumbing ───────────────────────────────────────────────────────────────


@check
def masked_points_are_excluded(v):
    """NaN disc points must never reach the data matrix."""
    case = _case(mask_discs=True)
    assert np.isnan(case.U).any(), "expected NaN over the disc footprints"
    Q = case.flat()
    ok = bool(np.isfinite(Q).all())
    n_masked = int((~case.fluid_mask).sum())
    if v:
        print(f"      {n_masked} masked grid points, Q shape {Q.shape}")
    return _report(
        "masked_points_are_excluded", ok, f"{n_masked} masked, Q finite: {ok}"
    )


@check
def grid_roundtrip(v):
    """flat() then to_grid() must return the original field on fluid points."""
    case = _case()
    Q = case.flat(subtract_mean=False)
    back = case.to_grid(Q[:, 5])  # (Nu, Nx, Ny)
    ref = case.U[:, 5]

    m = case.fluid_mask
    err = np.abs(back[:, m] - ref[:, m]).max() / np.abs(ref[:, m]).max()
    holes_ok = bool(np.isnan(back[:, ~m]).all()) if (~m).any() else True

    if v:
        print(f"      rel err {err:.2e}, holes preserved {holes_ok}")
    ok = err < 1e-6 and holes_ok
    return _report("grid_roundtrip", ok, f"rel err {err:.1e}")


@check
def generator_is_deterministic(v):
    """Same seed, same bytes -- otherwise nothing above is reproducible."""
    a, b = _case(seed=3), _case(seed=3)
    c = _case(seed=4)
    # equal_nan: the disc footprints are NaN, and NaN != NaN would make every
    # masked run look non-deterministic
    same = np.array_equal(a.U, b.U, equal_nan=True) and np.array_equal(a.S, b.S)
    differ = not np.array_equal(a.U, c.U, equal_nan=True)
    if v:
        print(f"      seed3==seed3 {same}, seed3!=seed4 {differ}")
    return _report(
        "generator_is_deterministic", same and differ, f"same {same}, differ {differ}"
    )


@check
def npz_roundtrip(v):
    """write_run / load_run preserve the arrays."""
    import shutil
    import tempfile

    from datasets.wake_synthetic import load_run, write_run

    case = _case(n_t=12, n_x=48, n_y=24)
    tmp = tempfile.mkdtemp()
    try:
        write_run(os.path.join(tmp, "run"), case)
        U, S, _t, _x, _y, ch = load_run(os.path.join(tmp, "run"))
        eu = np.nanmax(np.abs(U - case.U))
        es = np.nanmax(np.abs(S - case.S))
        ok = eu == 0 and es == 0 and ch == case.channels and U.shape == case.U.shape
        if v:
            print(f"      U err {eu}, S err {es}, shape {U.shape}, {len(ch)} channels")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _report("npz_roundtrip", ok, f"U {eu:g}, S {es:g}, {len(ch)} channels")


# ── runner ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="sparse-sensor verification suite")
    ap.add_argument("-k", metavar="PATTERN", help="only checks whose name matches")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    checks = [c for c in CHECKS if not args.k or args.k in c.__name__]
    if not checks:
        print(f"no checks match {args.k!r}")
        return 2

    print(f"\nsparse-sensor verification -- {len(checks)} checks, "
          f"estimator = {ESTIMATOR.__name__}\n")
    results = []
    for c in checks:
        try:
            results.append(bool(c(args.verbose)))
        except Exception:
            results.append(False)
            print(f"  [FAIL] {c.__name__:38s} raised:")
            traceback.print_exc(limit=3)

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} passed\n")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
