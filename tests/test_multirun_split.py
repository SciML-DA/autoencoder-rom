"""
The multi-run split, pinned.

`make_split` used to apply one contiguous cut to the *concatenated* record, so
`--runs A B C D E` without `--cross-yaw` did two silently wrong things: the
whole test block landed at the tail of the last run (making it an accidental
cross-yaw experiment without cross-yaw's safeguards), and the delay embedding
at each seam reached backwards into the previous run, handing the estimator 25
samples of a different flow as the history of this one.

Neither shows up as an error. Both change the answer.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

# the repo root, for `experiments` -- which lives outside src/ and so is
# deliberately not part of the installed package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from experiments.april_wake.data_preprocessing import make_split  # noqa: E402

N = 600  # snapshots per run
WARMUP = 24  # (n_delays - 1) * stride for the defaults below


class _Case:
    """Stand-in for a Case; make_split reads only `.run`."""

    def __init__(self, run):
        self.run = run


def _args(**kw):
    a = types.SimpleNamespace(delays=[25], delay_stride=1, delay_ahead=0, gap=100, test_fraction=0.25, cross_yaw=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _multi(n_runs=5, **kw):
    run_id = np.repeat(np.arange(n_runs), N)
    cases = [_Case(f"yaw{i}") for i in range(n_runs)]
    tr, te, held = make_split(_args(**kw), n_runs * N, run_id, cases)
    return tr, te, run_id


def test_single_run_is_unchanged():
    """One run must still produce exactly one contiguous train/test pair."""
    tr, te, _ = make_split(_args(), N, np.zeros(N, int), [_Case("r0")])
    assert tr[0] == WARMUP
    assert np.array_equal(tr, np.arange(WARMUP, tr[-1] + 1))
    assert np.array_equal(te, np.arange(te[0], N))
    assert te[0] - tr[-1] > _args().gap


def test_every_run_contributes_to_both_sides():
    """The point of the split: all yaws in training AND in test."""
    tr, te, run_id = _multi()
    assert set(run_id[tr]) == set(range(5)), "a run is missing from training"
    assert set(run_id[te]) == set(range(5)), "a run is missing from test"


def test_training_set_scales_with_the_number_of_runs():
    tr1, _, _ = _multi(n_runs=1)
    tr5, _, _ = _multi(n_runs=5)
    assert len(tr5) == 5 * len(tr1)


def test_no_window_spans_a_seam():
    """Every kept index is at least `warmup` past the start of its own run."""
    tr, te, run_id = _multi()
    for idx in (tr, te):
        offset = idx - np.array([np.flatnonzero(run_id == r)[0] for r in run_id[idx]])
        assert offset.min() >= WARMUP, (
            f"an index sits {offset.min()} samples into its run, inside the "
            "zero-padded warm-up -- the embedding is reading the previous run"
        )


def test_forward_lags_do_not_read_past_a_run_end():
    tr, te, run_id = _multi(delay_ahead=25)
    ends = {r: np.flatnonzero(run_id == r)[-1] for r in range(5)}
    for idx in (tr, te):
        room = np.array([ends[r] for r in run_id[idx]]) - idx
        assert room.min() >= 25, (
            f"an index is {room.min()} samples from its run's end with 25 forward lags -- it would read zero padding"
        )


def test_train_and_test_are_disjoint_and_gapped():
    tr, te, run_id = _multi()
    assert not set(tr) & set(te)
    for r in range(5):
        a, b = tr[run_id[tr] == r], te[run_id[te] == r]
        assert b[0] - a[-1] > 1, f"run {r}: no guard band between train and test"


def test_cross_yaw_still_holds_out_a_whole_run():
    """The other experiment must be unaffected by this change."""
    run_id = np.repeat(np.arange(5), N)
    cases = [_Case(f"yaw{i}") for i in range(5)]
    tr, te, held = make_split(_args(cross_yaw="yaw2"), 5 * N, run_id, cases)
    assert held == "yaw2"
    assert set(run_id[te]) == {2}
    assert 2 not in set(run_id[tr])


# ── per-run scoring ───────────────────────────────────────────────────────────


def test_nmse_by_run_is_self_normalised():
    """A pooled multi-run NMSE is not comparable with a single-run one.

    Pooling runs whose means differ puts the between-run offset into the
    denominator. That offset is large, low-rank and trivially predictable, so
    the pooled score improves for a reason unrelated to the reconstruction --
    which is how a five-yaw job can report 0.61 against a single-yaw 0.76 while
    being no better at the thing anyone cares about.

    `nmse_by_run` restores the comparison by normalising each run by itself.
    """
    import sys as _sys

    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "experiments", "april_wake", "scripts"))
    from sparse_sensor_study import nmse_by_run

    rng = np.random.default_rng(0)
    n_x, n = 40, 200
    # two runs with the SAME fluctuation quality but very different means
    fluct = rng.standard_normal((n_x, 2 * n))
    pred_err = 0.5 * rng.standard_normal((n_x, 2 * n))
    offset = np.zeros((n_x, 2 * n))
    offset[:, n:] = 25.0  # run 1 sits far from run 0
    Q = fluct + offset
    pred = Q - pred_err

    run_id = np.repeat([0, 1], n)
    te = np.arange(2 * n)
    cases = [_Case("runA"), _Case("runB")]

    got = nmse_by_run(Q, pred, te, run_id, cases)
    vals = [float(p.split("=")[1]) for p in got.split(";")]
    assert len(vals) == 2 and "runA" in got and "runB" in got

    # both runs are equally well reconstructed, so their self-normalised scores
    # must agree -- and both must exceed the flattering pooled number
    assert abs(vals[0] - vals[1]) < 0.05, f"per-run scores disagree: {vals}"
    from field_estimation.epod import nmse as _nmse

    pooled = _nmse(Q[:, te], pred[:, te])
    assert pooled < min(vals), f"pooled {pooled:.4f} should flatter relative to per-run {vals}"


def test_nmse_by_run_is_empty_for_a_single_run():
    import sys as _sys

    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "experiments", "april_wake", "scripts"))
    from sparse_sensor_study import nmse_by_run

    Q = np.random.default_rng(0).standard_normal((10, 50))
    assert nmse_by_run(Q, Q * 0.9, np.arange(50), np.zeros(50, int), [_Case("only")]) == ""
