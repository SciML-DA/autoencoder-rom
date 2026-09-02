"""
The learning-curve verdict, pinned.

This section exists to settle one specific disagreement: whether a single run
carries enough data. The multi-run experiment did not test that -- it pooled
five different yaws under one shared linear map, which is a heterogeneity
effect, not a data-volume one. The learning curve tests it by extrapolation
from the record that exists.

The verdict it prints is the deliverable, so the thresholds that produce it are
worth pinning. A curve that is still falling and one that has saturated lead to
opposite recommendations -- record more at one condition, or stop trying.
"""

from __future__ import annotations

import io
import os
import sys
import types
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from convergence_diagnosis import _verdict_learning_curve  # noqa: E402


def _rows(values, L=25, sizes=(500, 1000, 2000, 4000)):
    return [dict(section="learning_curve", n_delays=L, n_train=n,
                 n_params=7680, nmse_train=v - 0.01, nmse_test=v)
            for n, v in zip(sizes, values)]


def _verdict(rows, lc_delays):
    args = types.SimpleNamespace(lc_delays=lc_delays)
    buf = io.StringIO()
    with redirect_stdout(buf):
        _verdict_learning_curve(rows, args)
    return buf.getvalue()


def test_still_falling_reports_data_limited():
    """A curve dropping 0.03 over the last doubling must say DATA-LIMITED."""
    out = _verdict(_rows([0.90, 0.85, 0.80, 0.77]), [25])
    assert "DATA-LIMITED" in out, out
    assert "recording more" in out


def test_flat_curve_reports_saturated():
    out = _verdict(_rows([0.762, 0.759, 0.757, 0.7568]), [25])
    assert "flat" in out and "one run is enough" in out, out


def test_marginal_curve_is_called_marginal():
    out = _verdict(_rows([0.80, 0.785, 0.775, 0.770]), [25])
    assert "marginally" in out, out


def test_slope_is_per_doubling_not_per_point():
    """A 4x jump in data must not be scored as a single doubling."""
    wide = [dict(section="learning_curve", n_delays=25, n_train=n,
                 n_params=7680, nmse_train=v, nmse_test=v)
            for n, v in ((1000, 0.80), (4000, 0.76))]
    out = _verdict(wide, [25])
    # 0.04 over two doublings is 0.02 per doubling, not 0.04
    assert "+0.0200 per doubling" in out, out


def test_flags_when_only_long_windows_still_improve():
    """Different recommendation: shorten the window rather than record more."""
    rows = _rows([0.760, 0.758, 0.757, 0.7568], L=10) + \
           _rows([0.90, 0.85, 0.80, 0.77], L=100)
    out = _verdict(rows, [10, 100])
    assert "relative to model" in out, out
