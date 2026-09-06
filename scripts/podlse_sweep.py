#!/usr/bin/env python
"""
podlse_sweep.py
===============

Tune the linear reference itself.

    qsub -v BAND=20 hpc/podlse_sweep.pbs
    python scripts/podlse_sweep.py --band-hz 20 --tag b20

Why this exists
---------------
`hyper_search.py` scores every network against a `PODLSE` reference pinned at
the centre point -- ``r_field=16``, ``n_delays=25``, ``ridge=1e-3``, every
sensor direction kept -- while the networks search seventeen axes over a
thousand trials. The design doc calls `PODLSE` a method with "no
hyperparameters worth searching", and that is not right: `r_sensor` truncation
and the `sensor_basis` ranking are both documented in `tools/epod.py` as
mattering a great deal, and ``r_field=16`` sits far below the rank bound of
``N_s * n_delays = 300``.

That asymmetry cuts against us in the one place it matters. The claim being
built is that a linear estimator is at the information ceiling of twelve load
cells, and the networks have converged onto it. An untuned baseline cannot
support that claim: if the reference is under-powered, "the networks match the
linear solve" might only mean "the networks match a badly-configured linear
solve". Tuning it can only move the ceiling up, and a higher ceiling makes the
argument stronger, not weaker.

Splits and metric are imported from `hyper_search` rather than reimplemented,
so the numbers land on exactly the same train/validation blocks and are
directly comparable to every row in its CSV.

Selection discipline
--------------------
Validation only. The test block is not computed unless ``--with-test``, which
should be used once, at the end, for the handful of settings being reported --
the same rule the network search follows.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.sparse_sensors import add_data_args, band_limit, load_data  # noqa: E402
from tools.epod import PODLSE, delay_embed, nmse  # noqa: E402

from hyper_search import three_way  # noqa: E402  same splits, deliberately

# ── the grid ──────────────────────────────────────────────────────────────────
# n_delays is first because it is the only axis that raises the rank bound
# (N_s * n_delays), and the network search found it dominant by a wide margin.
GRID = {
    "n_delays":     [1, 5, 10, 25, 50, 100],
    "r_field":      [2, 4, 8, 16, 32, 64, 128],
    "r_sensor":     [None, 25, 100],
    "ridge":        [1e-4, 1e-3, 1e-2],
    "sensor_basis": ["pod", "pls"],
}

FIELDS = ["band_hz", "n_delays", "r_field", "r_sensor", "ridge", "sensor_basis",
          "rank_bound", "nmse_val", "nmse_test", "seconds"]
KEY = ("band_hz", "n_delays", "r_field", "r_sensor", "ridge", "sensor_basis")


def _keyfield(v):
    """Same None/'' normalisation as hyper_search: csv writes None as an empty
    field, so an unnormalised key re-runs every r_sensor=None row on resume."""
    return "" if v in (None, "", "None") else str(v)


def key_of(cfg):
    return tuple(_keyfield(cfg.get(k)) for k in KEY)


def rule(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--delays", type=int, nargs="+", default=[100])
    p.add_argument("--delay-stride", type=int, default=1)
    p.add_argument("--out", default="results/podlse_sweep")
    p.add_argument("--tag", default="default")
    p.add_argument("--max-hours", type=float, default=3.5)
    p.add_argument("--limit", type=int, default=0, help="cap fits, for testing")
    p.add_argument("--with-test", action="store_true",
                   help="also score the test block -- once, at the end, only")
    args = p.parse_args()

    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.csv")

    rule("0. data")
    Q_raw, S, unflat, case0, run_id, cases = load_data(args)
    # --band-hz is nargs="+" (one value low-passes, two band-pass), so this is a
    # list or None. band_limit does np.atleast_1d, so [20.0] and 20.0 are the
    # same filter -- which is what makes these rows comparable to hyper_search,
    # where the band arrives as a scalar. Record the scalar form for a
    # single-value band so the CSVs cross-reference on the same string.
    Q = Q_raw
    if args.band_hz:
        Q = band_limit(Q_raw, args.band_hz, 250.0, run_id)
        print(f"  [target band-limited: {args.band_hz} Hz]", flush=True)
        band = (float(args.band_hz[0]) if len(args.band_hz) == 1
                else "-".join(f"{b:g}" for b in args.band_hz))
    else:
        band = None
    tr, va, te = three_way(args, Q, run_id, cases)
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}   sensors {S.shape[0]}")

    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            done.add(key_of(r))
        print(f"  resuming: {len(done)} fit(s) already in {path}")
    else:
        with open(path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    cfgs = []
    for nd, rf, rs, lam, sb in itertools.product(*(GRID[k] for k in
            ("n_delays", "r_field", "r_sensor", "ridge", "sensor_basis"))):
        # r_sensor above the rank bound is the same fit as keeping everything,
        # and r_field above the bound cannot help -- skip both rather than
        # spend the budget re-measuring identical configurations.
        bound = S.shape[0] * nd
        if rs is not None and rs >= bound:
            continue
        cfgs.append(dict(band_hz=band, n_delays=nd, r_field=rf, r_sensor=rs,
                         ridge=lam, sensor_basis=sb, rank_bound=bound))
    # cheapest first, so a truncated run still covers the informative corner
    cfgs.sort(key=lambda c: (c["n_delays"], c["r_field"]))
    todo = [c for c in cfgs if key_of(c) not in done]
    if args.limit:
        todo = todo[: args.limit]
    rule(f"band {band if band else 'fullband'}: {len(cfgs)} configs, "
         f"{len(todo)} to run")

    t0 = time.time()
    best = (float("inf"), None)
    embed_cache: dict = {}
    for i, c in enumerate(todo):
        if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print("  walltime budget reached; stopping cleanly")
            break
        t1 = time.time()
        nd = c["n_delays"]
        if nd not in embed_cache:
            embed_cache.clear()          # one embedding at a time; these are large
            embed_cache[nd] = delay_embed(S, nd, args.delay_stride, args.delay_ahead)
        Sd = embed_cache[nd]
        m = PODLSE(r_field=c["r_field"], r_sensor=c["r_sensor"],
                   sensor_basis=c["sensor_basis"], ridge=c["ridge"],
                   pod_method="randomized").fit(Q[:, tr], Sd[:, tr])
        row = dict(c)
        row["nmse_val"] = nmse(Q[:, va], m.predict(Sd[:, va]))
        row["nmse_test"] = (nmse(Q[:, te], m.predict(Sd[:, te]))
                            if args.with_test else float("nan"))
        row["seconds"] = time.time() - t1
        with open(path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(
                {k: row.get(k) for k in FIELDS})
        if row["nmse_val"] < best[0]:
            best = (row["nmse_val"], c)
        flag = "  <-- best so far" if row["nmse_val"] == best[0] else ""
        print(f"  [{i + 1:>4}/{len(todo)}] L={nd:<4} r_f={c['r_field']:<4} "
              f"r_s={str(c['r_sensor']):<5} lam={c['ridge']:<7g} "
              f"{c['sensor_basis']:<4} val {row['nmse_val']:.4f} "
              f"{row['seconds']:.1f}s{flag}", flush=True)

    if best[1]:
        print(f"\n  best: val {best[0]:.4f}")
        print(f"    {best[1]}")
    rule("done")
    print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
