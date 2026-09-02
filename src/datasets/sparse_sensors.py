"""Dataset assembly for the sparse-sensor reconstruction scripts.

Builds the (Q, S) matrices and the train/test indices that
``scripts/sparse_sensor_sweep.py`` and ``scripts/diagnose_sensors.py`` consume,
so the loading, masking and splitting conventions are defined once.
"""

from __future__ import annotations

import numpy as np

from tools.epod import split_train_test

from .wake_experiment import RUNS, build_case, concat_cases

__all__ = ["add_data_args", "probe_points", "apply_force_lag", "load_data",
           "make_split"]


def add_data_args(parser) -> None:
    """
    Adds the arguments read by load_data and make_split to a parser.

    Every script calling load_data must call this: load_data reads the
    attributes directly, so a missing one raises AttributeError at load time
    rather than at parse time.
    """
    d = parser.add_argument_group("data")
    d.add_argument("--run", default=RUNS[0])
    d.add_argument("--runs", nargs="*", help="several runs, for --cross-yaw")
    d.add_argument("--n", type=int, default=None, help="snapshots per run")
    d.add_argument("--stride", type=int, default=1)
    d.add_argument("--sync", default="block", choices=["block", "decimate", "nearest"])
    d.add_argument("--no-drift", action="store_true", help="skip drift correction")
    d.add_argument("--force-smooth", type=int, default=0,
                   help="moving-average window on the raw 2500 Hz force record")
    d.add_argument("--notch", type=float, nargs="*", default=None, metavar="HZ",
                   help="notch these frequencies out of the force record")
    d.add_argument("--notch-q", type=float, default=8.0, help="notch quality factor")
    d.add_argument("--probes", type=int, nargs="*", default=None, metavar="IX IY",
                   help="in-field velocity probes as explicit ix iy pairs")
    d.add_argument("--n-probes", type=int, default=0,
                   help="auto-place this many probes on valid fluid points; this "
                        "stops the result being a force-only reconstruction")
    d.add_argument("--lowres", action="store_true",
                   help="read piv_snapshots/ rather than piv_snapshots_highres/. "
                        "The reference notebook uses the low-resolution export "
                        "(4950 points); the two are NOT row-for-row aligned and "
                        "two runs have no lowres data at all.")
    d.add_argument("--mask-tol", type=float, default=1.0, metavar="F",
                   help="keep a grid point unless it is invalid in more than F "
                        "of frames, filling what remains. The default 1.0 keeps "
                        "the whole field, as the reference notebook does. Pass "
                        "0.0 for the old strict rule, which discarded 63%% of "
                        "this grid to remove a body of at most 112 points.")
    d.add_argument("--mask-fill", default="interp", choices=["interp", "zero"],
                   help="how kept points' gaps are filled. 'zero' reproduces "
                        "the notebook's nan_to_num; 'interp' is better, because "
                        "a zero is a measured value to an SVD and a gap is not.")
    d.add_argument("--force-lag", type=int, default=0, metavar="N",
                   help="re-time the force record by N PIV samples against the "
                        "field before embedding. Negative N pairs each frame "
                        "with force samples recorded LATER than the pairing "
                        "says. Calibrate it with scripts/spectra.py or the lag "
                        "scan in diagnose_sensors.py, and quote it with every "
                        "score -- it moves NMSE by several percent.")
    d.add_argument("--delay-ahead", type=int, default=0, metavar="K",
                   help="also stack K FUTURE sensor samples, making the "
                        "estimator two-sided. The causal embedding cannot reach "
                        "a correlation that sits at negative lag, and the lag "
                        "scan put the optimum at -25 samples. Set 0 only if the "
                        "estimator has to run in real time.")
    d.add_argument("--sensor-basis", default="pod", choices=["pod", "pls"],
                   help="how sensor directions are ranked before truncation. "
                        "'pod' by sensor variance (which the rig resonance "
                        "wins); 'pls' by cross-covariance with the field.")
    d.add_argument("--test-fraction", type=float, default=0.25)
    d.add_argument("--gap", type=int, default=100)
    d.add_argument("--cross-yaw", nargs="?", const="", default=None, metavar="RUN",
                   help="hold out a whole run instead of a time block")


def probe_points(flat):
    """
    Converts a flat [ix, iy, ...] list into [(ix, iy), ...].

    Raises:
        ValueError: If the list holds an odd number of values.
    """
    if not flat:
        return None
    if len(flat) % 2:
        raise ValueError(f"--probes takes ix iy pairs, got {len(flat)} values")
    return [(int(flat[i]), int(flat[i + 1])) for i in range(0, len(flat), 2)]


def apply_force_lag(S, lag: int, run_id=None):
    """Re-times a sensor record against the field by `lag` PIV samples.

    Column `t` of the result holds `S[:, t - lag]`, matching the convention of
    the lag scan in `diagnose_sensors.py`, which slides the record with
    `np.roll`. A negative `lag` therefore pairs each PIV frame with force
    samples recorded *after* it.

    This exists because `delay_embed` is strictly causal. It reaches backwards
    only, so an offset in the other direction is unreachable no matter how long
    the window -- and the lag scan put the optimum at -25 samples with a
    25-sample window, i.e. exactly at the edge the embedding cannot see past.
    Whether that is an acquisition trigger offset or real physics, the estimator
    has to be able to reach it before either claim can be tested.

    The `|lag|` columns that would otherwise wrap are edge-replicated rather
    than wrapped or dropped. Wrapping would splice the end of the record onto
    the start; dropping would change `n_t` and desynchronise every downstream
    consumer that indexes `case.X`. Replication leaves `|lag|` frames at one end
    of the record holding a repeated sensor sample -- 25 of 6085 at the default
    scan range -- and those frames are reported, not hidden.

    Args:
        S: Sensor record, shape (N_s, N_t).
        lag: Shift in PIV samples. Zero returns `S` unchanged.
        run_id: Source run per column, from `load_data`. Shifting is done
            within each run so a concatenated record does not leak sensor
            samples across the seam between two runs.

    Returns:
        The re-timed record, shape (N_s, N_t).
    """
    lag = int(lag)
    if lag == 0:
        return S
    S = np.asarray(S)
    n_t = S.shape[1]
    if abs(lag) >= n_t:
        raise ValueError(f"--force-lag {lag} is longer than the record ({n_t})")

    out = np.empty_like(S)
    segs = ([np.arange(n_t)] if run_id is None
            else [np.flatnonzero(run_id == k) for k in np.unique(run_id)])
    for idx in segs:
        blk = S[:, idx]
        if lag > 0:  # column t takes S[t - lag]: pad at the start
            out[:, idx] = np.concatenate(
                [np.repeat(blk[:, :1], lag, axis=1), blk[:, :-lag]], axis=1)
        else:        # column t takes S[t + |lag|]: pad at the end
            k = -lag
            out[:, idx] = np.concatenate(
                [blk[:, k:], np.repeat(blk[:, -1:], k, axis=1)], axis=1)

    side = "start" if lag > 0 else "end"
    print(f"  force lag: {lag:+d} PIV samples "
          f"({1e3 * lag / 250.0:+.1f} ms, {lag * 10:+d} force samples); "
          f"{abs(lag)} frames edge-replicated at the {side} of each run")
    return out


def load_data(args):
    """
    Loads one or more runs into a single (Q, S) pair.

    Invalid vectors are dropped rather than zero-filled, and both the linear
    estimators and the autoencoders see the same rows.

    Returns:
        (Q, S, unflat, case0, run_id, cases). Q is (N_x, N_t), S is (N_s, N_t),
        unflat maps a flat array back onto the grid with NaN holes, and run_id
        labels each column with its source run.
    """
    runs = args.runs or [args.run]
    cases = [build_case(r, n_snapshots=args.n, stride=args.stride,
                        highres=not getattr(args, "lowres", False),
                        sync_method=args.sync, drift_correct=not args.no_drift,
                        force_smooth=args.force_smooth, notch=args.notch,
                        notch_q=args.notch_q, probes=probe_points(args.probes),
                        n_probes=args.n_probes,
                        mask_tol=getattr(args, "mask_tol", 1.0),
                        mask_fill=getattr(args, "mask_fill", "interp"))
             for r in runs]
    case0 = cases[0]

    if len(cases) == 1:
        c = cases[0]
        c.X[:, :, ~c.fluid_mask] = np.nan  # mask on any-frame invalidity
        Q, S = c.flat(), c.S
        run_id = np.zeros(Q.shape[1], int)
        unflat = c.unflat
    else:
        Q, S, run_id = concat_cases(cases)
        mask = np.logical_and.reduce([c.fluid_mask for c in cases])
        for c in cases:
            c.X[:, :, ~mask] = np.nan
        case0.fluid_mask = mask
        unflat = case0.unflat

    S = apply_force_lag(S, getattr(args, "force_lag", 0), run_id)

    print(f"\n  Q {Q.shape}  S {S.shape}  ({len(cases)} run(s), "
          f"{Q.nbytes / 1e9:.2f} GB field)")
    return Q.astype(np.float64), S.astype(np.float64), unflat, case0, run_id, cases


def make_split(args, n_t, run_id, cases):
    """
    Builds a contiguous in-time split, or holds out a whole run for --cross-yaw.

    gap is raised to at least the delay window so no test input reaches into a
    training snapshot.

    Returns:
        (train_idx, test_idx, held_run). held_run is "" for a time split.
    """
    warmup = (max(args.delays) - 1) * args.delay_stride
    # The forward blocks of a two-sided embedding read past the end of the
    # record, where delay_embed zero-pads. Those columns are as unusable as the
    # zero-padded ones at the start, and unlike the start nothing else excludes
    # them -- so a two-sided run would otherwise train and score on K columns of
    # padding at every block boundary.
    cooldown = getattr(args, "delay_ahead", 0) * args.delay_stride
    gap = max(args.gap, warmup, cooldown)

    if args.cross_yaw:
        held = args.cross_yaw if args.cross_yaw in [c.run for c in cases] else cases[-1].run
        h = [i for i, c in enumerate(cases) if c.run == held][0]
        te, tr = np.flatnonzero(run_id == h), np.flatnonzero(run_id != h)
        # the record is discontinuous at each run seam, so every block needs its
        # own warm-up
        starts = [np.flatnonzero(run_id == i)[0] for i in range(len(cases))]
        drop = np.concatenate([np.arange(s, s + warmup) for s in starts])
        tr, te = np.setdiff1d(tr, drop), np.setdiff1d(te, drop)
        print(f"  cross-yaw: holding out {held}  (train {len(tr)}, test {len(te)})")
        return tr, te, held

    # Split each run separately and concatenate. For a single run this is
    # exactly the old behaviour; for several it is the difference between a
    # valid experiment and a broken one.
    #
    # A single contiguous cut across the *concatenated* record puts the whole
    # test block at the tail of the last run -- so the model trains on four
    # yaws and is tested on a fifth, which is `--cross-yaw` by accident and
    # without its safeguards. Worse, the delay embedding at each seam reaches
    # backwards into the previous run: the estimator is handed 25 samples of a
    # different flow as the history of this one.
    #
    # Per-run splitting fixes both. Every run contributes its own gap-separated
    # train and test blocks, the warm-up is dropped at each run's start so no
    # window spans a seam, and the cool-down is dropped at each run's end so no
    # forward lag reads past it.
    #
    # This answers "does more training data help", with every yaw represented on
    # both sides. `--cross-yaw` answers the harder question of generalising to a
    # yaw never seen. They are different experiments and neither substitutes.
    keys = list(np.unique(run_id))
    bounds = [(int(np.flatnonzero(run_id == k)[0]),
               int(np.flatnonzero(run_id == k)[-1]) + 1) for k in keys]

    tr_parts, te_parts = [], []
    for k, (s0, e0) in zip(keys, bounds):
        n = e0 - s0
        a, b = split_train_test(n, args.test_fraction, gap=gap, warmup=warmup)
        if cooldown:
            keep = n - cooldown
            a, b = a[a < keep], b[b < keep]
        tr_parts.append(a + s0)
        te_parts.append(b + s0)
        if len(keys) > 1:
            name = cases[int(k)].run if int(k) < len(cases) else f"run {k}"
            print(f"    {name}: train {len(a)}, test {len(b)}  "
                  f"(of {n} snapshots)")
    tr = np.concatenate(tr_parts)
    te = np.concatenate(te_parts)

    print(f"  split: train {len(tr)}, test {len(te)} over {len(keys)} run(s), "
          f"gap {gap}, warmup {warmup}"
          + (f", cooldown {cooldown}" if cooldown else ""))
    return tr, te, ""
