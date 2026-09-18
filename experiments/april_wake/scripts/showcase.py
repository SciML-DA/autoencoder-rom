#!/usr/bin/env python
"""Trains the best sparse-sensor pipelines in JAX and saves showcase material.

Two pipelines, each a reconstruction from the force balances and an LSTM that
forecasts the same latent space:

    podlse   PODLSEJax on 50 force delays, 64 POD modes, ridge 0.1 -- the best
             estimator of the sparse-sensor sweeps (test NMSE 0.61) -- and an
             LSTMJax on the 64 POD coefficients
    ae       AEJax with 8 latent modes and a linear BranchedAEJax on 25 force
             delays -- the best autoencoder (test NMSE 0.67) -- and an LSTMJax on
             the 8 latent codes

each trained on two datasets:

    normal   4p5d_10ms_yaw_0_0_0, the run every sweep used
    all      all five yaws, each split into its own train and test blocks

Every case is one moment `t0` in a test block. The frames run from `t0 - pre + 1`
to `t0 + horizon`:

    [ lead-in: pre frames, up to and including t0 ][ forecast: horizon frames ]

In the lead-in, the PIV and the reconstruction from forces play in real time.
At `t0` the LSTM, warmed up on the frames before `t0`, forecasts the next
`horizon` frames, as fast as it can compute them. Each step's wall time is
recorded, so the forecast can be replayed at computation speed while the PIV
waits, and then the PIV can play the same frames in real time to catch up.

Output, under `results/showcase/`:

    data/<run>.npz                  the full PIV record and force records of a run
      X             (2, N_t, Nx, Ny)  velocity [u, v] in m/s, NaN outside the mask
      fluid_mask    (Nx, Ny)
      x_mm, y_mm    grid coordinates
      pair_index    (N_t,)          PIV pair of each snapshot
      forces        (12, N_t)       drift-corrected forces, synchronised to the PIV
      forces_raw    (12, N_f)       the force file as recorded, at 2500 Hz
      force_index   (N_t,)          sample of forces_raw synchronous with each snapshot
      channel_names (12,)
    <dataset>/summary.csv           one row per pipeline: scores and timings
    <dataset>/cases/<run>_t<t0>.npz one case, keys below
    <dataset>/cases/<run>_t<t0>.json the same case's metadata
    <dataset>/gifs/*.gif            rendered previews

Case keys (`T = pre + horizon`, frame `pre - 1` is `t0`):

    time_s            (T,)            time relative to t0
    snapshot          (T,)            snapshot index within the run
    pair_index        (T,)
    truth             (2, T, Nx, Ny)  PIV
    forces            (12, T)         synchronised forces
    forces_raw        (12, 10 T)      raw forces over the same span
    forces_raw_time_s (10 T,)
    <p>/recon                     (2, T, Nx, Ny)        reconstruction from forces
    <p>/forecast_truth_init       (2, horizon, Nx, Ny)  LSTM warmed up on PIV latents
    <p>/forecast_sensor_init      (2, horizon, Nx, Ny)  LSTM warmed up on force-based latents
    <p>/z_true, <p>/z_recon       (r, T)                latent codes of the PIV and of the reconstruction
    <p>/z_forecast_truth_init, <p>/z_forecast_sensor_init    (r, horizon)
    <p>/nmse_recon                (T,)                  per-frame NMSE
    <p>/nmse_forecast_truth_init, <p>/nmse_forecast_sensor_init    (horizon,)
    <p>/forecast_wall_s           (horizon,)  wall time until each forecast frame
                                              was computed and decoded, from t0
    <p>/recon_latency_s           ()          time to reconstruct one frame
                                              from its force window

NMSE is squared error relative to the mean squared fluctuation about the
training mean, over the dataset's test blocks; 1 is no better than the mean.

The ESN is not used.

    qsub experiments/april_wake/hpc/showcase.pbs
    python experiments/april_wake/scripts/showcase.py --datasets normal --cases-per-run 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    import romda.utils  # noqa: F401
except ModuleNotFoundError:
    # `esn.py` imports `romda.utils` when `models.data_driven` is imported. The
    # ESN is not used here, so a placeholder module is enough to import the rest.
    _romda = types.ModuleType("romda")
    _utils = types.ModuleType("romda.utils")

    def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError("romda is not installed")

    _utils.mean_vector_to_ensemble = _unavailable  # type: ignore[attr-defined]
    _utils.normalized_time = _unavailable  # type: ignore[attr-defined]
    _romda.utils = _utils  # type: ignore[attr-defined]
    sys.modules["romda"], sys.modules["romda.utils"] = _romda, _utils

from experiments.april_wake import case_reader as cr  # noqa: E402
from experiments.april_wake.data_preprocessing import add_data_args, load_data, make_split  # noqa: E402
from field_estimation import AutoencoderLatentJax, BranchedAEJax, PODLSEJax, delay_embed  # noqa: E402
from models.data_driven.autoencoders import AEJax  # noqa: E402
from models.data_driven.forecasters import LSTMJax  # noqa: E402

#: Channel order of the two six-component balances, discs 2 and 3. From the
#: sweep's channel sets; confirm against the rig before labelling a figure.
CHANNEL_NAMES = [f"disc{d}_{c}" for d in (2, 3) for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]

#: Crops as fractions of the grid, `(x0, x1, y0, y1)`, for previews of areas.
CROPS = {
    "full": (0.0, 1.0, 0.0, 1.0),
    "near_wake": (0.0, 0.45, 0.0, 1.0),
    "far_wake": (0.55, 1.0, 0.0, 1.0),
    "centre": (0.25, 0.75, 0.2, 0.8),
}


# ── Pipelines ─────────────────────────────────────────────────────────────────


@dataclass
class Pipeline:
    """A reconstruction from forces and an LSTM on the same latent space.

    Attributes:
      name: Pipeline name.
      encode_true: Maps flat PIV fields to latent codes.
      decode: Maps latent codes to flat fields.
      recon_latent: Maps snapshot indices to latent codes estimated from forces.
      recon_one: Reconstructs one snapshot from its force window alone.
      lstm: The forecaster.
      info: Scores and timings for the summary.
    """

    name: str
    encode_true: Callable[[np.ndarray], np.ndarray]
    decode: Callable[[np.ndarray], np.ndarray]
    recon_latent: Callable[[np.ndarray], np.ndarray]
    recon_one: Callable[[int], np.ndarray]
    lstm: Any = None
    info: dict[str, Any] = field(default_factory=dict)


def build_podlse(Q: np.ndarray, S: np.ndarray, tr: np.ndarray, args: argparse.Namespace) -> Pipeline:
    """Fits POD-LSE on delay-embedded forces."""
    Sd = delay_embed(S, args.pod_delays, args.delay_stride)
    t0 = time.perf_counter()
    model = PODLSEJax(r_field=args.r_field, r_sensor=None, ridge=args.pod_ridge, seed=args.seed).fit(Q[:, tr], Sd[:, tr])
    Psi, q_mean = model.Psi, model.q_mean
    return Pipeline(
        name="podlse",
        encode_true=lambda Qc: Psi.T @ (Qc - q_mean),
        decode=lambda Z: Psi @ Z + q_mean,
        recon_latent=lambda idx: model.encode(Sd[:, idx]),
        recon_one=lambda i: model.predict(Sd[:, [i]]),
        info=dict(fit_recon_s=time.perf_counter() - t0, r=args.r_field),
    )


def build_ae(Q: np.ndarray, S: np.ndarray, tr: np.ndarray, unflat: Callable[..., np.ndarray], args: argparse.Namespace) -> Pipeline:
    """Fits the autoencoder, then the linear sensor branch into its latent space."""
    r = args.ae_latent
    t0 = time.perf_counter()
    ae = AEJax(
        n_latent=r,
        hidden=tuple(max(int(f * r), r + 1) for f in args.ae_hidden_scale),
        epochs=args.ae_epochs,
        batch_size=args.ae_batch,
        learning_rate=args.ae_lr,
        patience=args.ae_patience,
        seed=args.seed,
    ).fit(unflat(Q[:, tr], dtype=np.float32))
    ae_s = time.perf_counter() - t0
    latent = AutoencoderLatentJax(ae)
    t0 = time.perf_counter()
    branch = BranchedAEJax(
        latent,
        branch="linear",
        n_delays=args.branch_delays,
        delay_stride=args.delay_stride,
        epochs=args.branch_epochs,
        learning_rate=args.branch_lr,
        patience=args.branch_patience,
        batch_size=args.branch_batch,
        seed=args.seed,
    ).fit(Q, S, tr)
    nd = args.branch_delays * args.delay_stride

    def recon_one(i: int) -> np.ndarray:
        window = S[:, max(i - nd + 1, 0) : i + 1]
        return branch.predict(window, np.array([window.shape[1] - 1]))

    return Pipeline(
        name="ae",
        encode_true=lambda Qc: np.asarray(latent.encode(Qc), np.float64),
        decode=lambda Z: np.asarray(latent.decode(np.asarray(Z, np.float64)), np.float64),
        recon_latent=lambda idx: np.asarray(branch.encode(S, idx), np.float64),
        recon_one=recon_one,
        info=dict(fit_ae_s=ae_s, fit_recon_s=time.perf_counter() - t0, r=r, ae_epochs_run=ae.training_history.n_epochs_run),
    )


def train_lstm(pipe: Pipeline, Q: np.ndarray, tr: np.ndarray, args: argparse.Namespace) -> None:
    """Trains an LSTMJax on the latent codes of each contiguous training block."""
    segments = [pipe.encode_true(Q[:, block]).T for block in blocks(tr)]
    segments = [s for s in segments if len(s) > args.lstm_wash + args.lstm_seq_len + 1]
    lstm = LSTMJax(
        N_dim_in=segments[0].shape[1],
        N_units=args.lstm_units,
        seed=args.seed,
        N_wash=args.lstm_wash,
        seq_len=args.lstm_seq_len,
        epochs=args.lstm_epochs,
        lr=args.lstm_lr,
    )
    t0 = time.perf_counter()
    lstm.train(segments, verbose=False)
    pipe.lstm = lstm
    pipe.info |= dict(fit_lstm_s=time.perf_counter() - t0, lstm_val_best=float(np.min(lstm.training_history.val)))


def blocks(idx: np.ndarray) -> list[np.ndarray]:
    """Splits sorted indices into contiguous blocks."""
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.split(idx, cuts)


# ── Cases ─────────────────────────────────────────────────────────────────────


def forecast(pipe: Pipeline, Z_hist: np.ndarray, z_now: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts from a warm start, timing each step including its decode.

    Returns:
      The forecast latent codes, shape `(r, horizon)`, and the wall time from
      the start of the forecast until each frame was decoded.
    """
    lstm = pipe.lstm
    _, state = lstm.openLoop(Z_hist.T)
    Y, _ = lstm.closedLoop(z_now, horizon, state)

    x, s = z_now, state
    y, _ = lstm.closedLoop(x, 1, s)  # compiles the single-step rollout and decode before timing
    pipe.decode(y[0])
    wall = np.empty(horizon)
    t0 = time.perf_counter()
    for k in range(horizon):
        y, s = lstm.closedLoop(x, 1, s)
        pipe.decode(y[0])
        wall[k] = time.perf_counter() - t0
        x = y[0, :, 0]
    return Y[:, :, 0].T, wall


def run_case(
    pipes: list[Pipeline], Q: np.ndarray, S: np.ndarray, unflat: Callable[..., np.ndarray], col: int,
    run: cr.Case, local: int, var_ref: float, args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:  # fmt: skip
    """Computes one case: truth, reconstructions, forecasts, forces, and timings."""
    pre, H = args.pre, args.horizon
    cols = np.arange(col - pre + 1, col + H + 1)
    hist = np.arange(col - args.warm, col)
    dt = args.stride / cr.F_PIV_HZ
    Q_case = Q[:, cols]

    pairs = np.asarray(run.pairs)[local - pre + 1 : local + H + 1]
    f_idx = np.array([cr.force_index_for_pair(int(a)) for a in pairs])
    F_raw = run_raw_forces(run)
    span = np.arange(f_idx[0], f_idx[0] + cr.FORCE_PER_PIV * len(cols))
    span = span[span < F_raw.shape[1]]

    arrays: dict[str, np.ndarray] = dict(
        time_s=(np.arange(len(cols)) - (pre - 1)) * dt,
        snapshot=np.arange(local - pre + 1, local + H + 1),
        pair_index=pairs,
        truth=unflat(Q_case, dtype=args.field_dtype),
        forces=S[:, cols].astype(np.float32),
        forces_raw=F_raw[:, span].astype(np.float32),
        forces_raw_time_s=(span - f_idx[pre - 1]) / cr.F_FORCE_HZ,
    )
    meta: dict[str, Any] = dict(run=run.run, t0_snapshot=int(local), t0_pair=int(pairs[pre - 1]), dt=dt, pre=pre, horizon=H)

    def frame_nmse(Q_true: np.ndarray, Q_hat: np.ndarray) -> np.ndarray:
        return np.mean((Q_true - Q_hat) ** 2, axis=0) / var_ref

    for pipe in pipes:
        p = pipe.name
        Z_true = pipe.encode_true(Q[:, np.concatenate([hist, cols])])
        Z_hist, Z_case = Z_true[:, : len(hist)], Z_true[:, len(hist) :]
        Z_recon_all = pipe.recon_latent(np.concatenate([hist, cols]))
        Z_recon_hist, Z_recon = Z_recon_all[:, : len(hist)], Z_recon_all[:, len(hist) :]
        Q_recon = pipe.decode(Z_recon)

        z_now_true, z_now_recon = Z_case[:, pre - 1], Z_recon[:, pre - 1]
        Zf_true, wall = forecast(pipe, Z_hist, z_now_true, H)
        Zf_sensor, _ = forecast(pipe, Z_recon_hist, z_now_recon, H)
        Qf_true, Qf_sensor = pipe.decode(Zf_true), pipe.decode(Zf_sensor)
        future = Q_case[:, pre:]

        pipe.recon_one(int(col))  # compiles before timing
        reps = [timed(lambda: pipe.recon_one(int(col))) for _ in range(20)]

        arrays |= {
            f"{p}/recon": unflat(Q_recon, dtype=args.field_dtype),
            f"{p}/forecast_truth_init": unflat(Qf_true, dtype=args.field_dtype),
            f"{p}/forecast_sensor_init": unflat(Qf_sensor, dtype=args.field_dtype),
            f"{p}/z_true": Z_case.astype(np.float32),
            f"{p}/z_recon": Z_recon.astype(np.float32),
            f"{p}/z_forecast_truth_init": Zf_true.astype(np.float32),
            f"{p}/z_forecast_sensor_init": Zf_sensor.astype(np.float32),
            f"{p}/nmse_recon": frame_nmse(Q_case, Q_recon),
            f"{p}/nmse_forecast_truth_init": frame_nmse(future, Qf_true),
            f"{p}/nmse_forecast_sensor_init": frame_nmse(future, Qf_sensor),
            f"{p}/forecast_wall_s": wall,
            f"{p}/recon_latency_s": np.asarray(float(np.median(reps))),
        }
        meta[p] = dict(
            nmse_recon=float(np.mean(arrays[f"{p}/nmse_recon"])),
            nmse_forecast_truth_init=float(np.mean(arrays[f"{p}/nmse_forecast_truth_init"])),
            nmse_forecast_sensor_init=float(np.mean(arrays[f"{p}/nmse_forecast_sensor_init"])),
            forecast_ms_per_step=1e3 * float(wall[-1]) / H,
            recon_latency_ms=1e3 * float(np.median(reps)),
        )
    return arrays, meta


def timed(fn: Callable[[], Any]) -> float:
    """Returns the wall time of `fn()`."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


_RAW_FORCES: dict[str, np.ndarray] = {}


def run_raw_forces(run: cr.Case) -> np.ndarray:
    """Reads a run's force file as recorded, once."""
    if run.run not in _RAW_FORCES:
        _RAW_FORCES[run.run] = cr.read_dat(run.force_file)
    return _RAW_FORCES[run.run]


def package_run(run: cr.Case, out_dir: str, args: argparse.Namespace) -> None:
    """Writes a run's full PIV record and force records, unless already written."""
    path = os.path.join(out_dir, f"{run.run}.npz")
    if os.path.exists(path):
        return
    F_raw = run_raw_forces(run)
    mf = run.mf
    # Written under a temporary name and renamed, so two array jobs packaging
    # the same run never leave a half-written file.
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(
        tmp,
        allow_pickle=False,
        X=run.X.astype(args.field_dtype),
        fluid_mask=run.fluid_mask,
        x_mm=mf.x if mf is not None else np.arange(run.X.shape[2]),
        y_mm=mf.y if mf is not None else np.arange(run.X.shape[3]),
        pair_index=np.asarray(run.pairs),
        forces=run.S.astype(np.float32),
        forces_raw=F_raw.astype(np.float32),
        force_index=np.array([cr.force_index_for_pair(int(a)) for a in run.pairs]),
        channel_names=np.array(CHANNEL_NAMES),
        dt=np.asarray(args.stride / cr.F_PIV_HZ),
        force_hz=np.asarray(cr.F_FORCE_HZ),
    )
    os.replace(tmp, path)
    print(f"  packaged {run.run} -> {path}", flush=True)


# ── Previews ──────────────────────────────────────────────────────────────────


def storyline_gif(arrays: dict[str, np.ndarray], pipe: str, crop: str, title: str, out: str, args: argparse.Namespace) -> None:
    """Renders the lead-in, the forecast at computation speed, and the catch-up.

    Real-time segments play `args.slowmo` times slower than real time. The
    forecast segment plays at its computation speed on the same clock, several
    forecast frames to a GIF frame when a step is faster than a GIF frame.
    """
    from PIL import Image

    pre, H = args.pre, args.horizon
    dt = float(arrays["time_s"][1] - arrays["time_s"][0])
    truth = arrays["truth"][0].astype(np.float32)
    recon = arrays[f"{pipe}/recon"][0].astype(np.float32)
    fc = arrays[f"{pipe}/forecast_truth_init"][0].astype(np.float32)
    wall = arrays[f"{pipe}/forecast_wall_s"]
    x0, x1, y0, y1 = CROPS[crop]
    Nx, Ny = truth.shape[1:]
    sx, sy = slice(int(x0 * Nx), max(int(x1 * Nx), 1)), slice(int(y0 * Ny), max(int(y1 * Ny), 1))
    mean = np.nanmean(truth, axis=0)
    lim = float(np.nanpercentile(np.abs(truth - mean), 99)) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    panels = []
    for ax, label in zip(axes, ("PIV", "from forces", "LSTM forecast"), strict=True):
        im = ax.imshow(np.zeros((sy.stop - sy.start, sx.stop - sx.start)), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        panels.append(im)
    sup = fig.suptitle(title, fontsize=9)

    frames: list[Image.Image] = []
    durations: list[int] = []

    def draw(i_truth: int, i_recon: int, fc_frame: np.ndarray | None, caption: str, ms: float) -> None:
        for im, F in zip(panels, (truth[i_truth], recon[i_recon], fc_frame), strict=True):
            im.set_data(np.full((sy.stop - sy.start, sx.stop - sx.start), np.nan) if F is None else (F - mean)[sx, sy].T)
        sup.set_text(f"{title}\n{caption}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        durations.append(max(int(round(ms)), 20))

    real_ms = 1e3 * dt * args.slowmo
    for i in range(pre):
        draw(i, i, None, f"t = {arrays['time_s'][i] * 1e3:+.0f} ms   real time x1/{args.slowmo:g}", real_ms)
    k, prev = 0, 0.0
    while k < H:
        step = 1e3 * float(wall[k] - prev) * args.slowmo
        n = max(1, int(np.ceil(20.0 / max(step, 1e-6))))
        k = min(k + n, H)
        draw(pre - 1, pre - 1, fc[k - 1], f"forecasting {k}/{H} frames ({1e3 * dt * k:.0f} ms ahead) in {1e3 * wall[k - 1]:.1f} ms", 1e3 * float(wall[k - 1] - prev) * args.slowmo)
        prev = float(wall[k - 1])
    for k in range(H):
        draw(pre + k, pre + k, fc[k], f"PIV catching up: t = {arrays['time_s'][pre + k] * 1e3:+.0f} ms", real_ms)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def run_dataset(name: str, runs: list[str], args: argparse.Namespace) -> None:
    """Trains both pipelines on one dataset and writes its cases."""
    print(f"\n==== dataset {name}: {', '.join(runs)}", flush=True)
    args.run, args.runs = runs[0], runs if len(runs) > 1 else None
    args.delays = [max(args.pod_delays, args.branch_delays)]
    Q, S, unflat, _, run_id, cases = load_data(args)
    tr, te, _ = make_split(args, Q.shape[1], run_id, cases)

    data_dir = os.path.join(args.out, "data")
    out = os.path.join(args.out, name)
    for sub in ("cases", "gifs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    for case in cases:
        package_run(case, data_dir, args)

    q_train_mean = Q[:, tr].mean(axis=1, keepdims=True)
    var_ref = float(np.mean((Q[:, te] - q_train_mean) ** 2))

    pipes: list[Pipeline] = []
    for p in args.pipelines:
        print(f"  fitting {p} ...", flush=True)
        pipe = build_podlse(Q, S, tr, args) if p == "podlse" else build_ae(Q, S, tr, unflat, args)
        train_lstm(pipe, Q, tr, args)
        Q_hat = np.concatenate([pipe.decode(pipe.recon_latent(b)) for b in np.array_split(te, max(1, len(te) // 2000))], axis=1)
        pipe.info["recon_test_nmse"] = float(np.mean((Q[:, te] - Q_hat) ** 2) / var_ref)
        print(f"    {p}: reconstruction test NMSE {pipe.info['recon_test_nmse']:.3f}  {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in pipe.info.items()})}", flush=True)
        pipes.append(pipe)

    summaries: list[dict[str, Any]] = []
    starts = {int(k): int(np.flatnonzero(run_id == k)[0]) for k in np.unique(run_id)}
    for k in np.unique(run_id):
        te_k = te[run_id[te] == k]
        lo, hi = int(te_k[0]) + max(args.pre, args.warm), int(te_k[-1]) - args.horizon
        if hi <= lo:
            print(f"  !! test block of {cases[int(k)].run} too short for pre={args.pre}, warm={args.warm}, horizon={args.horizon}")
            continue
        for col in np.linspace(lo, hi, args.cases_per_run).astype(int):
            run = cases[int(k)]
            local = int(col) - starts[int(k)]
            arrays, meta = run_case(pipes, Q, S, unflat, int(col), run, local, var_ref, args)
            stem = f"{run.run}_t{local:05d}"
            np.savez(os.path.join(out, "cases", stem + ".npz"), allow_pickle=False, **arrays)
            with open(os.path.join(out, "cases", stem + ".json"), "w") as f:
                json.dump(meta | dict(crops=CROPS, channel_names=CHANNEL_NAMES, dataset=name), f, indent=2)
            summaries.append(meta)
            print(f"  case {stem}: " + "  ".join(f"{p.name} recon {meta[p.name]['nmse_recon']:.2f} forecast {meta[p.name]['nmse_forecast_truth_init']:.2f}" for p in pipes), flush=True)
            if not args.no_gifs:
                crops = ["full"] + (list(args.gif_crops) if col == np.linspace(lo, hi, args.cases_per_run).astype(int)[0] else [])
                for pipe in pipes:
                    for crop in crops:
                        title = f"{pipe.name.upper()} | {name} | {run.run} | t0 = snapshot {local}"
                        storyline_gif(arrays, pipe.name, crop, title, os.path.join(out, "gifs", f"{stem}_{pipe.name}_{crop}.gif"), args)

    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        fields = ["dataset", "pipeline", "r", "recon_test_nmse", "recon_latency_ms", "forecast_ms_per_step",
                  "nmse_forecast_truth_init", "nmse_forecast_sensor_init", "fit_ae_s", "fit_recon_s", "fit_lstm_s", "n_cases"]  # fmt: skip
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for pipe in pipes:
            per = [s[pipe.name] for s in summaries]
            row: dict[str, Any] = dict(dataset=name, pipeline=pipe.name, n_cases=len(per)) | pipe.info
            for key in ("recon_latency_ms", "forecast_ms_per_step", "nmse_forecast_truth_init", "nmse_forecast_sensor_init"):
                row[key] = float(np.median([s[key] for s in per])) if per else float("nan")
            writer.writerow(row)
    print(f"  summary -> {os.path.join(out, 'summary.csv')}", flush=True)


def main() -> None:
    """Parses the options and runs each dataset."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--datasets", nargs="+", default=["normal", "all"], choices=["normal", "all"])
    p.add_argument("--pipelines", nargs="+", default=["podlse", "ae"], choices=["podlse", "ae"])
    p.add_argument("--out", default="results/showcase")
    p.add_argument("--delay-stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("podlse")
    g.add_argument("--r-field", type=int, default=64)
    g.add_argument("--pod-delays", type=int, default=50)
    g.add_argument("--pod-ridge", type=float, default=0.1)
    g = p.add_argument_group("autoencoder and branch")
    g.add_argument("--ae-latent", type=int, default=8)
    g.add_argument("--ae-hidden-scale", type=float, nargs="+", default=[8, 2])
    g.add_argument("--ae-epochs", type=int, default=400)
    g.add_argument("--ae-batch", type=int, default=64)
    g.add_argument("--ae-lr", type=float, default=1e-3)
    g.add_argument("--ae-patience", type=int, default=40)
    g.add_argument("--branch-delays", type=int, default=25)
    g.add_argument("--branch-epochs", type=int, default=2000)
    g.add_argument("--branch-lr", type=float, default=1e-3)
    g.add_argument("--branch-patience", type=int, default=200)
    g.add_argument("--branch-batch", type=int, default=128)
    g = p.add_argument_group("lstm")
    g.add_argument("--lstm-units", type=int, default=128)
    g.add_argument("--lstm-epochs", type=int, default=300)
    g.add_argument("--lstm-seq-len", type=int, default=100)
    g.add_argument("--lstm-wash", type=int, default=50)
    g.add_argument("--lstm-lr", type=float, default=3e-3)
    g = p.add_argument_group("cases")
    g.add_argument("--cases-per-run", type=int, default=4)
    g.add_argument("--pre", type=int, default=75, help="lead-in frames, up to and including t0")
    g.add_argument("--horizon", type=int, default=75, help="forecast frames after t0")
    g.add_argument("--warm", type=int, default=200, help="frames the LSTM reads before t0")
    g.add_argument("--field-dtype", default="float16", choices=["float16", "float32"])
    g.add_argument("--no-gifs", action="store_true")
    g.add_argument("--gif-crops", nargs="*", default=["near_wake", "far_wake"], choices=list(CROPS))
    g.add_argument("--slowmo", type=float, default=10.0, help="real-time segments of the GIFs play this many times slower")
    args = p.parse_args()
    args.field_dtype = np.dtype(args.field_dtype)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({k: str(v) if isinstance(v, np.dtype) else v for k, v in vars(args).items()}, f, indent=2)
    datasets = {"normal": [args.run], "all": list(cr.RUNS)}
    for name in args.datasets:
        run_dataset(name, datasets[name], args)


if __name__ == "__main__":
    main()
