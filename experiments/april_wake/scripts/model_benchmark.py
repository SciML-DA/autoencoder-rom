#!/usr/bin/env python
"""Trains every projector, the LSTM, and the LSTM ROMs on each yaw's PIV field.

A smoke benchmark rather than a study: each model trains briefly on one run, is
timed, and is scored on a held-out block of the same run.

    projectors   POD, SPOD, AE, CAE, AEJax, CAEJax
                 reconstruct the test block
    forecasters  LSTM and LSTMJax on the POD coefficients
                 forecasts the test block from a warm start
    ROMs         POD_LSTM, AE_LSTM, CAE_LSTM, AEJax_LSTM, CAEJax_LSTM
                 forecast the test block from a warm start

Each run is loaded from the PIV snapshots only; no force data is read. Points
invalid in every snapshot are masked, and the remaining dropouts are filled in
time, as `build_case` does.

For every model this writes a GIF of the truth, the reconstruction or forecast,
and their difference, with the NMSE in the title, and appends one row to
`results.csv`:

    fit_s          wall time of `fit`, or of construction for a ROM
    epochs         training epochs run
    s_per_epoch    fit_s / epochs. JAX figures include compilation.
    encode_ms      encode time per snapshot
    decode_ms      decode time per snapshot
    forecast_ms    forecast time per step
    nmse           mean squared error over the animated frames, divided by the
                   mean squared fluctuation of the truth about the training mean

The ESN and its ROMs are not run.

    RDS_ROOT=/path/to/april_experiment python experiments/april_wake/scripts/model_benchmark.py
    python experiments/april_wake/scripts/model_benchmark.py --runs 4p5d_10ms_yaw_0_0_0 --models POD AE AE_LSTM
    python experiments/april_wake/scripts/model_benchmark.py --n-snapshots 3000 --ae-epochs 100 --highres
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
import types
from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    import romda.utils  # noqa: F401
except ModuleNotFoundError:
    # `esn.py` imports `romda.utils` when `models.data_driven` is imported. The
    # ESN is not run here, so a placeholder module is enough to import the rest.
    _romda = types.ModuleType("romda")
    _utils = types.ModuleType("romda.utils")

    def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError("romda is not installed")

    _utils.mean_vector_to_ensemble = _unavailable  # type: ignore[attr-defined]
    _utils.normalized_time = _unavailable  # type: ignore[attr-defined]
    _romda.utils = _utils  # type: ignore[attr-defined]
    sys.modules["romda"], sys.modules["romda.utils"] = _romda, _utils

from datasets import split_indices  # noqa: E402
from experiments.april_wake import case_reader as cr  # noqa: E402

PROJECTORS = ("POD", "SPOD", "AE", "CAE", "AEJax", "CAEJax")
FORECASTERS = ("LSTM", "LSTMJax")
ROMS = ("POD_LSTM", "AE_LSTM", "CAE_LSTM", "AEJax_LSTM", "CAEJax_LSTM")
MODELS = PROJECTORS + FORECASTERS + ROMS

FIELDS = [
    "run", "model", "kind", "backend", "n_train", "n_test", "n_latent", "fit_s", "epochs",
    "s_per_epoch", "encode_ms", "decode_ms", "forecast_ms", "nmse", "frames", "gif", "error",
]  # fmt: skip


# ── Data ──────────────────────────────────────────────────────────────────────


class Run:
    """One yaw's PIV record, masked and split.

    Attributes:
      name: Run directory name.
      X: Velocity fields, shape `(2, N_t, Nx, Ny)`, NaN at masked points.
      fluid: Mask over the grid, `True` at kept points.
      tr: Training snapshot indices.
      te: Test snapshot indices.
      dt: Time between snapshots, in seconds.
      extent: Grid bounds for plotting, or `None`.
    """

    def __init__(self, name: str, args: argparse.Namespace) -> None:
        X = cr.load_run(name, highres=args.highres, stride=args.stride, max_snapshots=args.n_snapshots, root=args.root)
        invalid_frac = np.isnan(X).any(axis=0).mean(axis=0)
        self.fluid = invalid_frac < 1.0
        X = cr._fill_dropouts(X, self.fluid)
        X[:, :, ~self.fluid] = np.nan
        self.name = name
        self.X = X
        self.tr, _, self.te = split_indices(X.shape[1], val_frac=0, test_frac=args.test_frac, gap=args.gap)
        self.dt = args.stride / cr.F_PIV_HZ
        self.extent = _extent(args.root)
        self.q_mean = self.flat(X[:, self.tr]).mean(axis=1, keepdims=True)

    def flat(self, G: np.ndarray) -> np.ndarray:
        """Converts grid fields to the component-first flat layout."""
        Nu, n_t = G.shape[:2]
        return G.reshape(Nu, n_t, -1)[:, :, self.fluid.ravel()].transpose(0, 2, 1).reshape(-1, n_t)

    def grid(self, Q: np.ndarray) -> np.ndarray:
        """Converts flat fields back to the grid, with NaN at masked points."""
        Nu, _, Nx, Ny = self.X.shape
        n_t = Q.shape[1]
        out = np.full((Nu, n_t, Nx * Ny), np.nan, dtype=np.float32)
        out[:, :, self.fluid.ravel()] = Q.reshape(Nu, -1, n_t).transpose(0, 2, 1)
        return out.reshape(Nu, n_t, Nx, Ny)

    def nmse(self, Q_true: np.ndarray, Q_hat: np.ndarray) -> float:
        """Squared error relative to the fluctuation about the training mean."""
        err = np.mean((np.asarray(Q_true, np.float64) - Q_hat) ** 2)
        return float(err / np.mean((Q_true - self.q_mean) ** 2))


def _extent(root: str) -> tuple[float, float, float, float] | None:
    """Reads the grid bounds from the mean field, or returns `None` without one."""
    try:
        mf = cr.load_meanfield(os.path.join(root, cr.MEANFIELD))
    except Exception:
        return None
    return (float(mf.x.min()), float(mf.x.max()), float(mf.y.min()), float(mf.y.max()))


# ── Models ────────────────────────────────────────────────────────────────────


def projector(name: str, args: argparse.Namespace) -> Any:
    """Builds an unfitted projector."""
    from models.data_driven import autoencoders as ae

    r, seed = args.n_latent, args.seed
    train = dict(n_latent=r, epochs=args.ae_epochs, batch_size=args.batch, seed=seed)
    if name == "POD":
        return ae.POD(n_modes=r, random_state=seed)
    if name == "SPOD":
        return ae.SPOD(Nf=args.spod_nf, n_modes=r)
    if name == "AE":
        return ae.AE(**train, device=args.device)
    if name == "CAE":
        return ae.CAE(**train, channels=tuple(args.channels), device=args.device)
    if name == "AEJax":
        return ae.AEJax(**train)
    return ae.CAEJax(**train, channels=tuple(args.channels))


def rom_options(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Collects the constructor options of an LSTM ROM."""
    lstm = dict(N_units=args.lstm_units, N_wash=args.n_wash, forecaster_epochs=args.lstm_epochs, seq_len=args.seq_len)
    base = name.removesuffix("_LSTM")
    if base == "POD":
        proj: dict[str, Any] = dict(n_modes=args.n_latent, random_state=args.seed)
    else:
        proj = dict(n_latent=args.n_latent, projector_epochs=args.ae_epochs, batch_size=args.batch, seed=args.seed)
        if base in ("AE", "CAE"):
            proj["device"] = args.device
        if base in ("CAE", "CAEJax"):
            proj["channels"] = tuple(args.channels)
    return dict(Nq=args.nq, **proj, **lstm)


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Calls `fn` and returns its result and wall time in seconds."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_projector(name: str, run: Run, args: argparse.Namespace, fitted: dict[str, Any]) -> dict[str, Any]:
    """Fits a projector and reconstructs the test block."""
    model = projector(name, args)
    _, fit_s = timed(lambda: model.fit(run.X[:, run.tr]))
    fitted[name] = model
    history = getattr(model, "training_history", None)
    epochs = history.n_epochs_run if history is not None else 0

    X_te = run.X[:, run.te[: args.frames]]
    n = X_te.shape[1]
    model.decode(model.encode(X_te[:, :2]))  # compiles the JAX functions before timing
    Z, enc_s = timed(lambda: model.encode(X_te))
    Q_hat, dec_s = timed(lambda: np.asarray(model.decode(Z)))
    Q_true = run.flat(X_te)
    if Q_hat.shape != Q_true.shape:
        raise ValueError(f"{name} decodes {Q_hat.shape}, expected {Q_true.shape}; the flat layouts disagree")

    return dict(
        kind="projector",
        n_latent=model.N_latent,
        fit_s=fit_s,
        epochs=epochs,
        s_per_epoch=fit_s / epochs if epochs else float("nan"),
        encode_ms=1e3 * enc_s / n,
        decode_ms=1e3 * dec_s / n,
        forecast_ms=float("nan"),
        Q_true=Q_true,
        Q_hat=Q_hat,
    )


def run_lstm(name: str, run: Run, args: argparse.Namespace, fitted: dict[str, Any]) -> dict[str, Any]:
    """Trains an LSTM on POD coefficients and forecasts the test block."""
    from models.data_driven import forecasters

    pod = fitted.get("POD") or projector("POD", args).fit(run.X[:, run.tr])
    fitted["POD"] = pod
    Z_tr = pod.encode(run.X[:, run.tr])
    lstm = getattr(forecasters, name)(
        N_dim_in=Z_tr.shape[0],
        N_units=args.lstm_units,
        seed=args.seed,
        N_wash=args.n_wash,
        epochs=args.lstm_epochs,
        seq_len=args.seq_len,
    )
    _, fit_s = timed(lambda: lstm.train(Z_tr.T[np.newaxis], verbose=False))

    Z_warm, horizon, Q_true = warm_start(pod, run, args)
    _, state = lstm.openLoop(Z_warm[:, :-1].T)
    lstm.closedLoop(Z_warm[:, -1], horizon, state)  # compiles the JAX rollout before timing
    (Y, _), fc_s = timed(lambda: lstm.closedLoop(Z_warm[:, -1], horizon, state))
    Q_hat = np.asarray(pod.decode(Y[:, :, 0].T))

    return dict(
        kind="forecaster",
        n_latent=Z_tr.shape[0],
        fit_s=fit_s,
        epochs=lstm.training_history.n_epochs_run,
        s_per_epoch=fit_s / lstm.training_history.n_epochs_run,
        encode_ms=float("nan"),
        decode_ms=float("nan"),
        forecast_ms=1e3 * fc_s / horizon,
        Q_true=Q_true,
        Q_hat=Q_hat,
    )


def run_rom(name: str, run: Run, args: argparse.Namespace) -> dict[str, Any]:
    """Builds an LSTM ROM and forecasts the test block from a warm start."""
    import models.data_driven as dd

    cls = getattr(dd, name)
    rom, fit_s = timed(lambda: cls(data=run.X[:, run.tr], dt=run.dt, **rom_options(name, args)))
    try:
        Z_warm, horizon, Q_true = warm_start(rom, run, args)
        _, (h, c) = rom.openLoop(Z_warm[:, :-1].T)
        psi0 = rom.build_psi(u=Z_warm[:, -1:], h=h, c=c)
        rom.update_history(psi0[np.newaxis], t=np.array([0.0]), reset=True)
        (psi, _), fc_s = timed(lambda: rom.time_integrate(Nt=horizon))
        Z = psi[:, : rom.N_dim, 0].T
        Q_hat = np.asarray(rom.decode(Z))

        projector_epochs = rom.projector_history.n_epochs_run if rom.projector_history else 0
        forecaster_epochs = rom.forecaster_history.n_epochs_run if rom.forecaster_history else 0
        return dict(
            kind="rom",
            n_latent=rom.N_latent,
            fit_s=fit_s,
            epochs=f"{projector_epochs}+{forecaster_epochs}",
            s_per_epoch=float("nan"),
            encode_ms=float("nan"),
            decode_ms=float("nan"),
            forecast_ms=1e3 * fc_s / horizon,
            Q_true=Q_true,
            Q_hat=Q_hat,
        )
    finally:
        rom.close()


def warm_start(model: Any, run: Run, args: argparse.Namespace) -> tuple[np.ndarray, int, np.ndarray]:
    """Encodes the snapshots before the test block and the forecast targets.

    Returns:
      The latent states from `warm` snapshots before the test block up to its
      first snapshot, the forecast horizon, and the true flat fields over that
      horizon.
    """
    t0 = int(run.te[0])
    warm = min(args.warm, t0)
    horizon = min(args.frames, len(run.te) - 1)
    Z_warm = np.asarray(model.encode(run.X[:, t0 - warm : t0 + 1]), dtype=np.float64)
    Q_true = run.flat(run.X[:, t0 + 1 : t0 + 1 + horizon])
    return Z_warm, horizon, Q_true


# ── Output ────────────────────────────────────────────────────────────────────


def animate(run: Run, Q_true: np.ndarray, Q_hat: np.ndarray, title: str, out: str, fps: int) -> str:
    """Writes a GIF of the streamwise fluctuation: truth, model, and difference."""
    mean = run.grid(run.q_mean)[0, 0]
    T = run.grid(Q_true)[0] - mean
    P = run.grid(Q_hat)[0] - mean
    E = P - T
    lim = float(np.nanpercentile(np.abs(T), 99)) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    ims = []
    for ax, F, label in zip(axes, (T, P, E), ("PIV", "model", "difference"), strict=True):
        im = ax.imshow(
            F[0].T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=run.extent, interpolation="nearest"
        )
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)
    fig.colorbar(ims[0], ax=axes, fraction=0.02, pad=0.01, label="u'")
    sup = fig.suptitle(title, fontsize=10)

    def update(i: int) -> list[Any]:
        for im, F in zip(ims, (T, P, E), strict=True):
            im.set_data(F[i].T)
        sup.set_text(f"{title}   frame {i + 1}/{T.shape[0]}")
        return ims

    FuncAnimation(fig, update, frames=T.shape[0], blit=False).save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out


def append_row(path: str, row: dict[str, Any]) -> None:
    """Appends one row to the results CSV, writing the header first if needed."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    """Runs the benchmark and returns the number of models that failed."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=cr.ROOT, help="April experiment folder; defaults to $RDS_ROOT")
    p.add_argument("--runs", nargs="+", default=list(cr.RUNS))
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--out", default="results/model_benchmark")
    p.add_argument("--n-snapshots", type=int, default=1500)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--highres", action="store_true", help="read the (99, 159) snapshots instead of (55, 90)")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--gap", type=int, default=50)
    p.add_argument("--n-latent", type=int, default=8)
    p.add_argument("--ae-epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--spod-nf", type=int, default=10)
    p.add_argument("--lstm-epochs", type=int, default=10)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--n-wash", type=int, default=20)
    p.add_argument("--warm", type=int, default=100, help="snapshots the forecasters read before forecasting")
    p.add_argument("--nq", type=int, default=8, help="sensor points each ROM places")
    p.add_argument("--frames", type=int, default=80, help="frames animated, and the forecast horizon")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", default="cpu", help="torch device for AE and CAE")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "results.csv")
    failed: list[str] = []

    for run_name in args.runs:
        print(f"\n== {run_name}", flush=True)
        run = Run(run_name, args)
        print(f"   X {run.X.shape}, train {len(run.tr)}, test {len(run.te)}, fluid {int(run.fluid.sum())} points")
        run_dir = os.path.join(args.out, run_name)
        os.makedirs(run_dir, exist_ok=True)
        fitted: dict[str, Any] = {}

        for name in args.models:
            row: dict[str, Any] = dict(
                run=run_name,
                model=name,
                backend="jax" if "Jax" in name else "numpy" if name in ("POD", "SPOD", "LSTM", "POD_LSTM") else "torch",
                n_train=len(run.tr),
                n_test=len(run.te),
            )
            try:
                if name in PROJECTORS:
                    result = run_projector(name, run, args, fitted)
                elif name in FORECASTERS:
                    result = run_lstm(name, run, args, fitted)
                else:
                    result = run_rom(name, run, args)
                Q_true, Q_hat = result.pop("Q_true"), result.pop("Q_hat")
                nmse = run.nmse(Q_true, Q_hat)
                title = f"{name}  |  {run_name}  |  NMSE {nmse:.3f}"
                gif = animate(run, Q_true, Q_hat, title, os.path.join(run_dir, f"{name}.gif"), args.fps)
                row |= result | dict(nmse=nmse, frames=Q_true.shape[1], gif=gif)
                print(
                    f"   {name:12s} fit {row['fit_s']:7.1f}s  epochs {row['epochs']!s:>7}  "
                    f"NMSE {nmse:.3f}  -> {os.path.basename(gif)}",
                    flush=True,
                )
            except Exception as e:
                failed.append(f"{run_name}/{name}")
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"   {name:12s} FAILED  {row['error']}", flush=True)
                traceback.print_exc()
            append_row(csv_path, row)

    print(f"\nresults: {csv_path}")
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
