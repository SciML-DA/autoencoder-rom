"""
bench.py
========

Cluster benchmark of the JAX autoencoder rewrite against the code it replaced.

Four implementations of the same two models (dense AE, conv CAE), timed under
identical conditions on the same card:

    torch_orig   PyTorch before any perf work    e42c924^ src/tools/autoencoders.py
    jax_orig     JAX before the optimisation     e42c924^ src/tools/autoencoders_jax.py
    torch_new    PyTorch, current                src/models/data_driven/autoencoders/ae.py
    jax_new      JAX, current                    src/models/data_driven/autoencoders/{ae,cae}_jax.py

`stage.sh` assembles those into impls/ae_orig and impls/ae_new next to this file.

Measurement rules, carried over from the old perf/ work because each one was
learned from a wrong number:

* One process per (impl, model, latent). JAX's compile cache and allocator and
  torch's caching allocator never share a process, and JAX gets its default
  preallocation, i.e. the configuration it is actually run in.
* A warmup fit first, at the same shapes, so XLA JIT and cuDNN autotune are
  charged to `compile_s` rather than smeared into the per-epoch figure.
* Early stopping is disabled, so every timed fit runs exactly `--epochs`. The
  per-epoch time still includes the validation pass and best-weight snapshot,
  since those are part of real training.
* `--repeats` timed fits; the median is the figure, min/max are kept.
* The JAX backend is asserted to be on the GPU. The 7 Aug convergence results
  timed JAX on the CPU by accident, which is what inflated the old ~14x figure.

    python bench.py run --shard 0 --nshards 3 --out results/t4
    python bench.py report results/t4 results/a40
    python bench.py smoke     # synthetic data, CPU, 2 epochs: catches API drift before queueing
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "impls"))

IMPLS = ("torch_orig", "jax_orig", "torch_new", "jax_new")
MODELS = ("AE", "CAE")
LATENTS = (8, 32, 64, 128, 256)

# the bl convergence study's own hyperparameters (experiments/bl/scripts/convergence_study.py)
HIDDEN = lambda k: (8 * k, 2 * k)  # noqa: E731
CONV = {"channels": (16, 32, 64), "kernel_size": 3, "stride": 2, "pad": 1}
TRAIN = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "lr_factor": 0.5,
    "lr_patience": 10,
    "min_lr": 1e-5,
    "batch_size": 32,
    "seed": 0,
    "patience": 10**9,  # early stopping off: fixed epoch count
}


# ── model construction ────────────────────────────────────────────────────────


def epochs_and_val(m) -> tuple[int, list[float]]:
    """Epochs run and validation losses; the originals keep them as separate attributes."""
    history = getattr(m, "training_history", None)
    if history is not None:
        return history.n_epochs_run, list(history.val)
    return int(m.n_epochs_run), list(getattr(m, "val_loss_history", []))


def build(impl: str, model: str, k: int, n_epochs: int, val_fraction: float, device: str):
    kw = {**TRAIN, "n_epochs": n_epochs, "val_fraction": val_fraction}
    arch = {"layer_dims": HIDDEN(k)} if model == "AE" else dict(CONV)

    if impl.startswith("torch"):
        if impl == "torch_orig":
            from ae_orig import autoencoders as mod
        else:
            from ae_new.autoencoders import ae as mod
        cls = getattr(mod, model)
        kw = {**kw, **arch, "activation_function": "tanh", "device": device}
        # the torch classes drop unknown kwargs silently, so check here
        unknown = [x for x in kw if not hasattr(cls, x)]
        if unknown:
            raise TypeError(f"{impl}.{model} would silently drop {sorted(unknown)}")
        return cls(n_latent=k, **kw)

    if impl == "jax_orig":
        from ae_orig import autoencoders_jax as mod
    elif model == "AE":
        from ae_new.autoencoders import ae_jax as mod
    else:
        from ae_new.autoencoders import cae_jax as mod
    if model == "AE":
        arch = {"hidden": HIDDEN(k)}
    return getattr(mod, model + "Jax")(n_latent=k, activation="tanh", threshold=1e-4, **arch, **kw)


def check_backend(impl: str, device: str) -> str:
    """Return the device the framework will really use; fail if it is not `device`."""
    if impl.startswith("torch"):
        import torch

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("torch cannot see a GPU")
            return torch.cuda.get_device_name(0)
        return "cpu"
    import jax

    platforms = sorted({d.platform for d in jax.devices()})
    if device == "cuda" and not any(p in ("gpu", "cuda") for p in platforms):
        raise RuntimeError(f"JAX is on {platforms}, not the GPU -- this is the 7 Aug bug")
    return ",".join(platforms)


def sync(impl: str, device: str) -> None:
    if device == "cuda" and impl.startswith("torch"):
        import torch

        torch.cuda.synchronize()
    # JAX fits pull the loss to host every epoch, so they are already synced


def test_rel_error(m, X: np.ndarray) -> float:
    """Same metric as the convergence study's reconstruction_metrics."""
    try:
        Q = m.preprocess_snapshot(X)
        X_hat = m.reconstruct(X)
        return float(np.mean((Q + m.Q_mean - X_hat) ** 2) / np.mean(Q**2))
    except Exception as e:  # accuracy is a sanity check here, never a reason to lose a timing
        print(f"test_rel_error failed: {type(e).__name__}: {e}", file=sys.stderr)
        return float("nan")


# ── one config, in its own process ────────────────────────────────────────────


def cmd_worker(a) -> None:
    d = np.load(a.data)
    X_fit, X_test, vf = d["X_fit"], d["X_test"], float(d["val_fraction"])
    backend = check_backend(a.impl, a.device)

    def fit(n_epochs: int):
        m = build(a.impl, a.model, a.latent, n_epochs, vf, a.device)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):  # the JAX fits print per epoch
            m.fit(X_fit)
        sync(a.impl, a.device)
        return m, time.perf_counter() - t0

    if a.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    m, t_warm = fit(a.warm)
    warm_epochs = max(epochs_and_val(m)[0], 1)

    times, epochs = [], []
    for _ in range(a.repeats):
        del m
        m, t = fit(a.epochs)
        times.append(t)
        epochs.append(epochs_and_val(m)[0])

    s_ep = [t / max(e, 1) for t, e in zip(times, epochs)]
    med = statistics.median(s_ep)
    val = [float(v) for v in epochs_and_val(m)[1]]
    row = {
        "impl": a.impl,
        "model": a.model,
        "latent": a.latent,
        "backend": backend,
        "n_params": float(getattr(m, "n_params", float("nan")) or float("nan")),
        "epochs": epochs,
        "fit_s": times,
        "s_per_epoch": s_ep,
        "s_per_epoch_median": med,
        "warm_s": t_warm,
        "compile_s": t_warm - med * warm_epochs,
        "final_val_loss": val[-1] if val else float("nan"),
        "test_rel_error": test_rel_error(m, X_test),
    }
    print("RESULT " + json.dumps(row), flush=True)


# ── a shard of configs, one subprocess each ───────────────────────────────────


def prep(path: Path, synthetic: bool) -> dict:
    if synthetic:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((3, 120, 32, 24)).astype(np.float32)
        X[:, :, :3, :3] = np.nan  # a masked body, like bl's
        X_train, X_val, X_test = X[:, :72], X[:, 72:96], X[:, 96:]
        meta = {"synthetic": True}
    else:
        from datasets import SPECS, load_snapshots, prepare_split

        X = load_snapshots(SPECS["bl"])
        X_train, X_val, X_test, meta = prepare_split(X)
    X_fit = np.concatenate([X_train, X_val], axis=1)
    vf = X_val.shape[1] / X_fit.shape[1]
    np.savez(path, X_fit=X_fit, X_test=X_test, val_fraction=vf)
    return {
        "shape": list(X.shape),
        "n_x": int((~np.isnan(X[0, 0])).sum() * X.shape[0]),
        "n_fit": int(X_fit.shape[1]),
        "n_test": int(X_test.shape[1]),
        "val_fraction": vf,
        **{k: v for k, v in meta.items() if k in ("gap", "n_train", "n_val", "n_test", "synthetic")},
    }


def gpu_name() -> str:
    # asked of nvidia-smi rather than torch, so the parent never holds a CUDA context on the card
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        return out[0] if out else "none"
    except (OSError, subprocess.SubprocessError):
        return "none"


def versions() -> dict:
    code = "import json,sys,numpy,torch,jax;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'jax':jax.__version__}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    return json.loads(out.splitlines()[-1]) if out else {}


def cmd_run(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard{a.shard}.jsonl"

    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["impl"], r["model"], r["latent"]))

    groups = [(m, k) for m in a.models for k in a.latents]
    mine = groups[a.shard :: a.nshards]  # whole (model, latent) groups: all four impls share a card

    tmp = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    data = tmp / f"aebench_{os.environ.get('SLURM_JOB_ID', os.getpid())}_{a.shard}.npz"
    t0 = time.perf_counter()
    dmeta = prep(data, a.synthetic)
    env = {
        "host": socket.gethostname(),
        "gpu": gpu_name() if a.device == "cuda" else "cpu",
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        **versions(),
        "data": dmeta,
        "epochs": a.epochs,
        "warm_epochs": a.warm,
        "repeats": a.repeats,
    }
    if (HERE / "PROVENANCE").exists():
        env["provenance"] = (HERE / "PROVENANCE").read_text().strip()
    print(json.dumps(env, indent=1), f"\ndata prepared in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"shard {a.shard}/{a.nshards}: {mine}\n", flush=True)

    try:
        for model, k in mine:
            for impl in a.impls:
                if (impl, model, k) in done:
                    print(f"{impl:10s} {model:3s} k={k:<4d} already done, skipped", flush=True)
                    continue
                cmd = [
                    sys.executable, __file__, "worker", "--data", str(data), "--impl", impl,
                    "--model", model, "--latent", str(k), "--epochs", str(a.epochs),
                    "--warm", str(a.warm), "--repeats", str(a.repeats), "--device", a.device,
                ]
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    print(f"{impl:10s} {model:3s} k={k:<4d} TIMEOUT after {a.timeout}s", flush=True)
                    continue
                res = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
                if p.returncode != 0 or not res:
                    print(f"{impl:10s} {model:3s} k={k:<4d} FAILED (exit {p.returncode})", flush=True)
                    print("  " + "\n  ".join(p.stderr.strip().splitlines()[-15:]), flush=True)
                    continue
                row = {**json.loads(res[-1][7:]), **env}
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                s = row["s_per_epoch"]
                print(
                    f"{impl:10s} {model:3s} k={k:<4d} s/epoch={row['s_per_epoch_median']:.4f} "
                    f"(min {min(s):.4f} max {max(s):.4f})  compile={row['compile_s']:6.1f}s  "
                    f"val={row['final_val_loss']:.3e}  test_rel={row['test_rel_error']:.4f}  [{row['backend']}]",
                    flush=True,
                )
    finally:
        data.unlink(missing_ok=True)


# ── summary ───────────────────────────────────────────────────────────────────


def cmd_report(a) -> None:
    rows = []
    for d in a.dirs:
        for f in sorted(Path(d).glob("*.jsonl")):
            rows += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    if not rows:
        sys.exit("no results")

    by = {}
    for r in rows:
        by.setdefault((r["gpu"], r["model"], r["latent"]), {})[r["impl"]] = r

    def ratio(g, num, den):
        """Speedup of `den` over `num`: median, plus the least favourable pairing of repeats."""
        if num not in g or den not in g:
            return None, None
        med = g[num]["s_per_epoch_median"] / g[den]["s_per_epoch_median"]
        worst = min(g[num]["s_per_epoch"]) / max(g[den]["s_per_epoch"])
        return med, worst

    lines = [
        "| GPU | model | latent | torch_orig | jax_orig | torch_new | jax_new | jax_new vs torch_orig | jax_new vs jax_orig | jax_new vs torch_new |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (gpu, model, k), g in sorted(by.items()):
        cells = [f"{g[i]['s_per_epoch_median']:.3f}" if i in g else "--" for i in IMPLS]
        sp = []
        for base in ("torch_orig", "jax_orig", "torch_new"):
            med, worst = ratio(g, base, "jax_new")
            sp.append(f"{med:.2f}x (>= {worst:.2f}x)" if med else "--")
        lines.append(f"| {gpu.split(',')[0]} | {model} | {k} | " + " | ".join(cells + sp) + " |")
    print("s/epoch, median of repeats. Speedup in brackets is the least favourable pairing of repeats.\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worker")
    w.add_argument("--data", required=True)
    w.add_argument("--impl", choices=IMPLS, required=True)
    w.add_argument("--model", choices=MODELS, required=True)
    w.add_argument("--latent", type=int, required=True)
    w.add_argument("--epochs", type=int, default=20)
    w.add_argument("--warm", type=int, default=2)
    w.add_argument("--repeats", type=int, default=3)
    w.add_argument("--device", default="cuda")

    for name in ("run", "smoke"):
        r = sub.add_parser(name)
        r.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
        r.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
        r.add_argument("--out", default="results/bench" if name == "run" else "results/smoke")
        r.add_argument("--impls", type=lambda s: tuple(s.split(",")), default=IMPLS)
        r.add_argument("--models", type=lambda s: tuple(s.split(",")), default=MODELS)
        r.add_argument("--latents", type=lambda s: tuple(int(v) for v in s.split(",")), default=LATENTS)
        r.add_argument("--epochs", type=int, default=20)
        r.add_argument("--warm", type=int, default=2)
        r.add_argument("--repeats", type=int, default=3)
        r.add_argument("--timeout", type=int, default=3600)
        r.add_argument("--device", default="cuda")
        r.add_argument("--synthetic", action="store_true")

    rep = sub.add_parser("report")
    rep.add_argument("dirs", nargs="+")

    a = p.parse_args()
    if a.cmd == "smoke":
        a.synthetic, a.device, a.epochs, a.warm, a.repeats, a.latents = True, "cpu", 2, 1, 1, (8,)
        cmd_run(a)
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "worker":
        cmd_worker(a)
    else:
        cmd_report(a)
