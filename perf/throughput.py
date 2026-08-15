"""
throughput.py
=============

Two things the baseline could not answer, both needed before packing a
hyperparameter sweep onto a GPU.

**mem** -- real peak memory per config. The baseline's `peak_mem_mb` is a false
zero for JAX because `torch.cuda.max_memory_allocated` only sees torch's
allocator. JAX is asked for `memory_stats()` instead, with preallocation off:
by default JAX grabs ~75% of the card up front, which makes the number
meaningless *and* makes it impossible to run two JAX processes on one GPU.
XLA_PYTHON_CLIENT_PREALLOCATE=false is therefore both the measurement fix and
the packing fix.

**run** -- aggregate throughput with W independent workers sharing one GPU.
At 1-3% utilisation the card is mostly idle waiting on small kernels, so
several runs should overlap nearly for free. This measures how far that goes
before contention eats the gain.

    python perf/throughput.py mem
    python perf/throughput.py run --workers 1,2,4,6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# preallocation off before any jax import, or the setting is ignored
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

OUT = Path("perf/results")
EPOCHS = 60   # long enough that the timed fit dominates, not startup
# mid-size and large, dense and conv, both backends
PROBES = [
    ("AE", 64, "default"), ("CAE", 64, "default"),
    ("AEJax", 64, "default"), ("CAEJax", 64, "default"),
]


def build(dataset: str, model: str, latent: int, variant: str):
    from baseline import variant_spec
    from convergence_study import DATASETS, build_models, fit_data, pick_device
    from datasets import SPECS, load_snapshots, prepare_split

    X = load_snapshots(SPECS[dataset])
    X_train, X_val, _, _ = prepare_split(X)
    X_fit, val_fraction = fit_data(X_train, X_val)
    spec = variant_spec(DATASETS[dataset], model, variant)
    m = build_models(latent, 0, spec, pick_device(), val_fraction, n_epochs=EPOCHS)[model]
    return m, X_fit


def gpu_mem_mb(model: str) -> float:
    """Peak device memory, asked of whichever allocator actually owns it."""
    if "Jax" in model:
        import jax

        st = jax.local_devices()[0].memory_stats() or {}
        return float(st.get("peak_bytes_in_use", 0)) / 1e6
    import torch

    return torch.cuda.max_memory_allocated() / 1e6


def cmd_memworker(a) -> None:
    """Measure one config, alone in this process."""
    import torch

    m, X_fit = build(a.dataset, a.model, a.latent, "default")
    if "Jax" not in a.model:
        torch.cuda.reset_peak_memory_stats()
    m.fit(X_fit)
    print("RESULT " + json.dumps({"peak_mem_mb": gpu_mem_mb(a.model)}))


def cmd_mem(a) -> None:
    # one subprocess per config: JAX reports peak_bytes_in_use as a
    # process-cumulative high-water mark with no reset, so measuring several
    # configs in one process makes every one after the largest inherit its peak
    rows = []
    print(f"{'model':8s} {'latent':>6s} {'peak MB':>9s}  (preallocate=false, isolated process)")
    print("-" * 62)
    for model, latent, _variant in PROBES:
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        out = subprocess.run(
            [sys.executable, __file__, "memworker", "--dataset", a.dataset,
             "--model", model, "--latent", str(latent)],
            capture_output=True, text=True, env=env,
        ).stdout
        line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
        if not line:
            print(f"{model:8s} {latent:6d}   FAILED")
            continue
        mb = json.loads(line[0][7:])["peak_mem_mb"]
        rows.append(dict(model=model, latent=latent, peak_mem_mb=mb))
        print(f"{model:8s} {latent:6d} {mb:9.0f}")
    (OUT / "memory.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT / 'memory.json'}")


def cmd_worker(a) -> None:
    """One packed run.

    Loading bl is ~20 s and CUDA/XLA init is seconds more, so with a short fit
    most of a worker's wall time is startup -- and W workers reading the same
    4.4 GB file at once measures the filesystem, not the GPU. Everything
    expensive therefore happens *before* a filesystem barrier, and only the
    overlapped fit is timed.
    """
    m, X_fit = build(a.dataset, a.model, a.latent, "default")
    m.fit(X_fit)  # warmup: compile, autotune, page-in
    m2, _ = build(a.dataset, a.model, a.latent, "default")

    if a.barrier:
        bar = Path(a.barrier)
        (bar / f"ready.{os.getpid()}").write_text("1")
        deadline = time.time() + 300
        while len(list(bar.glob("ready.*"))) < a.expect and time.time() < deadline:
            time.sleep(0.05)

    t0 = time.perf_counter()
    m2.fit(X_fit)
    dt = time.perf_counter() - t0
    ran = int(getattr(m2, "n_epochs_run", EPOCHS) or EPOCHS)
    print("RESULT " + json.dumps({"elapsed": dt, "epochs": ran}))


def cmd_run(a) -> None:
    counts = [int(c) for c in a.workers.split(",")]
    results = []
    print(f"{'model':8s} {'latent':>6s} {'W':>3s} {'s/epoch':>9s} {'slowdown':>9s} {'throughput':>11s}")
    print("-" * 54)
    for model, latent, _v in PROBES:
        if a.model and model != a.model:
            continue
        solo = None
        for w in counts:
            bar = OUT / f"bar_{model}_{latent}_{w}"
            if bar.exists():
                for f in bar.glob("*"):
                    f.unlink()
            bar.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
            procs = [
                subprocess.Popen(
                    [sys.executable, __file__, "worker", "--dataset", a.dataset,
                     "--model", model, "--latent", str(latent),
                     "--barrier", str(bar), "--expect", str(w)],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True,
                )
                for _ in range(w)
            ]
            outs = [p.communicate()[0] for p in procs]
            eps = []
            for o in outs:
                line = [ln for ln in (o or "").splitlines() if ln.startswith("RESULT ")]
                if line:
                    eps.append(json.loads(line[0][7:]))
            for f in bar.glob("*"):
                f.unlink()
            bar.rmdir()
            if len(eps) != w:
                print(f"  {model}@{latent} w={w}: {len(eps)}/{w} workers returned -- skipped")
                continue
            # mean per-worker cost while all W overlap; startup is outside the timer
            per = sum(e["elapsed"] / max(e["epochs"], 1) for e in eps) / w
            solo = solo if solo is not None else per
            slow = per / solo
            results.append(dict(model=model, latent=latent, workers=w,
                                s_per_epoch=per, slowdown=slow, throughput=w / slow))
            print(f"{model:8s} {latent:6d} {w:3d} {per:9.3f} {slow:9.2f} {w / slow:10.2f}x",
                  flush=True)
    (OUT / "throughput.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'throughput.json'}")
    print("throughput = W / slowdown: aggregate epochs/s relative to one run alone")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["mem", "run", "worker", "memworker"])
    p.add_argument("--dataset", default="bl")
    p.add_argument("--model", default=None)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--workers", default="1,2,3,4")
    p.add_argument("--barrier", default=None)
    p.add_argument("--expect", type=int, default=1)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"mem": cmd_mem, "run": cmd_run, "worker": cmd_worker,
     "memworker": cmd_memworker}[a.cmd](a)
