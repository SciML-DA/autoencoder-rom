"""
profile_ops.py
==============

Which kernels actually consume the time, per model, on bl.

Written after four mechanistic guesses (per-batch .item() sync, torch.compile,
NCHW->NHWC layout, adjoint reformulation of the transposed conv) each sounded
right and each was killed by measurement. This replaces guessing with the op
table: torch.profiler for AE/CAE, jax.profiler for the JAX pair.

    python perf/profile_ops.py --model CAE --latent 64
    python perf/profile_ops.py --all
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

OUT = Path("perf/results/profiles")
STEPS = 12


def setup(dataset: str, model: str, latent: int):
    from baseline import variant_spec
    from convergence_study import DATASETS, build_models, fit_data, pick_device
    from datasets import SPECS, load_snapshots, prepare_split

    X = load_snapshots(SPECS[dataset])
    a, b, _, _ = prepare_split(X)
    X_fit, vf = fit_data(a, b)
    spec = variant_spec(DATASETS[dataset], model, "default")
    m = build_models(latent, 0, spec, pick_device(), vf, n_epochs=2)[model]
    return m, X_fit


def profile_torch(dataset: str, model: str, latent: int) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    m, X_fit = setup(dataset, model, latent)
    m.fit(X_fit)  # warmup: builds the networks and pays cuDNN autotune

    # profile a short fit rather than a hand-rolled step, so what is measured is
    # the real training loop including its indexing and optimizer, not a
    # reconstruction of it that might differ
    m.n_epochs = 3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        m.fit(X_fit)
    torch.cuda.synchronize()
    tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}_{latent}_torch.txt").write_text(tbl)
    print(tbl)


def profile_jax(dataset: str, model: str, latent: int) -> None:
    import jax

    m, X_fit = setup(dataset, model, latent)
    m.fit(X_fit)  # warmup: XLA compile happens here, outside the trace

    tdir = OUT / f"{model}_{latent}_jax"
    tdir.mkdir(parents=True, exist_ok=True)
    m.n_epochs = 3
    with jax.profiler.trace(str(tdir)):
        m.fit(X_fit)

    # the trace is a gzipped chrome trace; aggregate GPU op durations by name
    files = glob.glob(str(tdir / "**" / "*.trace.json.gz"), recursive=True)
    if not files:
        print(f"  no trace written under {tdir}")
        return
    with gzip.open(sorted(files)[-1], "rt") as f:
        ev = json.load(f).get("traceEvents", [])
    tot: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    for e in ev:
        # only complete events on GPU-side tracks carry kernel durations
        if e.get("ph") != "X" or "dur" not in e:
            continue
        name = e.get("name", "?")
        if name.startswith(("thread", "process", "Steps", "XLA Modules")):
            continue
        tot[name] += e["dur"]
        cnt[name] += 1
    rows = sorted(tot.items(), key=lambda kv: -kv[1])[:18]
    grand = sum(tot.values()) or 1.0
    lines = [f"{'op':52s} {'total ms':>10s} {'%':>6s} {'calls':>7s}"]
    for n, us in rows:
        lines.append(f"{n[:52]:52s} {us / 1e3:10.2f} {100 * us / grand:5.1f}% {cnt[n]:7d}")
    txt = "\n".join(lines)
    (OUT / f"{model}_{latent}_jax.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bl")
    p.add_argument("--model", default="CAEJax")
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = ("AE", "CAE", "AEJax", "CAEJax") if a.all else (a.model,)
    for mdl in todo:
        print(f"\n{'=' * 70}\n{mdl} @ latent {a.latent} on {a.dataset}\n{'=' * 70}", flush=True)
        (profile_jax if "Jax" in mdl else profile_torch)(a.dataset, mdl, a.latent)
