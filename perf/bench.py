"""Time the JAX and torch autoencoders on the real convergence-study path.

Naive timing here is actively misleading, in two ways that both flatter JAX:

1. JAX caches its JIT compilation *process-wide*, so a fit that pays for
   compilation and a later fit that inherits it are not comparable. Measured
   directly, a 30-epoch run came out faster in wall time than a 10-epoch one.
   Hence the throwaway warmup fit at each shape before the timed fit.
2. Compile cost is real but one-time, so folding it into a per-epoch average
   overstates steady-state cost at small epoch counts. It is reported
   separately instead.

`fit_time_s` in the study's own results CSV has neither correction, so it is
not a sound basis for comparing backends.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from convergence_study import (  # noqa: E402
    DATASETS,
    TRAIN_DEFAULTS,
    build_models,
    fit_data,
    pick_device,
    quiet,
)
from datasets import SPECS, load_snapshots, prepare_split  # noqa: E402

WARM, MEASURE = 2, 20
LATENTS = (8, 64, 256)
MODELS = tuple(os.environ.get("BENCH_MODELS", "AEJax,CAEJax").split(","))
FULL_LATENTS = (2, 4, 8, 16, 32, 64, 128, 256)
FULL_EPOCHS = TRAIN_DEFAULTS["n_epochs"]

dataset = os.environ.get("BENCH_DATASET", "bl")
spec = DATASETS[dataset]
device = pick_device()

import jax  # noqa: E402

print(f"torch={torch.__version__} device={device}")
print(f"jax={jax.__version__} backend={jax.default_backend()} {jax.devices()}", flush=True)

X = load_snapshots(SPECS[dataset])
X_train, X_val, X_test, meta = prepare_split(X)
X_fit, val_fraction = fit_data(X_train, X_val)
print(f"{dataset}: {X.shape}  n_train={meta['n_train']}\n", flush=True)


def timed(name: str, n_latent: int, n_epochs: int) -> tuple[float, int, float]:
    m = build_models(n_latent, 0, spec, device, val_fraction, n_epochs=n_epochs)[name]
    t0 = time.perf_counter()
    with quiet(True):
        m.fit(X_fit)
    dt = time.perf_counter() - t0
    ran = int(getattr(m, "n_epochs_run", n_epochs) or n_epochs)
    npar = float(getattr(m, "n_params", float("nan")))
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return dt, ran, npar


print(f"{'model':8s} {'latent':>7s} {'params':>12s} {'s/epoch':>9s} {'compile s':>10s}")
print("-" * 50, flush=True)

per_epoch: dict[tuple[str, int], float] = {}
for n_latent in LATENTS:
    for name in MODELS:
        # jax caches its JIT compilation process-wide, so a fit that pays for
        # compilation and one that inherits it are not comparable. burn a short
        # fit first at this exact shape, then measure only steady state.
        t_warm, r_warm, npar = timed(name, n_latent, WARM)
        t, ran, _ = timed(name, n_latent, MEASURE)
        s = t / max(ran, 1)
        compile_s = t_warm - s * max(r_warm, 1)  # what that first fit cost extra
        per_epoch[(name, n_latent)] = s
        print(
            f"{name:8s} {n_latent:7d} {npar:12,.0f} {s:9.3f} {compile_s:9.1f}",
            flush=True,
        )

print(f"\nprojection: {len(FULL_LATENTS)} latent sizes x {FULL_EPOCHS} epochs, seed 0")
xs = np.log2(np.array(LATENTS, dtype=float))
total = 0.0
for name in MODELS:
    ys = np.array([per_epoch[(name, k)] for k in LATENTS])
    sub = sum(float(np.interp(np.log2(k), xs, ys)) * FULL_EPOCHS for k in FULL_LATENTS)
    total += sub
    print(f"  {name:8s} {sub / 3600:6.2f} h")
print(f"  {'TOTAL':8s} {total / 3600:6.2f} h  ({'+'.join(MODELS)}, worst case)")
