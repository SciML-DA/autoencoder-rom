# regression golden

Reference loss curves for the **unoptimised** models, used to prove a perf
change did not move the numbers. Captured by `perf/regress.py capture`;
checked by `perf/regress.py compare`.

`golden.json` — per model/latent: full train and val loss history, test MSE,
epochs run, parameter count. Plus a `noise` block recording how much the
*untouched* code disagrees with itself across two identical runs.

Captured on `bl`, latents 8 and 64, seed 0, 40 epochs, on an NVIDIA T4.

## Why the noise block exists

GPU training is not automatically bit-reproducible, so "the loss changed by
1e-7" means nothing without knowing what the code already does to itself. The
measured floor splits cleanly by family:

| | run-to-run noise (relative) |
|---|---|
| `AE`, `AEJax` | **0.0** — bitwise reproducible |
| `CAE`, `CAEJax` | 6.4e-4 @ latent 8, ~1e-5 @ latent 64 |

Conv backward accumulates with atomics in both cuDNN and XLA, so its reduction
order varies between runs. Dense matmuls do not.

Consequences for validating an optimisation:

* **Dense** — the floor is zero, so `compare` falls back to `atol=1e-9`. Changes
  must be essentially exact, and any real numerical difference is caught.
* **Conv** — nothing can be verified tighter than ~1e-3 at latent 8. A conv
  change that shifts results by 1e-4 is indistinguishable from noise. Do not
  read a conv agreement of 1e-4 as proof of equivalence.

## Caveats

- **GPU-specific.** cuDNN and XLA pick algorithms by hardware heuristic, so this
  golden is only valid on a T4. Recapture on a different card.
- Covers latents 8 and 64 on `bl` only. A change that breaks solely at latent
  256, or solely on another dataset, would pass.
