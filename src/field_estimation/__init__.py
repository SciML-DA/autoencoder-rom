# pyright: strict
"""Estimates a full flow field from a few sensor channels.

This package takes sensor measurements and estimates the field they
came from.

The package contains:
- `epod`: The linear estimators `PODLSE` and `ExtendedPOD`, plus
  `delay_embed`, `mode_observability`, and `projection_floor`.
- `branched_ae`: `BranchedAE`, which trains a sensor branch against the
  encoder and decoder of a `LinearLatent` or `TorchLatent`, and
  `LatentForecaster`.
- `epod_jax` and `branched_ae_jax`: JAX implementations of both families.
- `plots`: Figures for the estimators' results.

Importing this package imports torch and JAX, and enables 64-bit precision in
JAX.

Typical usage example:

  from field_estimation import PODLSE, delay_embed, split_train_test

  Sd = delay_embed(S, n_delays=25)
  tr, te = split_train_test(Q.shape[1], 0.25, gap=100, warmup=24)
  model = PODLSE(r_field=100, r_sensor=60, ridge=1e-4).fit(Q[:, tr], Sd[:, tr])
  model.score(Q[:, te], Sd[:, te])
"""

from .branched_ae import (
    BranchedAE,
    LatentForecaster,
    LinearLatent,
    SensorBranch,
    TorchLatent,
)
from .branched_ae_jax import BranchedAEJax, LinearLatentJax
from .epod import (
    PODLSE,
    ExtendedPOD,
    delay_embed,
    extended_pod,
    mode_observability,
    nmse,
    projection_floor,
    ridge_cv,
    split_train_test,
)
from .epod_jax import ExtendedPODJax, PODLSEJax, pod_jax, ridge_cv_jax

__all__ = [
    # JAX implementations.
    "PODLSEJax",
    "ExtendedPODJax",
    "pod_jax",
    "ridge_cv_jax",
    "BranchedAEJax",
    "LinearLatentJax",
    # Linear estimators.
    "PODLSE",
    "ExtendedPOD",
    "extended_pod",
    "delay_embed",
    "mode_observability",
    "projection_floor",
    "nmse",
    "ridge_cv",
    "split_train_test",
    # Nonlinear estimators.
    "BranchedAE",
    "SensorBranch",
    "LinearLatent",
    "TorchLatent",
    "LatentForecaster",
]
