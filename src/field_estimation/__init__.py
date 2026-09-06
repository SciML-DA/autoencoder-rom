"""Field estimation from sparse sensors: recover a full field from a few channels.

The inverse of what ``models/`` does. A ROM takes a field, reduces it and rolls
it forward in time; this package takes a handful of instantaneous measurements
and estimates the field they came from, at one instant.

    linear      `PODLSE`, `ExtendedPOD` -- Borée's extended POD and linear
                stochastic estimation, the baseline           [epod.py]
    nonlinear   `BranchedAE` -- any `Projector` as encoder/decoder plus a
                learned sensor branch, trained against a frozen
                decoder on the latent loss              [branched_ae.py]
    support     `delay_embed` (causal lag stacking), `mode_observability` and
                `projection_floor` (the ceilings that say whether the basis,
                the sensors or the fit is the limit)          [epod.py]
    JAX         ports of both families      [epod_jax.py, branched_ae_jax.py]
    figures     `plots` -- spectra, observability, comparisons, animation.
                Not imported here: nothing in the estimators needs matplotlib,
                and a headless fit should not pay for it.

Not under ``models/`` deliberately. Everything there is, or composes, a
`dynamodels.Model` -- state, history, an integrator, a ``time_step``. Nothing
here has any of those; these are static maps from measurements to a field.

Rig-independent: the April campaign motivates the design choices and is named
in the comments that explain them, but every sampling rate, channel count and
frequency band lives in ``experiments/april_wake/``, never here.

Importing this package pulls in torch and jax eagerly, via ``branched_ae`` and
``branched_ae_jax``.
"""

from .epod_jax import PODLSEJax, ExtendedPODJax, pod_jax, ridge_cv_jax
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
from .branched_ae import (
    BranchedAE,
    LatentForecaster,
    LinearLatent,
    SensorBranch,
    TorchLatent,
)

__all__ = [
    # JAX ports of the sparse-sensor stack
    "PODLSEJax",
    "ExtendedPODJax",
    "pod_jax",
    "ridge_cv_jax",
    "BranchedAEJax",
    "LinearLatentJax",
    # sparse-sensor reconstruction: linear
    "PODLSE",
    "ExtendedPOD",
    "extended_pod",
    "delay_embed",
    "mode_observability",
    "projection_floor",
    "nmse",
    "ridge_cv",
    "split_train_test",
    # sparse-sensor reconstruction: nonlinear
    "BranchedAE",
    "SensorBranch",
    "LinearLatent",
    "TorchLatent",
    "LatentForecaster",
]
