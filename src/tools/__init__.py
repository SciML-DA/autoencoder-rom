"""Numerical tools: sparse-sensor reconstruction (extended POD / LSE and the
branched autoencoder), linear and JAX variants.

This package is now exactly the sparse-sensor stack. What used to also live
here has moved to where it belongs:

* the projectors and the POD/SPOD algorithms -> ``models.data_driven.autoencoders``
* the LSTM forecaster (``lstm_core``)         -> ``models.data_driven``

both to mirror romda's layout and because importing this package pulls in torch
and jax eagerly (via ``branched_ae*``), which would drag both frameworks into
the POD-only paths those modules deliberately keep clean.
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
