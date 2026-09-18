"""Projectors that reduce flow snapshots to latent coefficients.

Every projector implements `Projector`: `fit`, `encode`, `decode`,
`reconstruct`, and `score`.

    POD, SPOD        linear                               [pod.py]
    AE, CAE          dense and convolutional, PyTorch     [ae.py, cae.py]
    AEJax, CAEJax    dense and convolutional, JAX         [ae_jax.py, cae_jax.py]

Importing this package imports neither torch nor JAX. Each projector loads the
module that defines it on first access.

Typical usage example:

  from models.data_driven.autoencoders import AE, POD

  pod = POD(n_modes=20).fit(X)
  ae = AE(n_latent=20, hidden=(256, 64)).fit(X)
  errors = pod.score(X_test), ae.score(X_test)
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import Autoencoder, Projector

__all__ = ["AE", "CAE", "POD", "SPOD", "AEJax", "Autoencoder", "CAEJax", "Projector"]

_LAZY = {
    "POD": ".pod",
    "SPOD": ".pod",
    "AE": ".ae",
    "CAE": ".cae",
    "AEJax": ".ae_jax",
    "CAEJax": ".cae_jax",
}


def __getattr__(name: str) -> Any:
    """Imports a projector's module the first time the projector is accessed.

    Args:
      name: Attribute name.

    Returns:
      The named attribute of the module that defines it.

    Raises:
      AttributeError: If the package exports no attribute called `name`.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    """Lists the package's exports.

    Returns:
      The names in `__all__`, sorted.
    """
    return sorted(__all__)
