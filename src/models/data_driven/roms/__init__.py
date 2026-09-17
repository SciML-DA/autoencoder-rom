"""Latent reduced-order models.

Each ROM combines a projector with a forecaster:

    projector    ESN           LSTM
    POD          POD_ESN       POD_LSTM
    AE           AE_ESN        AE_LSTM
    CAE          CAE_ESN       CAE_LSTM
    AEJax        AEJax_ESN     AEJax_LSTM
    CAEJax       CAEJax_ESN    CAEJax_LSTM

`LatentROM` in `models.data_driven.latent_rom` holds the shared behavior.
Importing this package imports neither torch nor JAX; the autoencoder ROMs load
their module, and with it torch or JAX, on first access.

Typical usage example:

  from models.data_driven.roms import AE_ESN

  rom = AE_ESN(data=X, dt=0.01, n_latent=8, Nq=10)
"""

from __future__ import annotations

import importlib
from typing import Any

from .pod_esn import POD_ESN
from .pod_lstm import POD_LSTM

__all__ = [
    "AEJax_ESN",
    "AEJax_LSTM",
    "AE_ESN",
    "AE_LSTM",
    "CAEJax_ESN",
    "CAEJax_LSTM",
    "CAE_ESN",
    "CAE_LSTM",
    "POD_ESN",
    "POD_LSTM",
]

#: Module of each autoencoder ROM.
_LAZY = {
    "AE_ESN": ".autoencoder_roms",
    "AE_LSTM": ".autoencoder_roms",
    "CAE_ESN": ".autoencoder_roms",
    "CAE_LSTM": ".autoencoder_roms",
    "AEJax_ESN": ".autoencoder_roms_jax",
    "AEJax_LSTM": ".autoencoder_roms_jax",
    "CAEJax_ESN": ".autoencoder_roms_jax",
    "CAEJax_LSTM": ".autoencoder_roms_jax",
}


def __getattr__(name: str) -> Any:
    """Loads an autoencoder ROM the first time it is accessed.

    Args:
      name: Attribute name.

    Returns:
      The named ROM class.

    Raises:
      AttributeError: If the package exports no attribute called `name`.
    """
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_LAZY[name], __name__), name)


def __dir__() -> list[str]:
    """Lists the package's exports.

    Returns:
      The names in `__all__`, sorted.
    """
    return sorted(__all__)
