"""Data-driven reduced-order models.

A latent ROM pairs a projector, which reduces flow snapshots to latent
coefficients, with a forecaster, which advances those coefficients in time.

    autoencoders/   projectors: POD, SPOD, AE, CAE, AEJax, CAEJax
    forecasters/    forecasters: ESN_model, LSTM, LSTMJax, LSTM_model
    roms/           ROMs: POD, AE, CAE, AEJax, and CAEJax, each with ESN or LSTM
    latent_rom.py   LatentROM, the base class the ROMs share

This package re-exports the forecasters and ROMs. Importing it imports neither
torch nor JAX; the autoencoder ROMs and `LSTMJax` load their modules on first
access.

Typical usage example:

  from models.data_driven import POD_ESN

  rom = POD_ESN(data=X, dt=0.01, n_modes=10, Nq=8)
  psi, t = rom.time_integrate(Nt=100)
"""

from __future__ import annotations

from typing import Any

from . import autoencoders, forecasters, roms
from .forecasters import LSTM, ESN_model, LSTM_model, phi_to_esn_layout
from .latent_rom import LatentROM
from .roms import POD_ESN, POD_LSTM

__all__ = [
    "AEJax_ESN",
    "AEJax_LSTM",
    "AE_ESN",
    "AE_LSTM",
    "CAEJax_ESN",
    "CAEJax_LSTM",
    "CAE_ESN",
    "CAE_LSTM",
    "LSTM",
    "LSTMJax",
    "POD_ESN",
    "POD_LSTM",
    "ESN_model",
    "LSTM_model",
    "LatentROM",
    "autoencoders",
    "forecasters",
    "phi_to_esn_layout",
    "roms",
]

_LAZY = ("AE_ESN", "CAE_ESN", "AE_LSTM", "CAE_LSTM", "AEJax_ESN", "CAEJax_ESN", "AEJax_LSTM", "CAEJax_LSTM")


def __getattr__(name: str) -> Any:
    """Loads an autoencoder ROM or `LSTMJax` the first time it is accessed.

    Args:
      name: Attribute name.

    Returns:
      The named class.

    Raises:
      AttributeError: If the package exports no attribute called `name`.
    """
    if name in _LAZY:
        return getattr(roms, name)
    if name == "LSTMJax":
        return forecasters.LSTMJax
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Lists the package's exports.

    Returns:
      The names in `__all__`, sorted.
    """
    return sorted(__all__)
