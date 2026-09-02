"""Data-driven reduced-order models: a `Projector` crossed with a `Forecaster`.

``POD_ESN`` is linear-reduced, ``AE_ESN``/``CAE_ESN`` autoencoder-reduced; all
three share `LatentROMMixin` and `SensorPlacementMixin`, so they differ only in
which projector they reduce with.

``AE_ESN`` and ``CAE_ESN`` are resolved lazily. Importing them pulls in torch
(via `autoencoders.ae`), and eager exports here would drag torch into every
``import models`` -- including the POD-only paths that the projector package
deliberately keeps torch-free. Naming them still works exactly as usual:

    from models.data_driven import AE_ESN     # imports torch, as it must
    from models.data_driven import POD_ESN    # does not
"""

from .esn import ESN_model, phi_to_esn_layout
from .forecaster import Forecaster
from .latent_rom import LatentROMMixin, SensorPlacementMixin
from .lstm import LSTM_model
from .lstm_rom import POD_LSTM
from .pod_esn import POD_ESN

__all__ = [
    "ESN_model",
    "phi_to_esn_layout",
    "POD_ESN",
    "AE_ESN",
    "CAE_ESN",
    "LSTM_model",
    "POD_LSTM",
    "AE_LSTM",
    "CAE_LSTM",
    # the projector x forecaster plumbing these are built from
    "Forecaster",
    "LatentROMMixin",
    "SensorPlacementMixin",
]

_LAZY = {
    "AE_ESN": ".ae_esn",
    "CAE_ESN": ".ae_esn",
    "AE_LSTM": ".lstm_rom",
    "CAE_LSTM": ".lstm_rom",
}


def __getattr__(name: str):
    """PEP 562 lazy access -- keeps torch off the POD-only import path."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
