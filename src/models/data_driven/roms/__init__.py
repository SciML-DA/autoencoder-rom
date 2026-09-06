"""The assembled reduced-order models: a `Projector` crossed with a `Forecaster`.

``autoencoders/`` holds the projectors and ``forecasters/`` the forecasters;
this package is where the two are joined. Every class here is a leaf
declaration -- the construction lives once in `_base._LatentROM` and the
projector-agnostic plumbing once in `latent_rom`.

    projector \\ forecaster    ESN          LSTM
    -----------------------   ---------    ----------
    POD                       POD_ESN      POD_LSTM
    AE                        AE_ESN       AE_LSTM
    CAE                       CAE_ESN      CAE_LSTM

`POD_ESN` is the exception: it predates the generalisation, is the class romda's
side recognises, and carries plot/PDF logic the others do not, so it keeps its
own constructor and shares only the mixins.

Import boundary
---------------
The four autoencoder-backed ROMs pull in torch (through `AE`/`CAE`) and are
resolved lazily, so ``import models`` and the POD-only ROMs stay torch-free --
the same boundary ``autoencoders/`` maintains. Naming them works as usual::

    from models.data_driven.roms import AE_ESN     # imports torch, as it must
    from models.data_driven.roms import POD_ESN    # does not
"""

from .pod_esn import POD_ESN
from .pod_lstm import POD_LSTM

__all__ = [
    "POD_ESN",
    "POD_LSTM",
    "AE_ESN",
    "CAE_ESN",
    "AE_LSTM",
    "CAE_LSTM",
]

_LAZY = {
    "AE_ESN": ".autoencoder_roms",
    "CAE_ESN": ".autoencoder_roms",
    "AE_LSTM": ".autoencoder_roms",
    "CAE_LSTM": ".autoencoder_roms",
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
