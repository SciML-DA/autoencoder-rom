"""Data-driven reduced-order models: a `Projector` crossed with a forecaster.

Three subpackages, one per role:

    ``autoencoders/``  the projectors -- `Projector`, POD/SPOD, AE/CAE, the JAX pair
    ``forecasters/``   the forecasters -- `ESN_model`, `LSTM`, `LSTM_model`
    ``roms/``          the assembled models -- POD/AE/CAE x ESN/LSTM

plus `latent_rom`, the projector- and forecaster-agnostic plumbing the ROMs
share (`LatentROMMixin`, `SensorPlacementMixin`).

``autoencoders/`` is romda's own name and grouping; ``forecasters/`` and
``roms/`` are the symmetric ones added here, and are on the agenda to discuss
with A. Nóvoa rather than something her structure already prescribes.

Everything is re-exported at this level, so the historical import paths still
read the same::

    from models.data_driven import POD_ESN, AE_ESN, ESN_model

Import boundary
---------------
`AE_ESN`, `CAE_ESN`, `AE_LSTM` and `CAE_LSTM` are resolved lazily. Importing
them pulls in torch (via `autoencoders.ae`), and eager exports here would drag
torch into every ``import models`` -- including the POD-only paths that the
projector package deliberately keeps torch-free. Naming them still works
exactly as usual::

    from models.data_driven import AE_ESN     # imports torch, as it must
    from models.data_driven import POD_ESN    # does not
"""

from . import autoencoders, forecasters, roms
from .forecasters import LSTM, ESN_model, LSTM_model, phi_to_esn_layout
from .latent_rom import LatentROMMixin, SensorPlacementMixin
from .roms import POD_ESN, POD_LSTM

__all__ = [
    # subpackages
    "autoencoders",
    "forecasters",
    "roms",
    # forecasters
    "ESN_model",
    "LSTM_model",
    "LSTM",
    "phi_to_esn_layout",
    # the ROMs
    "POD_ESN",
    "POD_LSTM",
    "AE_ESN",
    "CAE_ESN",
    "AE_LSTM",
    "CAE_LSTM",
    # the projector x forecaster plumbing these are built from
    "LatentROMMixin",
    "SensorPlacementMixin",
]

#: torch-backed ROMs, deferred to `roms` (which defers to `autoencoders`)
_LAZY = ("AE_ESN", "CAE_ESN", "AE_LSTM", "CAE_LSTM")


def __getattr__(name: str):
    """PEP 562 lazy access -- keeps torch off the POD-only import path."""
    if name in _LAZY:
        return getattr(roms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
