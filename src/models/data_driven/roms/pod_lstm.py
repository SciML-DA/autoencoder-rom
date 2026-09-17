"""`POD_LSTM` -- POD reduction + LSTM forecaster.

`POD_ESN` with the other forecaster, and the reason `_base._LatentROM` is
parameterised rather than inherited. It exists mainly as the acceptance test for
`LatentROMMixin` / `SensorPlacementMixin`: if swapping the forecaster had
required changes inside the mixins, the forecaster boundary would have been
drawn in the wrong place.

It very nearly held. Two names had to be generalised -- `state_labels` assumed a
single reservoir block (an LSTM carries ``h`` and ``c``), and `reset_case`
called ``reset_ESN``. Both are now asked of the forecaster
(``forecaster_state_labels``, ``reset_forecaster``); nothing else in either
mixin moved, and sensor placement, observables and latent bookkeeping were
reused untouched.

Kept out of ``autoencoder_roms.py`` because it is torch-free: `POD` is pure
numpy, so naming this class must not drag torch into the process the way
`AE_LSTM` necessarily does.
"""

from __future__ import annotations

from ..autoencoders import POD
from ..forecasters import LSTM_model
from ..latent_rom import LatentROMMixin, SensorPlacementMixin
from ._base import _LatentROM

__all__ = ["POD_LSTM"]


class POD_LSTM(_LatentROM, LatentROMMixin, SensorPlacementMixin, LSTM_model, POD):
    """POD reduction + LSTM forecaster."""

    _projector_cls = POD
    _forecaster_cls = LSTM_model
    #: keep POD's historical \\Phi_j notation in figures rather than the generic z_j
    latent_symbol = "\\Phi"

    extra_print_params = [*LSTM_model.extra_print_params, "Nq", "measure_modes",
                          "N_modes"]
