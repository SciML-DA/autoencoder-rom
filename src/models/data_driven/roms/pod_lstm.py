"""POD reduction with an LSTM forecaster.

Typical usage example:

  rom = POD_LSTM(data=X, dt=0.01, n_modes=10, Nq=8, N_units=64, forecaster_epochs=40)
  psi, t = rom.time_integrate(Nt=200)
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from ..autoencoders.base import FloatArray
from ..autoencoders.pod import POD
from ..forecasters import LSTM_model
from ..latent_rom import LatentROM

__all__ = ["POD_LSTM"]


class POD_LSTM(LatentROM, LSTM_model, POD):
    """Proper orthogonal decomposition with an LSTM.

    POD reduces the field to `n_modes` temporal coefficients `Phi`, and the LSTM
    forecasts them. `LatentROM` documents construction and observables.
    """

    _projector_cls = POD
    _forecaster_cls = LSTM_model
    latent_symbol = "\\Phi"
    extra_print_params = [*LSTM_model.extra_print_params, "Nq", "measure_modes", "N_modes"]

    def _training_latents(self, data: npt.NDArray[Any]) -> FloatArray:
        """Returns the POD coefficients of the training snapshots.

        Args:
          data: The training snapshots.

        Returns:
          `Phi`, shape `(N_modes, N_t)`.
        """
        del data
        return self.Phi
