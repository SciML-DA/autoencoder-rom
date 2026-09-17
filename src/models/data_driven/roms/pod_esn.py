"""POD reduction with an echo state network forecaster.

Typical usage example:

  rom = POD_ESN(data=X, dt=0.01, n_modes=10, Nq=8, N_units=100)
  psi, t = rom.time_integrate(Nt=200)
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from ..autoencoders.base import FloatArray
from ..autoencoders.pod import POD
from ..forecasters import ESN_model
from ..latent_rom import LatentROM

__all__ = ["POD_ESN"]


class POD_ESN(LatentROM, ESN_model, POD):
    """Proper orthogonal decomposition with an echo state network.

    POD reduces the field to `n_modes` temporal coefficients `Phi`, and the ESN
    forecasts them. `LatentROM` documents construction and observables.
    """

    _projector_cls = POD
    _forecaster_cls = ESN_model
    latent_symbol = "\\Phi"
    extra_print_params = [*ESN_model.extra_print_params, "Nq", "measure_modes", "N_modes"]

    def get_POD_coefficients(self, Nt: int = 1) -> FloatArray:
        """Reads the latest POD coefficients from the model history.

        Args:
          Nt: Number of most recent time steps.

        Returns:
          Coefficients, shape `(N_modes, m)` for `Nt=1`, otherwise
          `(Nt, N_modes, m)`.
        """
        return self.get_latent_coefficients(Nt=Nt)

    def _training_latents(self, data: npt.NDArray[Any]) -> FloatArray:
        """Returns the POD coefficients of the training snapshots.

        Args:
          data: The training snapshots.

        Returns:
          `Phi`, shape `(N_modes, N_t)`.
        """
        del data
        return self.Phi
