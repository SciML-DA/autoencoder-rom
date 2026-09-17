"""Autoencoder reduction with an echo state network or LSTM forecaster.

Importing this module imports torch.

Typical usage example:

  rom = CAE_ESN(data=X, dt=0.01, n_latent=8, channels=(16, 32), Nq=8, N_units=100)
  psi, t = rom.time_integrate(Nt=200)
"""

from __future__ import annotations

from ..autoencoders.ae import AE
from ..autoencoders.cae import CAE
from ..forecasters import ESN_model, LSTM_model
from ..latent_rom import LatentROM

__all__ = ["AE_ESN", "AE_LSTM", "CAE_ESN", "CAE_LSTM"]


class AE_ESN(LatentROM, ESN_model, AE):
    """Dense autoencoder with an echo state network.

    `LatentROM` documents construction and observables.
    """

    _projector_cls = AE
    _forecaster_cls = ESN_model
    extra_print_params = [*ESN_model.extra_print_params, "Nq", "measure_modes", "N_latent"]


class CAE_ESN(LatentROM, ESN_model, CAE):
    """Convolutional autoencoder with an echo state network.

    `LatentROM` documents construction and observables.
    """

    _projector_cls = CAE
    _forecaster_cls = ESN_model
    extra_print_params = [*ESN_model.extra_print_params, "Nq", "measure_modes", "N_latent"]


class AE_LSTM(LatentROM, LSTM_model, AE):
    """Dense autoencoder with an LSTM.

    `LatentROM` documents construction and observables.
    """

    _projector_cls = AE
    _forecaster_cls = LSTM_model
    extra_print_params = [*LSTM_model.extra_print_params, "Nq", "measure_modes", "N_latent"]


class CAE_LSTM(LatentROM, LSTM_model, CAE):
    """Convolutional autoencoder with an LSTM.

    `LatentROM` documents construction and observables.
    """

    _projector_cls = CAE
    _forecaster_cls = LSTM_model
    extra_print_params = [*LSTM_model.extra_print_params, "Nq", "measure_modes", "N_latent"]
