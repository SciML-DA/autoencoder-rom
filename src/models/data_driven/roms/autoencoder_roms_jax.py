"""JAX autoencoder reduction with an echo state network or LSTM forecaster.

Importing this module imports JAX.

Typical usage example:

  rom = CAEJax_ESN(data=X, dt=0.01, n_latent=8, channels=(16, 32), Nq=8, N_units=100)
  psi, t = rom.time_integrate(Nt=200)
"""

from __future__ import annotations

from ..autoencoders.ae_jax import AEJax
from ..autoencoders.cae_jax import CAEJax
from ..forecasters import ESN_model, LSTM_model
from ..latent_rom import LatentROM

__all__ = ["AEJax_ESN", "AEJax_LSTM", "CAEJax_ESN", "CAEJax_LSTM"]


class AEJax_ESN(LatentROM, ESN_model, AEJax):
    """Dense JAX autoencoder with an echo state network.

    `LatentROM` documents construction and observables.
    """

    _projector_cls = AEJax
    _forecaster_cls = ESN_model
    extra_print_params = [*ESN_model.extra_print_params, "Nq", "measure_modes", "N_latent"]


class CAEJax_ESN(LatentROM, ESN_model, CAEJax):
    """Convolutional JAX autoencoder with an echo state network.

    `LatentROM` documents construction and observables.
    """

    _projector_cls = CAEJax
    _forecaster_cls = ESN_model
    extra_print_params = [*ESN_model.extra_print_params, "Nq", "measure_modes", "N_latent"]


class AEJax_LSTM(LatentROM, LSTM_model, AEJax):
    """Dense JAX autoencoder with an LSTM.

    `LatentROM` documents construction and observables.
    """

    _projector_cls = AEJax
    _forecaster_cls = LSTM_model
    extra_print_params = [*LSTM_model.extra_print_params, "Nq", "measure_modes", "N_latent"]


class CAEJax_LSTM(LatentROM, LSTM_model, CAEJax):
    """Convolutional JAX autoencoder with an LSTM.

    `LatentROM` documents construction and observables.
    """

    _projector_cls = CAEJax
    _forecaster_cls = LSTM_model
    extra_print_params = [*LSTM_model.extra_print_params, "Nq", "measure_modes", "N_latent"]
