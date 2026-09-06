"""The four autoencoder-reduced ROMs: `AE_ESN`, `CAE_ESN`, `AE_LSTM`, `CAE_LSTM`.

The nonlinear counterparts of `POD_ESN` / `POD_LSTM`. An autoencoder reduces the
field to a latent trajectory and a forecaster rolls that trajectory forward; the
forecast latents are decoded back to sensor readings by `get_observables`.

All four are declarations and nothing else -- the whole constructor lives in
`_base._LatentROM`, and sensor placement, observables, labels and resets in
`latent_rom`. That is the point of the arrangement: the projector x forecaster
matrix costs one class statement per cell, not one constructor per cell.

Torch lives behind this module. `AE`/`CAE` are resolved lazily by the projector
package, so importing *this* module does pull torch in -- but importing
``models.data_driven`` or naming only the POD-backed ROMs does not. `roms`
keeps these four behind its own lazy boundary for that reason.

Where the autoencoder ROMs differ from the POD ones
---------------------------------------------------
* **Sensor placement.** POD pivots QR on its spatial modes. An autoencoder has
  no such modes, so `Projector.spatial_basis` supplies the decoder Jacobian at
  the mean training latent -- the local linear basis of the decoder manifold.
  Sensor positions therefore depend on where in latent space the model is
  linearised; `_LatentROM` uses the mean of the training trajectory and stores
  it on ``_z_mean`` so the choice is inspectable rather than implicit.
* **Sensor readout.** POD reads sensors as ``Psi[idx] @ Z``, touching only the
  rows it needs. A dense decoder cannot produce a subset of its outputs, so
  `Projector.decode_at` decodes the whole field and indexes it. That is
  inherent to a nonlinear decoder, not an oversight.
"""

from __future__ import annotations

from ..autoencoders import AE, CAE
from ..forecasters import ESN_model, LSTM_model
from ..latent_rom import LatentROMMixin, SensorPlacementMixin
from ._base import _LatentROM

__all__ = ["AE_ESN", "CAE_ESN", "AE_LSTM", "CAE_LSTM"]


class AE_ESN(_LatentROM, LatentROMMixin, SensorPlacementMixin, ESN_model, AE):
    """Dense autoencoder + echo state network.

    ``AE`` reduces the field to ``n_latent`` coefficients through an MLP
    bottleneck; an ESN forecasts them. The nonlinear counterpart of `POD_ESN`,
    and the model this repository exists to compare against it at fixed latent
    size.
    """

    _projector_cls = AE
    _forecaster_cls = ESN_model
    figs_folder: str = "figs/AE-ESN/"

    extra_print_params = [*ESN_model.extra_print_params, "Nq", "measure_modes",
                          "N_latent"]


class CAE_ESN(_LatentROM, LatentROMMixin, SensorPlacementMixin, ESN_model, CAE):
    """Convolutional autoencoder + echo state network.

    As `AE_ESN`, but the encoder/decoder are Conv2d / ConvTranspose2d stacks
    over the spatial grid, so the reduction exploits spatial locality rather
    than treating the field as an unstructured vector.
    """

    _projector_cls = CAE
    _forecaster_cls = ESN_model
    figs_folder: str = "figs/CAE-ESN/"

    extra_print_params = [*ESN_model.extra_print_params, "Nq", "measure_modes",
                          "N_latent"]


class AE_LSTM(_LatentROM, LatentROMMixin, SensorPlacementMixin, LSTM_model, AE):
    """Dense autoencoder + LSTM forecaster -- `AE_ESN` with the other forecaster."""

    _projector_cls = AE
    _forecaster_cls = LSTM_model
    figs_folder: str = "figs/AE-LSTM/"

    extra_print_params = [*LSTM_model.extra_print_params, "Nq", "measure_modes",
                          "N_latent"]


class CAE_LSTM(_LatentROM, LatentROMMixin, SensorPlacementMixin, LSTM_model, CAE):
    """Convolutional autoencoder + LSTM forecaster."""

    _projector_cls = CAE
    _forecaster_cls = LSTM_model
    figs_folder: str = "figs/CAE-LSTM/"

    extra_print_params = [*LSTM_model.extra_print_params, "Nq", "measure_modes",
                          "N_latent"]
