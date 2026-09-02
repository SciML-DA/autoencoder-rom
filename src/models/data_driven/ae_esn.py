"""Autoencoder-projected echo state networks: `AE_ESN` and `CAE_ESN`.

The nonlinear counterparts of `POD_ESN`. An autoencoder reduces the field to a
latent trajectory and an `ESN_model` forecasts that trajectory in time; the
forecast latents are decoded back to sensor readings by `get_observables`.

The two classes are thin on purpose. Everything that does not depend on *which*
projector reduced the field lives in `latent_rom.LatentROMMixin` and
`latent_rom.SensorPlacementMixin`, which `POD_ESN` uses too -- so the linear and
nonlinear ROMs share one implementation of sensor placement, observables,
labels and resets rather than three copies of it.

Where this differs from `POD_ESN`
---------------------------------
* **Sensor placement.** POD pivots QR on its spatial modes. An autoencoder has
  no such modes, so `Projector.spatial_basis` supplies the decoder Jacobian at
  the mean training latent -- the local linear basis of the decoder manifold.
  Sensor positions therefore depend on where in latent space the model is
  linearised; the mean of the training trajectory is used, and is stored on
  ``_z_mean`` so the choice is inspectable rather than implicit.
* **Sensor readout.** POD reads sensors as ``Psi[idx] @ Z``, touching only the
  rows it needs. A dense decoder cannot produce a subset of its outputs, so
  `Projector.decode_at` decodes the whole field and indexes it. That is
  inherent to a nonlinear decoder, not an oversight.

Torch is imported only through `AE`/`CAE`, which the projector package resolves
lazily -- importing this module does pull it in, but importing `POD_ESN` or the
projector package does not.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .autoencoders import AE, CAE
from .esn import ESN_model, phi_to_esn_layout
from .latent_rom import LatentROMMixin, SensorPlacementMixin

__all__ = ["AE_ESN", "CAE_ESN", "_AutoencoderESN"]


class _AutoencoderESN(LatentROMMixin, SensorPlacementMixin, ESN_model):
    """Shared construction for the autoencoder-backed ROMs.

    Not used directly -- `AE_ESN` and `CAE_ESN` add the projector base class.
    Kept separate so the two leaf classes carry no logic beyond naming their
    projector, which is what makes adding `AE_LSTM` (or a third projector) a
    matter of declaring a class rather than editing one.
    """

    #: which `Projector` subclass this ROM reduces with; set by the leaf class
    _projector_cls: type = AE

    Nq = 10
    perform_test = False
    latent_symbol = "z"

    extra_print_params = [*ESN_model.extra_print_params, "Nq", "measure_modes",
                          "N_latent"]

    def __init__(
        self,
        data,
        dt,
        train_ESN: bool = True,
        skip_sensor_placement: bool = False,
        domain_of_measurement=None,
        down_sample_measurement=None,
        plot_case: bool = False,
        **kwargs,
    ):
        """Fit the autoencoder, train the ESN on its latent trajectory, place sensors.

        Parameters
        ----------
        data : np.ndarray
            Snapshots, ``(Nu, N_t, Nx, Ny)`` -- the layout every `Projector`
            expects -- or already-flat ``(N_x, N_t)``.
        dt : float
            Time step of the snapshots.
        train_ESN : bool
            Train the forecaster. False fits only the projector, which is what
            a latent-dimension sweep wants.
        skip_sensor_placement : bool
            Leave ``sensor_locations`` unset (``Nq`` becomes the latent size).
        **kwargs
            Forwarded to the projector, `ESN_model` and `Model`; e.g.
            ``n_latent``, ``grid_shape``, ``domain``, ``N_units``, ``Nq``.
        """
        for key in list(kwargs.keys()):
            if key in vars(type(self)):
                setattr(self, key, kwargs.pop(key))

        # Split construction kwargs by which half they configure. POD's
        # constructor takes the geometry positionally so POD_ESN can forward
        # everything blindly; the autoencoders take `n_latent` plus a bag of
        # training hyperparameters, and passing those on to `Model.__init__`
        # makes it warn about keys it cannot assign. Route each to its owner.
        proj_kwargs = {
            k: kwargs.pop(k)
            for k in list(kwargs)
            if k == "n_latent" or hasattr(self._projector_cls, k)
        }

        # ---- fit the projector ------------------------------------------
        self._projector_cls.__init__(self, **proj_kwargs)
        self.fit(data)

        # latent trajectory the ESN is trained on, (N_latent, N_t)
        Z = self.encode(data)
        self._latent_training_trajectory = Z
        # where the decoder is linearised for sensor placement -- see module
        # docstring. Stored rather than recomputed so it can be inspected.
        self._z_mean = np.asarray(Z).mean(axis=1)

        # ---- train the forecaster on it ---------------------------------
        if train_ESN:
            ESN_model.__init__(
                self,
                data=phi_to_esn_layout(Z),
                dt=dt,
                plot_training=plot_case,
                **kwargs,
            )

        # ---- place sensors ----------------------------------------------
        if self.measure_modes or skip_sensor_placement:
            self.Nq = self.N_latent
        elif self.sensor_locations is None:
            self.domain_of_measurement = domain_of_measurement
            self.down_sample_measurement = down_sample_measurement
            self.sensor_locations = self.define_sensors(
                N_sensors=self.Nq, plot=plot_case, z0=self._z_mean
            )
            self.Nq = len(self.sensor_locations)
        else:
            self.Nq = len(self.sensor_locations)

        print(f"========= {type(self).__name__} model complete =========")

    # ---- what LatentROMMixin needs ---------------------------------------

    @property
    def latent_training_trajectory(self) -> np.ndarray:
        """The latent coefficients the ESN was trained on, ``(N_latent, N_t)``."""
        return self._latent_training_trajectory

    def spatial_basis(self, z0: Optional[np.ndarray] = None) -> np.ndarray:
        """Decoder Jacobian, defaulting to the mean of the training latents.

        `Projector.spatial_basis` linearises about the origin when given no
        ``z0``; for a trained autoencoder the origin is not where the data is,
        so the training mean is the meaningful default here.
        """
        if z0 is None:
            z0 = getattr(self, "_z_mean", None)
        return self._projector_cls.spatial_basis(self, z0)


class AE_ESN(_AutoencoderESN, AE):
    """Dense autoencoder + echo state network.

    ``AE`` reduces the field to ``n_latent`` coefficients through an MLP
    bottleneck; an ESN forecasts them. The nonlinear counterpart of `POD_ESN`,
    and the model this repository exists to compare against it at fixed latent
    size.
    """

    _projector_cls = AE
    figs_folder: str = "figs/AE-ESN/"


class CAE_ESN(_AutoencoderESN, CAE):
    """Convolutional autoencoder + echo state network.

    As `AE_ESN`, but the encoder/decoder are Conv2d / ConvTranspose2d stacks
    over the spatial grid, so the reduction exploits spatial locality rather
    than treating the field as an unstructured vector.
    """

    _projector_cls = CAE
    figs_folder: str = "figs/CAE-ESN/"
