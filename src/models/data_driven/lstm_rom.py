"""LSTM-forecast latent ROMs: `POD_LSTM`, `AE_LSTM`, `CAE_LSTM`.

The other half of the projector x forecaster matrix. These exist mainly as the
acceptance test for `LatentROMMixin` / `SensorPlacementMixin`: if swapping the
forecaster had required changes inside the mixins, the `Forecaster` boundary
would have been drawn in the wrong place.

It very nearly held. Two names had to be generalised -- `state_labels` assumed
a single reservoir block (an LSTM carries ``h`` and ``c``), and `reset_case`
called ``reset_ESN``. Both are now asked of the forecaster
(``forecaster_state_labels``, ``reset_forecaster``); nothing else in either
mixin moved, and sensor placement, observables and latent bookkeeping were
reused untouched.

Each class below is its declaration and nothing else, which is the point.
"""

from __future__ import annotations

import numpy as np

from .autoencoders import POD
from .esn import phi_to_esn_layout
from .latent_rom import LatentROMMixin, SensorPlacementMixin
from .lstm import LSTM_model

__all__ = ["POD_LSTM", "AE_LSTM", "CAE_LSTM"]


class _LatentLSTM(LatentROMMixin, SensorPlacementMixin, LSTM_model):
    """Shared construction for the LSTM-forecast ROMs.

    Mirrors `ae_esn._AutoencoderESN` step for step, with `LSTM_model` in place
    of `ESN_model` -- deliberately, so the two are easy to diff.
    """

    _projector_cls: type = POD

    Nq = 10
    latent_symbol = "z"

    def __init__(
        self,
        data,
        dt,
        train_forecaster: bool = True,
        skip_sensor_placement: bool = False,
        domain_of_measurement=None,
        down_sample_measurement=None,
        plot_case: bool = False,
        **kwargs,
    ):
        for key in list(kwargs.keys()):
            if key in vars(type(self)):
                setattr(self, key, kwargs.pop(key))

        proj_kwargs = {
            k: kwargs.pop(k)
            for k in list(kwargs)
            if k in ("n_latent", "n_modes") or hasattr(self._projector_cls, k)
        }

        # ---- fit the projector ------------------------------------------
        self._projector_cls.__init__(self, **proj_kwargs)
        self.fit(data)

        Z = self.encode(data)
        self._latent_training_trajectory = Z
        self._z_mean = np.asarray(Z).mean(axis=1)

        # ---- train the forecaster on the latent trajectory ---------------
        if train_forecaster:
            LSTM_model.__init__(
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

    @property
    def latent_training_trajectory(self) -> np.ndarray:
        return self._latent_training_trajectory

    def spatial_basis(self, z0=None) -> np.ndarray:
        if z0 is None:
            z0 = getattr(self, "_z_mean", None)
        return self._projector_cls.spatial_basis(self, z0)


class POD_LSTM(_LatentLSTM, POD):
    """POD reduction + LSTM forecaster -- `POD_ESN` with the other forecaster."""

    _projector_cls = POD
    latent_symbol = "\\Phi"
    figs_folder: str = "figs/POD-LSTM/"


def __getattr__(name: str):
    """`AE_LSTM`/`CAE_LSTM` are resolved lazily so importing this module does
    not pull torch in for the POD-only case -- the same boundary the projector
    package and `models.data_driven` maintain."""
    if name in ("AE_LSTM", "CAE_LSTM"):
        from .autoencoders import AE, CAE

        projector = AE if name == "AE_LSTM" else CAE
        cls = type(
            name,
            (_LatentLSTM, projector),
            {
                "_projector_cls": projector,
                "figs_folder": f"figs/{name.replace('_', '-')}/",
                "__doc__": f"{projector.__name__} reduction + LSTM forecaster.",
                "__module__": __name__,
            },
        )
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
