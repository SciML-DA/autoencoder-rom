"""`_LatentROM` -- the construction shared by every projector x forecaster ROM.

This was two classes. ``ae_esn._AutoencoderESN`` and ``lstm_rom._LatentLSTM``
were written a few days apart and their own docstrings admitted it: *"Mirrors
`ae_esn._AutoencoderESN` step for step, with `LSTM_model` in place of
`ESN_model` -- deliberately, so the two are easy to diff."* Two files that must
be diffed to stay correct are one file that has not been written yet, so here it
is, parameterised by the forecaster instead of by inheritance.

What a leaf class supplies
--------------------------
    _projector_cls   the `Projector` half   (POD, AE, CAE, ...)
    _forecaster_cls  the Model half         (ESN_model, LSTM_model)

and its own bases, which must be ``(_LatentROM, LatentROMMixin,
SensorPlacementMixin, <forecaster>, <projector>)``. `_LatentROM` carries no
Model base of its own precisely so that the two halves can vary: a single base
class cannot inherit from both `ESN_model` and `LSTM_model`.

Nothing else. `AE_ESN` is four lines, and adding a projector or a forecaster
costs leaf declarations rather than another copy of the constructor.

`POD_ESN` deliberately does not use this. It predates the whole projector x
forecaster generalisation, is the class romda's side recognises, and carries
plot/PDF logic none of the others have -- so it keeps its own constructor and
only shares the mixins. Folding it in would change the construction path of the
one ROM with results already on disk, for no structural gain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

__all__ = ["_LatentROM"]


class _LatentROM:
    """Fit the projector, train the forecaster on its latents, place sensors."""

    #: the `Projector` subclass this ROM reduces with; set by the leaf class
    _projector_cls: type
    #: the forecaster-as-Model this ROM forecasts with; set by the leaf class
    _forecaster_cls: type

    if TYPE_CHECKING:
        # Supplied by the sibling bases in the leaf class's MRO, not by this
        # class -- it deliberately has no Model or Projector base so that both
        # halves can vary. Declared (annotation only, no runtime effect) so the
        # contract this constructor depends on is checkable rather than
        # discovered at call time. `SensorPlacementMixin` documents the same
        # dependency in prose; this is the machine-readable half.
        fit: Callable[..., Any]                     # Projector
        encode: Callable[..., Any]                  # Projector
        N_latent: int                               # Projector
        measure_modes: bool                         # SensorPlacementMixin
        sensor_locations: Optional[np.ndarray]      # SensorPlacementMixin
        define_sensors: Callable[..., np.ndarray]   # SensorPlacementMixin

    # `domain_of_measurement` and `down_sample_measurement` are deliberately
    # NOT declared above. They are *properties* on `SensorPlacementMixin`, and
    # `_LatentROM` precedes that mixin in every leaf class's MRO -- so any
    # declaration here, attribute or property, reads to a type checker as each
    # leaf overriding the property. Left undeclared, they resolve through the
    # MRO at runtime and the setters below are annotated at the call site.

    Nq = 10
    perform_test = False
    latent_symbol = "z"

    def __init__(
        self,
        data,
        dt,
        train_forecaster: bool = True,
        train_ESN: Optional[bool] = None,
        skip_sensor_placement: bool = False,
        domain_of_measurement=None,
        down_sample_measurement=None,
        plot_case: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Snapshots, ``(Nu, N_t, Nx, Ny)`` -- the layout every `Projector`
            expects -- or already-flat ``(N_x, N_t)``.
        dt : float
            Time step of the snapshots.
        train_forecaster : bool
            Train the temporal half. False fits only the projector, which is
            what a latent-dimension sweep wants.
        train_ESN : bool, optional
            Deprecated alias for ``train_forecaster``, kept because callers
            predating `LSTM_model` pass it. Wins if given.
        skip_sensor_placement : bool
            Leave ``sensor_locations`` unset (``Nq`` becomes the latent size).
        **kwargs
            Forwarded to the projector, the forecaster and `Model`; e.g.
            ``n_latent``, ``grid_shape``, ``domain``, ``N_units``, ``Nq``.
        """
        if train_ESN is not None:
            train_forecaster = train_ESN

        for key in list(kwargs.keys()):
            if key in vars(type(self)):
                setattr(self, key, kwargs.pop(key))

        # Split construction kwargs by which half they configure. The
        # projectors take their own size argument plus a bag of training
        # hyperparameters; passing those on to `Model.__init__` makes it warn
        # about keys it cannot assign. Route each to its owner.
        #
        # ``n_modes`` as well as ``n_latent``: POD's constructor spells it the
        # first way and the autoencoders the second, and this base serves both.
        proj_kwargs = {
            k: kwargs.pop(k)
            for k in list(kwargs)
            if k in ("n_latent", "n_modes") or hasattr(self._projector_cls, k)
        }

        # ---- fit the projector ------------------------------------------
        self._projector_cls.__init__(self, **proj_kwargs)
        self.fit(data)

        # latent trajectory the forecaster is trained on, (N_latent, N_t)
        Z = self.encode(data)
        self._latent_training_trajectory = Z
        # where the decoder is linearised for sensor placement. Stored rather
        # than recomputed so the choice is inspectable rather than implicit.
        self._z_mean = np.asarray(Z).mean(axis=1)

        # ---- train the forecaster on it ---------------------------------
        if train_forecaster:
            from ..forecasters import phi_to_esn_layout

            # `_forecaster_cls` is `ESN_model` or `LSTM_model`, whose
            # `__init__` takes these; the annotation can only say `type`, so
            # the checker sees `type.__init__` and cannot know that.
            self._forecaster_cls.__init__(  # type: ignore[call-arg]
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
            self.domain_of_measurement = domain_of_measurement  # type: ignore[attr-defined]
            self.down_sample_measurement = down_sample_measurement  # type: ignore[attr-defined]
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
        """The latent coefficients the forecaster was trained on, ``(r, N_t)``."""
        return self._latent_training_trajectory

    def spatial_basis(self, z0: Optional[np.ndarray] = None) -> np.ndarray:
        """Projector's spatial basis, defaulting to the training-latent mean.

        `Projector.spatial_basis` linearises about the origin when given no
        ``z0``; for a trained autoencoder the origin is not where the data is,
        so the training mean is the meaningful default. `POD.spatial_basis`
        ignores ``z0`` and returns ``Psi``, so this is a no-op for POD.
        """
        if z0 is None:
            z0 = getattr(self, "_z_mean", None)
        return self._projector_cls.spatial_basis(self, z0)
