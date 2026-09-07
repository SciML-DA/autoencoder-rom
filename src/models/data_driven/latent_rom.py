"""Projector-agnostic machinery shared by every latent ROM.

A latent ROM is a `Projector` (POD, SPOD, AE, CAE, ...) that reduces a field to
a handful of coefficients, plus a `Forecaster` (ESN, LSTM) trained to roll those
coefficients forward. Everything in this module is the part that does not care
which of either it got:

`SensorPlacementMixin`
    Where to put sensors, and what the measurement grid is. Lifted verbatim out
    of `POD_ESN`, with the single POD-specific line -- QR-pivoting on ``Psi`` --
    replaced by ``self.spatial_basis(...)``. `POD.spatial_basis` returns ``Psi``
    unchanged, so placement is bit-identical to before the lift.

`LatentROMMixin`
    Latent <-> forecaster plumbing: reading latent coefficients out of the model
    history, decoding them at the sensors, labels, and resets.

Both are written against the `Forecaster` protocol rather than `ESN_model`, so
adding `LSTM_model` costs leaf classes and not edits here. The leaf classes and
the constructor they share live in ``roms/``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

__all__ = ["SensorPlacementMixin", "LatentROMMixin"]


class SensorPlacementMixin:
    """Chooses sensor locations on the projector's spatial basis.

    Expects the host to provide (from its `Projector` half) ``grid_shape``,
    ``domain``, ``fluid_mask_flat``, ``_to_physical_grid`` and
    ``spatial_basis``, and (from its `Model` half) ``rng``.
    """

    measure_modes = False
    sensor_locations = None
    qr_selection = True

    @property
    def N_sensors(self):
        if self.measure_modes:
            return 0
        return int(self.Nq / 2)

    # ---- measurement domain -------------------------------------------------

    @property
    def domain_of_measurement(self):
        if not hasattr(self, "_domain_of_measurement"):
            self._domain_of_measurement = self.domain
        return self._domain_of_measurement

    @domain_of_measurement.setter
    def domain_of_measurement(self, dom):
        self._domain_of_measurement = dom  # type: list

    @property
    def down_sample_measurement(self):
        if not hasattr(self, "_down_sample_measurement"):
            self.down_sample_measurement = None
        return self._down_sample_measurement

    @down_sample_measurement.setter
    def down_sample_measurement(self, dsm):
        if dsm is not None:
            if isinstance(dsm, int):
                dsm = [dsm, dsm]
            elif not (isinstance(dsm, list) and len(dsm) == 2 and all(isinstance(x, int) for x in dsm)):
                raise ValueError()
        self._down_sample_measurement = dsm

    @property
    def grid_of_measurement(self):
        Nx, Ny = self.grid_shape[1:]

        if self.domain == self.domain_of_measurement or self.domain_of_measurement is None:
            x_idx = np.arange(Nx)
            y_idx = np.arange(Ny)
        else:
            x_min, x_max, y_min, y_max = self.domain
            doi_x_min, doi_x_max, doi_y_min, doi_y_max = self.domain_of_measurement

            # Generate 1D spatial grids for original domain
            x = np.linspace(x_min, x_max, Nx)
            y = np.linspace(y_min, y_max, Ny)

            # Find indices within domain_of_interest along each axis
            x_idx = np.where((x >= doi_x_min) & (x <= doi_x_max))[0]
            y_idx = np.where((y >= doi_y_min) & (y <= doi_y_max))[0]

        if len(x_idx) == 0 or len(y_idx) == 0:
            raise ValueError("Domain of interest does not overlap with original domain grid.")

        down_sample = self.down_sample_measurement
        if down_sample is not None:
            step_x, step_y = down_sample
            x_idx = x_idx[::step_x]
            y_idx = y_idx[::step_y]

        grid = np.ravel_multi_index(np.ix_(x_idx, y_idx), dims=(Nx, Ny))
        # remove the idx not on fluid
        grid_idx_fluid = np.where(self.fluid_mask_flat)[0]
        grid_idx_fluid = np.intersect1d(grid_idx_fluid, grid)

        # append the idx for the other variables (e.g., uy) if needed
        if self.grid_shape[0] > 1:
            grid_idx_fluid = np.concatenate(
                [grid_idx_fluid + Nx * Ny * i for i in range(self.grid_shape[0])],
                axis=None,
            )

        return grid_idx_fluid

    # ---- placement ----------------------------------------------------------

    def select_sensors(
        self,
        measure_modes=False,
        domain_of_measurement=None,
        down_sample_measurement=None,
        N_sensors=None,
        qr_selection=False,
        plot=False,
    ):
        self.measure_modes = measure_modes
        if measure_modes:
            self.Nq = self.N_latent
            self.sensor_locations = None
        else:
            self.domain_of_measurement = domain_of_measurement
            self.down_sample_measurement = down_sample_measurement
            self.qr_selection = qr_selection
            self.sensor_locations = self.define_sensors(N_sensors=N_sensors, plot=plot)
            self.Nq = len(self.sensor_locations)

    def define_sensors(self, N_sensors=None, plot=False, z0=None):
        """Place ``N_sensors`` sensors by QR column-pivoting on the projector's
        spatial basis, or at random when ``qr_selection`` is off.

        ``plot`` defaults to False: this used to call ``plt.show()``
        unconditionally, which blocks a batch job on any interactive backend.
        ``z0`` is forwarded to `Projector.spatial_basis` -- ignored by POD,
        used by the autoencoders to choose where to linearise the decoder.
        """
        Nu, Nx, Ny = self.grid_shape
        measure_grid_idx = np.asarray(self.grid_of_measurement)

        one_dom = measure_grid_idx[
            measure_grid_idx < Nx * Ny
        ]  # only the first variable (e.g., ux) for sensor placement

        if N_sensors is None:
            N_sensors = self.N_sensors

        if self.qr_selection:
            # the only line that used to be POD-specific: `self.Psi` became
            # `self.spatial_basis(z0)`. POD returns Psi, so this is a no-op for
            # POD_ESN; an autoencoder returns its local decoder Jacobian.
            basis = self.spatial_basis(z0)

            basis = self._to_physical_grid(basis).transpose(1, 0, 2, 3)  # (r, Nu, Nx, Ny)

            basis = np.nan_to_num(basis, nan=0.0)
            basis = basis.reshape(basis.shape[0], Nu, Nx * Ny)

            # choose one variable block for placement, e.g. variable 0
            A = basis.reshape(basis.shape[0], -1)  # shape (n_candidates, r)
            A = A[:, measure_grid_idx].T  # shape (r, n_candidates)

            if N_sensors > A.shape[1]:
                A = np.dot(A, A.T)  # shape (n_candidates, n_candidates)

            qr_idx = sla.qr(A.T, pivoting=True)[-1]

            sensor_idx = measure_grid_idx[qr_idx[:N_sensors]]
            sensor_idx = sensor_idx.ravel() % (Nx * Ny)  # only the first variable (e.g., ux) for sensor placement

            if np.unique(sensor_idx).size < N_sensors:
                print(
                    f"Warning: QR selection returned {np.unique(sensor_idx).size} unique sensors, less than requested {N_sensors}."
                )
                sensor_idx = np.unique(sensor_idx)
                extra_needed = N_sensors - sensor_idx.size
                if extra_needed > 0:
                    extra_sensor_idx = measure_grid_idx[qr_idx[N_sensors : N_sensors + extra_needed]]
                    extra_sensor_idx = extra_sensor_idx.ravel() % (Nx * Ny)
                    sensor_idx = np.concatenate([sensor_idx, extra_sensor_idx])
        else:
            if N_sensors < len(one_dom):
                sensor_idx = np.sort(self.rng.choice(one_dom, size=N_sensors, replace=False), axis=None)
            else:
                sensor_idx = one_dom.copy()

        if N_sensors > len(measure_grid_idx):
            print(
                f"Requested number of sensors {N_sensors} >= grid size in domain of measurement ({len(measure_grid_idx)})"
            )

        if plot:
            # sensors against the original and measurement grids, for debugging
            plt.figure()
            x_idx, y_idx = np.unravel_index(np.arange(Nx * Ny), (Nx, Ny))
            plt.scatter(x_idx, y_idx, label="Original grid", alpha=0.01)
            x_idx, y_idx = np.unravel_index(measure_grid_idx[: len(measure_grid_idx) // 2], (Nx, Ny))
            plt.scatter(x_idx, y_idx, label="Measurement grid")
            x_idx, y_idx = np.unravel_index(sensor_idx, (Nx, Ny))
            plt.scatter(x_idx, y_idx, label="Sensors")
            plt.legend()
            plt.show()

        # sensors for all u
        sensor_idx = [sensor_idx + Nx * Ny * i for i in range(self.grid_shape[0])]

        return np.array(sensor_idx).reshape((-1,))


class LatentROMMixin:
    """Latent-space plumbing between a `Projector` and a `Forecaster`.

    Deliberately phrased in terms of *latent coefficients*, not POD modes, and
    of the `Forecaster` protocol, not `ESN_model`, so the same code serves
    POD/AE/CAE crossed with ESN/LSTM.
    """

    #: label for the latent coordinates in figures; overridden by POD_ESN to
    #: keep its historical `\Phi_j` notation
    latent_symbol = "z"

    def get_latent_coefficients(self, Nt=1):
        """The last ``Nt`` latent states from the model history, ``(Nt, r, m)``."""
        if Nt == 1:
            return self.hist[-1, : self.N_latent]
        return self.hist[-Nt:, : self.N_latent]

    def get_observables(self, Nt=1, Z=None, **kwargs):
        """Sensor readings from the forecast latent trajectory, ``(Nt, Nq, m)``.

        With ``measure_modes`` the latent coefficients *are* the observables;
        otherwise they are decoded at ``sensor_locations``.
        """
        if self.measure_modes:
            return self.get_latent_coefficients(Nt=Nt)

        if Z is None:
            Z = self.get_latent_coefficients(Nt=Nt)  # Nt x r x m

        og_shape = Z.shape
        reshape = Z.ndim == 3
        if reshape:
            Z = Z.transpose(1, 0, 2)  # r x Nt x m
            Z = Z.reshape(self.N_latent, -1)  # r x Nt*m

        obs = self.decode_at(Z, idx=self.sensor_locations)  # (Nq, Nt*m)

        if reshape:
            obs = obs.reshape(self.Nq, og_shape[0], og_shape[2])  # Nq x Nt x m
            obs = obs.transpose(1, 0, 2)  # Nt x Nq x m

        return obs  # Nt x Nq x m

    # ---- labels -------------------------------------------------------------

    @property
    def obs_labels(self):
        if self.measure_modes:
            obs_labels = [f"${self.latent_symbol}_{j + 1}$" for j in np.arange(self.N_latent)]
        else:
            ux_labels = ["${u_x}" + f"_{j}$" for j in np.arange(self.N_sensors)]
            uy_labels = ["${u_y}" + f"_{j}$" for j in np.arange(self.N_sensors)]
            obs_labels = [*ux_labels, *uy_labels]
        assert len(obs_labels) == self.Nq
        return obs_labels

    @property
    def state_labels(self):
        """Latent coordinates followed by the forecaster's own recurrent state.

        The second half is delegated rather than assumed: an ESN contributes one
        reservoir block, an LSTM contributes two (h and c). Asking the
        forecaster for its labels is what keeps this method from being
        ESN-shaped.
        """
        labels = []
        if self.update_state:
            labels += [f"${self.latent_symbol}_{j + 1}$" for j in np.arange(self.N_latent)]
        return labels + self.forecaster_state_labels

    # ---- resets -------------------------------------------------------------

    def reset_case(self, reset_projector=False, reset_forecaster=False, Z0=None, reset_ESN=None, **kwargs):
        """Refit the projector and/or reset the forecaster.

        Refitting the projector always forces a forecaster reset: the latent
        coordinates it forecasts have changed meaning.

        ``reset_ESN`` is accepted as a deprecated alias for
        ``reset_forecaster`` -- the parameter was ESN-specific before the LSTM
        gained a wrapper, and callers predating that still pass it.
        """
        if reset_ESN is not None:
            reset_forecaster = reset_ESN

        if reset_projector:
            self.refit_projector(**kwargs)
            reset_forecaster = True

        if reset_forecaster:
            if Z0 is None:
                Z0 = self.latent_training_trajectory[0]
            self.reset_forecaster(psi0=Z0, **kwargs)

    def refit_projector(self, **kwargs):
        """Hook for subclasses whose projector can be refitted in place."""
        raise NotImplementedError(f"{type(self).__name__} does not support refitting its projector")
