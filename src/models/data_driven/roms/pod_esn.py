from ..autoencoders import POD
from ..forecasters import ESN_model
from ..latent_rom import LatentROMMixin, SensorPlacementMixin
import numpy as np


class POD_ESN(LatentROMMixin, SensorPlacementMixin, ESN_model, POD):
    """Performs POD for a data matrix and trains an ESN to forecast the POD
    temporal coefficients.


    D(x,t) = Σ_j sigma_j φ_j(x) ψ_j(t)    for  j = 0, ..., N_modes-1

    [latex]
    D(x,t) = \\sum_j \\sigma_j \\phi_j(x) \\psi_j(t)
    for j = 0, ..., N_modes-1

    POD properties:
        - Psi: temporal basis [N, N_modes], with N = Ndim x Nx x Ny
        - Phi: spatial basis [Nt, N_modes]
        - Sigma: POD Sigmas [N_modes, ],  note: Lambdas can be computed as: Sigma = np.sqrt(Lambda)
    """

    Nq = 10

    perform_test = False  # Wether to perform testing of the ESN model

    extra_print_params = [
        *ESN_model.extra_print_params,
        "Nq",
        "measure_modes",
        "N_modes",
    ]

    def __init__(
        self,
        data,
        dt,
        skip_sensor_placement=False,
        train_ESN=True,
        domain_of_measurement=None,
        down_sample_measurement=None,
        **kwargs,
    ):
        """
        Initialize the POD-ESN model.

        Args:
            - data  (np.ndarray): Data to be used for the POD decomposition and ESN training  [ (Nu, N_t, Nx, Ny) or (N_t, Ndim*Nx*Ny) ]
            - skip_sensor_placement (bool, optional): Whether to skip sensor placement. Defaults to False.
            - train_ESN (bool, optional): Whether to train the ESN. Defaults to True.
            - **kwargs: Additional keyword arguments to configure the parent classes Model/ESN/POD.
                e.g.,   domain (list): Domain of the data.
                        grid_shape (tuple): Shape of the grid.
                        t_CR (float): Time constant for the ESN.
                        Nq (int): Number of measurements or sensors.
                        sensor_locations (list): Locations of the sensors.
                        etc.
        """

        for key in list(kwargs.keys()):
            if key in vars(POD_ESN):
                setattr(self, key, kwargs.pop(key))

        # __________________________ Init POD ___________________________ #
        POD.__init__(
            self, X=data, **kwargs
        )  # Initialize POD class and run decomposition

        # __________________________ Init ESN ___________________________ #
        # Initialize ESN to forecast the POD coefficients
        if train_ESN:
            Phi = self.Phi.copy()
            if Phi.ndim == 2:
                Phi = Phi[np.newaxis, ...]

            Phi = Phi.transpose(0, 2, 1)  # must be LxNtxNdim for ESN
            ESN_model.__init__(self, data=Phi, dt=dt, **kwargs)

        # __________________________ Select sensors ___________________________ #
        if self.measure_modes or skip_sensor_placement:
            self.Nq = self.N_modes
        elif self.sensor_locations is None:
            self.domain_of_measurement = domain_of_measurement
            self.down_sample_measurement = down_sample_measurement
            self.sensor_locations = self.define_sensors(N_sensors=self.Nq)
            self.Nq = len(self.sensor_locations)
        else:
            # If the sensors are already defined, use them
            self.Nq = len(self.sensor_locations)

        print("========= POD-ESN model complete =========")

    # ---------------------------------------------------------------------
    # Everything that used to live between here and the PLOTS section --
    # obs_labels, state_labels, N_sensors, get_POD_coefficients,
    # get_observables, reset_case, select_sensors, domain_of_measurement,
    # down_sample_measurement, grid_of_measurement and define_sensors -- was
    # projector-agnostic apart from one line, and now lives in
    # `latent_rom.LatentROMMixin` / `latent_rom.SensorPlacementMixin` so that
    # AE_ESN and CAE_ESN reuse it rather than copy it.
    #
    # The one POD-specific line was QR-pivoting on `self.Psi`; it is now
    # `self.spatial_basis(z0)`, and `POD.spatial_basis` returns `Psi`, so the
    # sensor positions this class produces are unchanged.
    # ---------------------------------------------------------------------

    #: keep the historical \Phi_j notation in figures rather than the generic z_j
    latent_symbol = "\\Phi"

    def get_POD_coefficients(self, Nt=1):
        """Backward-compatible alias for `get_latent_coefficients`."""
        return self.get_latent_coefficients(Nt=Nt)

    @property
    def latent_training_trajectory(self):
        """Latent coefficients the ESN was trained on -- POD's `Phi`, (r, N_t)."""
        return self.Phi
