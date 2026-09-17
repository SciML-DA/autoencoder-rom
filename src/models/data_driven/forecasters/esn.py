import inspect

import numpy as np
import scipy.linalg as sla
from dynamodels import DiscreteIntegrator, Model
from echostatenetwork import EchoStateNetwork


def phi_to_esn_layout(Z):
    """``(N_latent, N_t)`` latent coefficients (a projector's ``encode()`` output,
    or POD's ``Phi``) -> the ``(L, N_t, N_latent)`` layout `ESN_model` expects as
    ``data``; a 3-D input is taken as ``(L, N_latent, N_t)`` segments."""
    Z = np.asarray(Z)
    if Z.ndim == 2:
        Z = Z[np.newaxis, ...]  # (1, N_latent, N_t)
    return Z.transpose(0, 2, 1)  # (L, N_t, N_latent)


class ESN_model(EchoStateNetwork, Model):
    """ESN model Class
    - Use a ESN as a data-driven forecast model
    - Note: training data is a mandatory input to the initialization
    """

    update_reservoir = True
    update_state = True
    training_data_filename = (
        None  # Filename of the data used for training (for config saving/loading only, not used in the actual training)
    )

    Wout_svd = False
    validation_data = None

    t_train, t_val, t_test = None, None, 0.0

    perform_test = True
    save_ESN_training = False

    upsample = 1

    N_wash = 5  # Number of washout steps i.e., open-loop initialization
    N_units = 50  # Number of neurons
    N_func_evals = 40
    N_grid = 5
    noise = 1e-2
    Win_type = "sparse"
    N_folds = 8
    N_split = 5

    # Hyperparameter optimization ranges
    rho_range = (0.2, 0.8)
    sigma_in_range = (-2, 2)
    tikh_range = [1e-6, 1e-9, 1e-12]

    # params = ['Wout']
    extra_print_params = [
        "rho",
        "sigma_in",
        "N_units",
        "N_wash",
        "upsample",
        "update_reservoir",
        "update_state",
    ]

    def __init__(
        self,
        dt,
        **kwargs,
    ):
        """
        Arguments inside kwargs:
            - data: data to train the ESN
                np.array to train with shape [L x Nt x Ndim], where
                    L is the number of different sets of parameters (e.g., different experiments),
                    Nt is the number of time steps (train + validate + test), and
                    Ndim is the number of dimensions of the system.
            - y0: initial state to initialize the ESN (if None, use first data point)
            - Other ESN and Model parameters can be provided as keyword arguments,
                e.g., N_units, N_wash, update_reservoir, update_state, etc.
        """

        data = kwargs.pop("data", None)
        y0 = kwargs.pop("y0", None)

        # =================== STEP 1: EchoStateNetwork INITIALIZATION ======================

        [setattr(self, key, kwargs.pop(key)) for key in list(kwargs.keys()) if key in vars(ESN_model)]

        if data is not None:
            assert isinstance(data, np.ndarray), f"Expected data to be a numpy array, got {type(data)}"
            data = self._process_initialization_data(data, dt, kwargs)  # type: np.ndarray # with shape (L, Nt, Ndim)
            y0 = data[0, 0]
        elif y0 is None:
            raise ValueError("Either training data or initial state y0 must be provided to initialize the ESN_model.")

        initial_dict = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in vars(EchoStateNetwork)}
        EchoStateNetwork.__init__(self, y=y0, dt=dt, **initial_dict)

        # =================== STEP 2: EchoStateNetwork TRAINING ======================
        # Train the network if not already trained
        if not self.trained:
            print("Training ESN model...")
            self.train(train_data=data, plot_training=False, **kwargs)

            # save validation data for initialization
            Y_wtv = self._split_and_format_data(data)[1]
            self.validation_data = Y_wtv[-(self.N_wash + self.N_val) :]

        # ================== STEP 3: DEFINE INITIAL STATE & PARAMS ======================
        if not hasattr(self, "Nq"):
            self.Nq = len(self.observed_idx)  # Number of observed dimensions (for the physical state)

        psi0 = self.initialize_from_val_data()  # shape (Ndim + N_units + Na, m)

        # Initialise SVD Wout terms if required
        if self.Wout_svd:
            [self.Wout_U, self.Wout_Sigma0, self.Wout_Vh] = sla.svd(self.Wout, full_matrices=False)
            self.Wout_Sigma = self.Wout_Sigma0

        # =================== STEP 4: Model INITIALIZATION ======================
        Model.__init__(self, dt=dt, psi0=psi0, integrator_class=DiscreteIntegrator, **kwargs)

    @property
    def t_transient(self):
        return self.t_train + self.t_val + self.t_test

    def _process_initialization_data(self, data, dt, kwargs) -> np.ndarray:
        """Process the data input for initialization"""
        # Increase ndim if there is only one set of parameters
        if data.ndim == 1:
            data = data[np.newaxis, :, np.newaxis]
        elif data.ndim == 2:
            data = data[np.newaxis, :]

        # Check that the times are provided and not in time steps
        Nt = data.shape[1]
        for key in ["train", "val", "test"]:
            if f"N_{key}" in kwargs.keys():
                setattr(self, f"t_{key}", kwargs.pop(f"N_{key}") * dt)

        # Set other ESN_model attributes provided in kwargs

        #  Set time attributes  #
        t_total = Nt * dt
        self.t_train = self.t_train or t_total * 0.8
        self.t_val = self.t_val or self.t_train * 0.2

        if self.perform_test:
            self.t_test = self.t_test or t_total - self.t_train - self.t_val

            assert abs((ts := sum([self.t_train, self.t_val, self.t_test])) - t_total) <= dt / 2.0, (
                f"t_train + t_val + t_test {ts} <= t_total {t_total}"
            )

        return data

    # ______________________ New class attributes ______________________ #
    def modify_settings(self, **kwargs):
        # Modify the settings of the ESN_model
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f"Key {key} not in ESN_model class")

        if self.ensemble_cfg:
            est_alpha = self.est_alpha.copy()
            # If Wout is being estimated, we need to update the est_alpha list to include the SVD components
            # and remove Wout. We do not directly estimate Wout, but rather its singular values.
            if "Wout" in est_alpha:
                if not self.Wout_svd:
                    self.Wout_svd = True
                    [self.Wout_U, self.Wout_Sigma0, self.Wout_Vh] = sla.svd(self.Wout, full_matrices=False)
                    self.Wout_Sigma = self.Wout_Sigma0
                # Update the est_alpha list with the new SVD component keys
                new_keys = [f"svd_{qi}" for qi in range(self.N_dim)]
                self.est_alpha = [a for a in est_alpha if a != "Wout"] + new_keys

                self.alpha_labels = {
                    key: f"$\\sigma_{{{key.split('_')[1]}}}$" for key in new_keys
                }  # update the alpha labels with the new ones (e.g., svd_0, svd_1, etc.)

                self.alpha_lims = {
                    key: (None, None) for key in new_keys
                }  # update the alpha lims with the new ones (e.g., svd_0, svd_1, etc.)
                self.M = None  # Set the M matrix to None to force re-computation
        self.M = None

    @property
    def dt_step(self):
        return self.dt_ESN

    @property
    def t_CR(self):
        return self.t_val

    @property
    def Wout_U(self):
        """Dimensions N_dim x N_dim"""
        return self._Wout_U

    @Wout_U.setter
    def Wout_U(self, U):
        assert U.shape == self.Wout.shape, f"Expected shape {self.Wout.shape}, got {U.shape}"
        self._Wout_U = U

    @property
    def Wout_Vh(self):
        """Dimensions N_dim x N_dim"""
        return self._Wout_Vh

    @Wout_Vh.setter
    def Wout_Vh(self, Vh):
        assert Vh.shape == (
            self.N_dim,
            self.N_dim,
        ), f"Expected shape ({self.N_dim}, {self.N_dim}), got {Vh.shape}"
        self._Wout_Vh = Vh

    @property
    def Wout_Sigma(self):
        if self.Wout_svd:
            self.Wout_Sigma = self.alpha_to_Sigma
        return self._Wout_Sigma

    @property
    def alpha_to_Sigma(self):
        alpha_matrix = self.get_alpha_matrix

        eigs = np.zeros((self.m, self.N_dim, self.N_dim))

        for qi in range(self.N_dim):
            key = f"svd_{qi}"
            if key in self.est_alpha:
                ai = self.est_alpha.index(key)
                vals = alpha_matrix[ai]
            else:
                vals = self.Wout_Sigma0[qi] * np.ones(self.m)

            eigs[:, qi, qi] = vals

        return eigs

    @property
    def get_alpha_matrix(self):
        alpha = np.empty((len(self.est_alpha), self.m))
        for aj, param in enumerate(self.est_alpha):
            for mi, alpha_dict in enumerate(self.get_alpha()):
                alpha[aj, mi] = alpha_dict[param]
        return alpha

    @property
    def Wout_Sigma0(self):
        return self._Wout_Sigma0

    @Wout_Sigma0.setter
    def Wout_Sigma0(self, eigs):
        self._Wout_Sigma0 = eigs
        params = self.params.copy()

        for eig_i, val in enumerate(eigs):
            setattr(self, f"svd_{eig_i}", val)
            params.append(f"svd_{eig_i}")
            # Keep alpha0 in sync if the model is already initialized
            if hasattr(self, "_alpha0"):
                self._alpha0[f"svd_{eig_i}"] = val

        self.params = params

    @Wout_Sigma.setter
    def Wout_Sigma(self, eigs):
        if eigs.ndim == 1:
            assert eigs.shape[0] == self.N_dim, f"Expected shape ({self.N_dim},) got {eigs.shape}"
            eigs = np.diag(eigs)
        elif eigs.ndim == 2:
            assert eigs.shape[-1] == self.N_dim, f"Expected shape ({self.N_dim},) got {eigs.shape}"
            if eigs.shape[0] != self.m:
                assert eigs.shape[0] == self.N_dim and np.allclose(eigs, np.diag(np.diagonal(eigs))), (
                    f"Expected diagonal matrix, got {eigs.shape}"
                )
            else:
                eigs = np.array([np.diag(e) for e in eigs])  ## this will be needed for the parameter estimation
                assert eigs.shape == (
                    self.m,
                    self.N_dim,
                    self.N_dim,
                ), f"Expected shape ({self.m}, {self.N_dim},{self.N_dim}) got {eigs.shape}"
        else:
            assert eigs.shape == (
                self.m,
                self.N_dim,
                self.N_dim,
            ), f"Expected shape ({self.m}, {self.N_dim},{self.N_dim}) got {eigs.shape}"

        self._Wout_Sigma = eigs

    # ______________________ Changed EchoStateNetwork class attributes ______________________ #

    @property
    def N_ens(self):
        if isinstance(self.ensemble_cfg, dict):
            return self.ensemble_cfg.get("m")
        else:
            return self.current_state.shape[-1]

    def initialize_from_val_data(self, N_ens=1, seed=0):
        """Initialise the ESN state using traiining data"""
        assert self.validation_data is not None

        data = self.validation_data.copy()

        if hasattr(self, "seed"):
            seed = self.seed
        rng0 = np.random.default_rng(seed)

        # initialise state with a random sample from test data
        u_init, r_init = np.empty((self.N_dim, N_ens)), np.empty((self.N_units, N_ens))

        # Random time windows and dimension
        if data.shape[0] == 1:
            dim_ids = [0] * N_ens
        else:
            # Choose a random dimension from the data
            replace = N_ens >= data.shape[0]
            dim_ids = rng0.choice(data.shape[0], size=N_ens, replace=replace)

        # Choose random time indices from the data
        t_ids = rng0.choice(data.shape[1] - self.N_wash, size=N_ens, replace=False)

        for ii, ti, dim_i in zip(range(N_ens), t_ids, dim_ids):
            u_wash = data[dim_i, ti : ti + self.N_wash]
            r_open = np.zeros((self.N_units, 1))
            u_open = np.zeros((self.N_dim, 1))
            # Open-loop reservoir
            for u_in in u_wash:
                u_open, r_open = self._single_step(u_in, r_open)

            # store final state into the initialization arrays
            u_init[:, ii] = u_open.squeeze()
            r_init[:, ii] = r_open.squeeze()

        # Set physical and reservoir states as ensembles
        return self.build_psi(u=u_init, r=r_init)

    @property
    def forecaster_state_labels(self):
        """Labels for the reservoir block of ``psi``. `LatentROMMixin` asks each
        forecaster for these rather than assuming a single block, because an
        LSTM contributes two (see `LSTM_model`)."""
        if not self.update_reservoir:
            return []
        return [f"$r_{j + 1}$" for j in np.arange(self.N_units)]

    def reset_forecaster(self, *args, **kwargs):
        """Forecaster-neutral name for `reset_ESN`, used by
        `LatentROMMixin.reset_case`."""
        return self.reset_ESN(*args, **kwargs)

    def reset_ESN(self, data, u0=None, **kwargs):

        if u0 is None:
            u0 = self.reservoir_to_physical(self.reservoir_state)

        EchoStateNetwork.__init__(self, y=u0, dt=self.dt, **kwargs)
        # Train the network
        possible_args = inspect.getfullargspec(self.train)[0]
        train_args = {key: val for key, val in kwargs.items() if key in possible_args}

        # Train network
        self.train(train_data=data, plot_training=False, **train_args)

        # Reset model class
        kwargs["psi0"] = self.build_psi()
        self.reset_model(**kwargs)

    # ______________________ Changed Model class attributes ______________________ #

    @property
    def state_labels(self):

        return [f"$u_{{{j + 1}}}$" for j in np.arange(self.N_dim)] + [
            f"$r_{{{j + 1}}}$" for j in np.arange(self.N_units)
        ]

    @property
    def obs_labels(self):
        return [f"$u_{{{j + 1}}}$" for j in self.observed_idx]

    @property
    def reservoir_state(self):
        return self.current_state[self.N_dim : self.N_dim + self.N_units, :]

    def reservoir_to_physical(self, r):

        bias_out = self.bias_out * np.ones((1, r.shape[-1]))
        r_aug = np.concatenate((r, bias_out), axis=0)

        if not self.Wout_svd:
            return np.dot(r_aug.T, self.Wout).T
        else:
            if r.shape[-1] == self.m:
                Wout = np.einsum("ij,kjl,lm->imk", self.Wout_U, self.Wout_Sigma, self.Wout_Vh)
                return np.einsum("ij,ikj->kj", r_aug, Wout)
            else:
                # average the alpha values
                print("Averaging Wout_Sigma for reservoir_to_physical")
                Wout_Sigma_avg = np.mean(self.Wout_Sigma, axis=0)
                Wout = np.dot(self.Wout_U, np.dot(Wout_Sigma_avg, self.Wout_Vh))
                return np.dot(r_aug.T, Wout).T

    def time_step(self, Nt=10, averaged=False):
        """
        Args:
            Nt: number of forecast steps (physical time, not dt_ESN)
            averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                            the ensemble is forecast as a mean, i.e., every member is the mean forecast.
            alpha: possibly-varying input_parameters
        Returns:
            psi: forecasted state (Nt x N x m)
            t: time of the propagated psi
        """

        assert self.trained, "ESN model not trained"
        # 1. get initial condition

        t = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)
        psi0 = self.current_state
        u, r_out = np.empty((Nt + 1, self.N_dim, self.m)), np.empty((Nt + 1, self.N_units, self.m))
        u[0], r_out[0] = self.unbuild_psi(psi0)

        if averaged:
            # Mean state
            u_m, r_m = (np.mean(yy[0], axis=-1, keepdims=True) for yy in [u, r_out])
            u_dev, r_dev = u[0] - u_m[0], r_out[0] - r_m[0]

            for i in range(Nt):
                u_m, r_m = self._single_step(u_m, r_m)
                u[i + 1] = u_m + u_dev
                r_out[i + 1] = r_m + r_dev

        else:
            for i in range(Nt):
                u[i + 1], r_out[i + 1] = self._single_step(u[i], r_out[i])

        psi = self.build_psi(u=u, r=r_out)

        return psi, t

    def _single_step(self, u, r):
        u_input = self.outputs_to_inputs(full_state=u)
        return self.step(u_input, r)

    def build_psi(self, u=None, r=None):
        """Build the full state vector psi from physical states u and reservoir states r
        Returns:
           psi: full state vector (Nt x (Nphi + Na) x m)
        """
        if r is None:
            r = self.reservoir_state
        if u is None:
            u = self.reservoir_to_physical(r)

        if u.ndim == 2 and r.ndim == 2:
            ax_dim = 0
        elif u.ndim == 3 and r.ndim == 3:
            ax_dim = 1
            if u.shape[0] != r.shape[0]:
                raise ValueError(f"Incompatible time steps for u ({u.shape[0]}) and r ({r.shape[0]})")
        else:
            raise ValueError(f"Incompatible dimensions for u ({u.ndim}) and r ({r.ndim})")

        if self.update_state and self.update_reservoir:
            phi = np.concatenate((u, r), axis=ax_dim)
        elif self.update_state:
            phi = u
        else:
            phi = r

        if self.Na > 0:
            alph = self.get_alpha_matrix
            if u.ndim == 3:
                alph = np.tile(alph, reps=(u.shape[0], 1, 1))  # repeat for all time steps (alpha is constant in time)
            return np.concatenate((phi, alph), axis=ax_dim)
        else:
            return phi

    def unbuild_psi(self, psi=None):
        """Extract physical states u and reservoir states r from the full state vector psi
        Args:
            psi: full state vector (N x m). If None, use the current_state
        Returns:
            u: physical states (N_dim x m) (or None if not updated)
            r: reservoir states (N_units x m) (or None if not updated)
        """
        if psi is None:
            psi = self.current_state
        if psi.ndim == 2:
            psi = np.expand_dims(psi, axis=0)
            squeeze = True
        else:
            squeeze = False

        assert psi.shape[1] > self.N_units, f"Expected psi shape (N x m) with N > {self.N_units}, got {psi.shape}"
        u = psi[:, : self.N_dim]
        r = psi[:, self.N_dim : self.N_dim + self.N_units]

        if squeeze:
            u = u.squeeze(axis=0)
            r = r.squeeze(axis=0)

        return u, r
