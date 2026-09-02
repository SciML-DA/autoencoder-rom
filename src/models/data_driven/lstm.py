"""`LSTM_model` -- the LSTM as a `dynamodels.Model`, sibling of `ESN_model`.

`lstm_core.LSTM` is a complete forecaster (truncated BPTT, closed-loop
validation, gradient check) that had no wrapper making it a Model, so nothing
could use it. This is that wrapper, and it is deliberately the *second*
implementation of the same bridge: whatever `ESN_model` does to turn a
`Forecaster` into a Model, this does too, and anything the two need in common
belongs in `LatentROMMixin` rather than in either.

State layout
------------
`ESN_model` packs ``psi = [u ; r]`` -- latent coordinates and one reservoir
state. An LSTM carries *two* recurrent states, so here ``psi = [u ; h ; c]``
and ``Nphi = N_dim + 2 * N_units``. That difference is the reason
`LatentROMMixin.state_labels` asks for ``forecaster_state_labels`` instead of
assuming a single block of reservoir units.

Ensembles
---------
`LSTM.step` asserts strictly ``(N, 1)`` shapes, so this wrapper is
single-member only; ``m > 1`` raises rather than silently forecasting one
member. `ESN_model` supports ensembles because the reservoir update is a plain
matrix product that broadcasts. Lifting that limit means vectorising
`LSTM.step`, which is a change to the forecaster, not to this wrapper.
"""

from __future__ import annotations

import numpy as np
from dynamodels import DiscreteIntegrator, Model

from .lstm_core import LSTM

__all__ = ["LSTM_model"]


class LSTM_model(LSTM, Model):
    """An LSTM used as a data-driven forecast model.

    Training data is mandatory at construction, as for `ESN_model`.
    """

    figs_folder: str = "figs/LSTM/"

    update_state = True
    update_memory = True  # carry (h, c) in psi -- the LSTM's analogue of the reservoir

    t_train = None
    t_val = None
    t_test = 0.0

    N_units = 50
    N_wash = 5

    extra_print_params = ["N_units", "N_wash", "seq_len", "epochs", "lr"]

    # ---- two collisions between LSTM and Model, resolved here -------------
    #
    # 1. `rng`. `LSTM.__init__` assigns `self.rng`, but `Model.rng` is a lazy
    #    property with no setter, so the assignment raises. `ESN_model` never
    #    hits this only because `EchoStateNetwork` happens to define its own
    #    `rng` property; nothing guarantees a forecaster does. Re-declaring it
    #    here (ahead of Model in the MRO) keeps Model's lazy-from-seed
    #    behaviour while allowing the assignment.
    #
    # 2. `history`. `LSTM.history` is the training-loss curve; `Model.history`
    #    is the HistoryTracker state buffer. Model.__init__ runs last and would
    #    silently destroy the loss curves, so they are moved to `loss_history`
    #    first -- see __init__.

    @property
    def rng(self):
        """numpy.random.Generator, lazily created from `seed` as in `Model`,
        but assignable, because `LSTM.__init__` sets it directly."""
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    @rng.setter
    def rng(self, value):
        self._rng = value

    def __init__(self, data=None, dt=1.0, y0=None, plot_training=False, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            Training trajectories, ``(L, Nt, N_dim)`` -- the same layout
            `ESN_model` takes, i.e. what `phi_to_esn_layout` produces.
        dt : float
            Time step of the training data.
        y0 : np.ndarray, optional
            Initial state; taken from the data when omitted.
        """
        for key in list(kwargs.keys()):
            if key in vars(LSTM_model):
                setattr(self, key, kwargs.pop(key))

        if data is None:
            raise ValueError("LSTM_model requires training data")
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        y0 = data[0, 0] if y0 is None else y0
        n_dim = data.shape[-1]

        # ---- 1. build the forecaster ------------------------------------
        lstm_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs) if k in vars(LSTM)
        }
        LSTM.__init__(self, N_dim_in=n_dim, N_units=self.N_units,
                      seed=kwargs.get("seed", 0), **lstm_kwargs)

        # window lengths, mirroring ESN_model._process_initialization_data
        t_total = data.shape[1] * dt
        self.t_train = self.t_train or t_total * 0.8
        self.t_val = self.t_val or self.t_train * 0.2

        # ---- 2. train it -------------------------------------------------
        if not self.trained:
            print("Training LSTM model...")
            self.train(data, verbose=plot_training)

        # Rescue the training-loss curves before Model.__init__ rebinds
        # `history` to its HistoryTracker (see the collision note above).
        self.loss_history = getattr(self, "history", None)

        # ---- 3. initial state, then Model --------------------------------
        self.N_dim = n_dim
        if not hasattr(self, "Nq"):
            self.Nq = n_dim

        psi0 = self.build_psi(
            u=np.asarray(y0, dtype=float).reshape(n_dim, 1),
            h=np.zeros((self.N_units, 1)),
            c=np.zeros((self.N_units, 1)),
        )

        Model.__init__(self, dt=dt, psi0=psi0,
                       integrator_class=DiscreteIntegrator, **kwargs)

    # ---- Model protocol ---------------------------------------------------

    @property
    def t_transient(self):
        return self.t_train + self.t_val + self.t_test

    @property
    def dt_step(self):
        return self.dt

    @property
    def t_CR(self):
        return 10 * self.dt

    @property
    def obs_labels(self):
        return [f"$u_{j+1}$" for j in np.arange(self.N_dim)]

    @property
    def forecaster_state_labels(self):
        """Labels for the recurrent block of ``psi`` -- two of them, unlike the
        ESN's single reservoir. Consumed by `LatentROMMixin.state_labels`."""
        if not self.update_memory:
            return []
        return ([f"$h_{j+1}$" for j in np.arange(self.N_units)]
                + [f"$c_{j+1}$" for j in np.arange(self.N_units)])

    @property
    def state_labels(self):
        labels = []
        if self.update_state:
            labels += [f"$u_{j+1}$" for j in np.arange(self.N_dim)]
        return labels + self.forecaster_state_labels

    def get_observables(self, Nt=1, **kwargs):
        if Nt == 1:
            return self.hist[-1, : self.N_dim]
        return self.hist[-Nt:, : self.N_dim]

    # ---- state packing ----------------------------------------------------

    def build_psi(self, u, h, c):
        """``[u ; h ; c]`` -> psi, for 2-D ``(N, m)`` or 3-D ``(Nt, N, m)`` input."""
        if u.ndim == 2:
            return np.concatenate((u, h, c), axis=0)
        if u.ndim == 3:
            return np.concatenate((u, h, c), axis=1)
        raise ValueError(f"u must be 2- or 3-dimensional, got {u.ndim}")

    def unbuild_psi(self, psi=None):
        """psi -> ``(u, h, c)``."""
        if psi is None:
            psi = self.current_state
        n, k = self.N_dim, self.N_units
        if psi.ndim == 2:
            return psi[:n], psi[n : n + k], psi[n + k : n + 2 * k]
        return psi[:, :n], psi[:, n : n + k], psi[:, n + k : n + 2 * k]

    # ---- the discrete route the ntsa protocol asks for --------------------

    def time_step(self, Nt=10, averaged=False):
        """Free-run ``Nt`` steps, returning ``(Nt + 1, Nphi, m)`` including the
        initial condition -- the shape contract in the ntsa model protocol.

        ``averaged`` is accepted for signature parity with `ESN_model` and
        ignored: this wrapper is single-member, so there is nothing to average.
        """
        if not self.trained:
            raise RuntimeError("LSTM model not trained")
        if self.m != 1:
            raise NotImplementedError(
                f"LSTM_model is single-member (m={self.m}); LSTM.step asserts "
                "(N, 1) shapes. Vectorise the forecaster to lift this."
            )

        t = np.round(
            self.current_time + np.arange(0, Nt + 1) * self.dt, self.precision_t
        )
        u0, h0, c0 = self.unbuild_psi(self.current_state)

        u = np.empty((Nt + 1, self.N_dim, self.m))
        h = np.empty((Nt + 1, self.N_units, self.m))
        c = np.empty((Nt + 1, self.N_units, self.m))
        u[0], h[0], c[0] = u0, h0, c0

        for i in range(Nt):
            # the LSTM steps in normalised space and reads out the next input
            x = self.normalize_input(u[i])
            h_new, c_new, y, _ = self.step(x, h[i], c[i])
            h[i + 1], c[i + 1] = h_new, c_new
            u[i + 1] = self.denormalize_output(y)

        return self.build_psi(u=u, h=h, c=c), t

    # ---- resets -----------------------------------------------------------

    def reset_forecaster(self, psi0=None, **kwargs):
        """Re-seed the recurrent state (and optionally the latent state).

        `ESN_model` retrains on reset; here the trained weights are kept and
        only ``(h, c)`` are zeroed, which is the meaningful reset for an LSTM
        whose parameters did not change.
        """
        u0 = (
            np.asarray(psi0, dtype=float).reshape(self.N_dim, 1)
            if psi0 is not None
            else self.current_state[: self.N_dim]
        )
        psi = self.build_psi(
            u=u0,
            h=np.zeros((self.N_units, 1)),
            c=np.zeros((self.N_units, 1)),
        )
        self.update_history(psi[np.newaxis], t=np.array([self.current_time]),
                            reset=True)
