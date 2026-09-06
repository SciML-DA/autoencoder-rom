"""Checks for the latent-ROM layer: the ntsa model protocol, the projector
additions sensor placement is built on, and the AE/CAE-backed ROMs themselves.

The important one is `test_pod_sensor_placement_unchanged`: sensor placement was
lifted out of `POD_ESN` into `SensorPlacementMixin` and its one POD-specific
line replaced by `spatial_basis`. `POD.spatial_basis` returns `Psi`, so that
lift must be a no-op for POD_ESN -- if this test ever fails, the generalisation
changed linear behaviour, which it must not.

Slow cases (anything that trains an ESN) are skipped unless VERIFY_SLOW=1, in
keeping with the rest of tests/.
"""

import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # placement plots must never block a headless run

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from models.data_driven.autoencoders import POD, Projector  # noqa: E402

SLOW = os.environ.get("VERIFY_SLOW") == "1"
slow = pytest.mark.skipif(not SLOW, reason="set VERIFY_SLOW=1 to run (trains a model)")

GRID = (2, 16, 8)
DOMAIN = [0.0, 2 * np.pi, 0.0, np.pi]


def synthetic_field(N_t=300, Nx=16, Ny=8, seed=0):
    """(2, N_t, Nx, Ny) travelling waves, low-rank plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, N_t)
    x = np.linspace(0, 2 * np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = (np.sin(X)[None] * np.cos(t)[:, None, None]
         + 0.5 * np.sin(2 * X + Y)[None] * np.sin(2 * t)[:, None, None])
    v = np.cos(X + Y)[None] * np.sin(t)[:, None, None]
    return np.array([u + 0.02 * rng.standard_normal(u.shape),
                     v + 0.02 * rng.standard_normal(v.shape)])


@pytest.fixture(scope="module")
def data():
    return synthetic_field()


@pytest.fixture(scope="module")
def pod(data):
    return POD(n_modes=6, method="exact", grid_shape=GRID, domain=DOMAIN).fit(data)


# ---------------------------------------------------------------------------
# Projector additions
# ---------------------------------------------------------------------------

def test_pod_spatial_basis_is_psi(pod):
    """A linear decoder's Jacobian is its own matrix -- POD must hand back Psi
    itself, not a copy or an approximation, or sensor placement would move."""
    assert pod.spatial_basis() is pod.Psi


def test_generic_jacobian_recovers_the_exact_linear_basis(pod):
    """The base-class finite-difference Jacobian is validated against the one
    projector whose answer is known exactly."""
    exact = pod.spatial_basis()
    generic = Projector.spatial_basis(pod)
    assert generic.shape == exact.shape
    assert np.linalg.norm(generic - exact) / np.linalg.norm(exact) < 1e-10


def test_spatial_basis_rejects_wrong_sized_z0(pod):
    with pytest.raises(ValueError, match="expected N_latent"):
        Projector.spatial_basis(pod, np.zeros(pod.N_latent + 1))


def test_decode_at_matches_decoding_then_indexing(pod, data):
    """POD overrides decode_at with the cheap `Psi[idx] @ Z` route; it must
    agree with the generic decode-everything-then-index path."""
    Z = pod.encode(data)[:, :5]
    idx = np.array([3, 17, 44, 100])
    np.testing.assert_allclose(
        pod.decode_at(Z, idx=idx), Projector.decode_at(pod, Z, idx=idx), rtol=1e-12
    )


def test_decode_at_without_idx_is_full_decode(pod, data):
    Z = pod.encode(data)[:, :4]
    np.testing.assert_allclose(pod.decode_at(Z), pod.decode(Z), rtol=1e-12)


# ---------------------------------------------------------------------------
# The Forecaster protocol
# ---------------------------------------------------------------------------

def test_both_forecasters_satisfy_the_protocol():
    """The protocol is ours, and only earns its place if the two forecasters
    written independently of it both already satisfy it."""
    from echostatenetwork import EchoStateNetwork

    from models.data_driven.forecasters import LSTM

    required = ("train", "step", "normalize_input", "compute_nRMSE", "copy",
                "trained", "N_wash", "norm", "shift", "norm_method")
    for cls in (EchoStateNetwork, LSTM):
        missing = [m for m in required if not hasattr(cls, m)]
        assert not missing, f"{cls.__name__} is missing {missing}"


# ---------------------------------------------------------------------------
# POD_ESN regression: the lift must not have moved anything
# ---------------------------------------------------------------------------

def _reference_placement(case, N_sensors):
    """The pre-lift algorithm, pivoting explicitly on ``Psi``.

    This is what `POD_ESN.define_sensors` did before placement moved into
    `SensorPlacementMixin` and started going through `spatial_basis`. Kept
    inline (rather than as a recorded array of indices) so the invariant is
    checked against the computation itself and does not silently rot the moment
    the fixture data changes.
    """
    import scipy.linalg as sla

    Nu, Nx, Ny = case.grid_shape
    measure_grid_idx = np.asarray(case.grid_of_measurement)

    Psi = case._to_physical_grid(case.Psi).transpose(1, 0, 2, 3)
    Psi = np.nan_to_num(Psi, nan=0.0)
    Psi = Psi.reshape(Psi.shape[0], Nu, Nx * Ny)
    A = Psi.reshape(Psi.shape[0], -1)[:, measure_grid_idx].T
    if N_sensors > A.shape[1]:
        A = np.dot(A, A.T)
    qr_idx = sla.qr(A.T, pivoting=True)[-1]
    sensor_idx = measure_grid_idx[qr_idx[:N_sensors]].ravel() % (Nx * Ny)
    return np.array(
        [sensor_idx + Nx * Ny * i for i in range(Nu)]
    ).reshape((-1,))


@slow
def test_pod_sensor_placement_unchanged(data):
    """Generalising placement must be a no-op for POD.

    `POD.spatial_basis` returns `Psi`, so the mixin pivots QR on exactly the
    matrix `POD_ESN` used to pivot on. Asserted against a reference
    implementation of the pre-lift algorithm rather than a recorded array, so
    the check survives changes to the test data.
    """
    from models.data_driven import POD_ESN

    case = POD_ESN(data=data, dt=0.01, n_modes=6, method="exact",
                   random_state=0, grid_shape=GRID, domain=DOMAIN, Nq=8, seed=0,
                   N_units=30, N_wash=5, N_func_evals=4, N_grid=2,
                   perform_test=False, plot_case=False, train_ESN=False,
                   skip_sensor_placement=True)
    case.domain_of_measurement = None
    case.down_sample_measurement = None

    np.testing.assert_array_equal(
        case.define_sensors(N_sensors=8), _reference_placement(case, 8)
    )


# ---------------------------------------------------------------------------
# AE_ESN / CAE_ESN
# ---------------------------------------------------------------------------

@slow
@pytest.mark.parametrize("name", ["AE_ESN", "CAE_ESN"])
def test_autoencoder_rom_end_to_end(data, name):
    import models.data_driven as dd

    cls = getattr(dd, name)
    m = cls(data=data, dt=0.01, n_latent=6, grid_shape=GRID, domain=DOMAIN,
            Nq=8, seed=0, N_units=30, N_wash=5, N_func_evals=4, N_grid=2,
            n_epochs=30, perform_test=False, plot_case=False)

    # the projector actually learned something
    Q = m.preprocess_snapshot(data)
    rel = np.mean((Q - m.reconstruct(Q)) ** 2) / np.mean(Q**2)
    assert rel < 0.2, f"{name} reconstruction is no better than the mean ({rel:.3f})"

    # sensors were placed on the decoder Jacobian, one per field component
    assert m.spatial_basis().shape == (Q.shape[0], m.N_latent)
    assert m.Nq == len(m.sensor_locations) == 8 * GRID[0]
    assert m._z_mean.shape == (m.N_latent,)

    # the ntsa protocol's discrete route: time_step(Nt) yields Nt+1 states,
    # initial condition included
    psi = m.time_step(Nt=10)[0]
    assert psi.shape[0] == 11, f"time_step must return Nt+1 states, got {psi.shape[0]}"
    assert psi.shape[1] == m.Nphi

    # and a closed-loop forecast produces finite observables
    p, t = m.time_integrate(Nt=20)
    m.update_history(p, t)
    obs = m.get_observable_hist()
    assert obs.shape[1] == m.Nq
    assert np.isfinite(obs).all()
    m.close()


@slow
@pytest.mark.parametrize("name", ["POD_ESN", "AE_ESN"])
def test_ntsa_protocol_surface(data, name):
    """The attributes and methods ntsa's model protocol requires.

    https://andreanovoa.github.io/ntsa/protocol/ -- this is the checklist the
    port is meant to satisfy, so it is worth asserting rather than assuming.
    """
    import models.data_driven as dd

    common = dict(data=data, dt=0.01, grid_shape=GRID, domain=DOMAIN, Nq=8,
                  seed=0, N_units=30, N_wash=5, N_func_evals=4, N_grid=2,
                  perform_test=False, plot_case=False)
    if name == "POD_ESN":
        m = dd.POD_ESN(n_modes=6, method="exact", random_state=0, **common)
    else:
        m = dd.AE_ESN(n_latent=6, n_epochs=30, **common)

    for attr in ("Nphi", "Nq", "psi0", "alpha0", "params", "fixed_params",
                 "dt", "t_transient", "t_CR", "obs_labels", "alpha_labels", "name"):
        assert hasattr(m, attr), f"{name} lacks protocol attribute {attr!r}"
    for meth in ("time_integrate", "update_history", "get_observable_hist",
                 "close", "time_step"):
        assert callable(getattr(m, meth)), f"{name} lacks protocol method {meth!r}"

    assert len(m.obs_labels) == m.Nq
    assert np.asarray(m.psi0).ndim == 2  # (Nphi, m)
    m.close()


# ---------------------------------------------------------------------------
# Pass 4: the LSTM half of the matrix.
#
# These are the acceptance test for the mixins. If a forecaster swap needed
# changes *inside* LatentROMMixin/SensorPlacementMixin, the Forecaster boundary
# would be in the wrong place -- so what matters is that these classes are
# declarations and nothing more.
# ---------------------------------------------------------------------------

@slow
def test_lstm_model_state_layout_and_step_contract(data):
    """An LSTM carries (h, c), so psi is longer than the ESN's [u; r] -- and
    time_step must still honour the protocol's (Nt+1, Nphi, m) shape."""
    from models.data_driven import POD_LSTM

    m = POD_LSTM(data=data, dt=0.01, n_modes=6, method="exact", random_state=0,
                 grid_shape=GRID, domain=DOMAIN, Nq=8, N_units=16, N_wash=5,
                 epochs=2, seq_len=50, plot_case=False)

    assert m.Nphi == 6 + 2 * 16, "psi must be [latent ; h ; c]"
    psi, t = m.time_step(Nt=10)
    assert psi.shape == (11, m.Nphi, m.m)
    assert t.shape == (11,)

    # both recurrent blocks are labelled, not just one
    labels = m.state_labels
    assert sum(lab.startswith("$h_") for lab in labels) == 16
    assert sum(lab.startswith("$c_") for lab in labels) == 16
    m.close()


@slow
def test_lstm_training_history_survives_model_init(data):
    """`LSTM.history` (loss curves) and `Model.history` (state buffer) collide;
    Model.__init__ runs last, so the curves must be rescued to `loss_history`
    or they are silently destroyed."""
    from models.data_driven import POD_LSTM

    m = POD_LSTM(data=data, dt=0.01, n_modes=6, method="exact", random_state=0,
                 grid_shape=GRID, domain=DOMAIN, Nq=8, N_units=16, N_wash=5,
                 epochs=2, seq_len=50, plot_case=False)
    assert set(m.loss_history) == {"train", "val"}
    assert len(m.loss_history["train"]) > 0
    m.close()


@slow
@pytest.mark.parametrize("name", ["POD_LSTM", "AE_LSTM", "CAE_LSTM"])
def test_lstm_roms_reuse_the_mixins(data, name):
    """Every LSTM ROM gets sensor placement, observables and latent bookkeeping
    from the same mixins the ESN ROMs use."""
    import models.data_driven as dd

    cls = getattr(dd, name)
    kw = dict(data=data, dt=0.01, grid_shape=GRID, domain=DOMAIN, Nq=8,
              N_units=16, N_wash=5, epochs=2, seq_len=50, plot_case=False)
    kw |= (dict(n_modes=6, method="exact", random_state=0) if "POD" in name
           else dict(n_latent=6, n_epochs=20))
    m = cls(**kw)

    assert m.Nq == len(m.sensor_locations) == 8 * GRID[0]
    assert m.spatial_basis().shape[1] == m.N_latent
    p, t = m.time_integrate(Nt=15)
    m.update_history(p, t)
    obs = m.get_observable_hist()
    assert obs.shape[1] == m.Nq and np.isfinite(obs).all()
    m.close()


@slow
def test_lstm_model_rejects_ensembles_explicitly(data):
    """LSTM.step asserts (N, 1) shapes. m > 1 must raise rather than quietly
    forecasting a single member."""
    from models.data_driven import POD_LSTM

    m = POD_LSTM(data=data, dt=0.01, n_modes=6, method="exact", random_state=0,
                 grid_shape=GRID, domain=DOMAIN, Nq=8, N_units=16, N_wash=5,
                 epochs=2, seq_len=50, plot_case=False)
    m.psi0 = np.tile(m.psi0, (1, 3))  # pretend a 3-member ensemble
    m.update_history(m.psi0[np.newaxis], t=np.array([0.0]), reset=True)
    with pytest.raises(NotImplementedError, match="single-member"):
        m.time_step(Nt=2)
    m.close()
