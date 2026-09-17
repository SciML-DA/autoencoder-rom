"""Checks for the latent-ROM layer: the ntsa model protocol, the projector
additions sensor placement is built on, and the AE/CAE-backed ROMs themselves.

`test_pod_sensor_placement_unchanged` checks that placing sensors on
`spatial_basis` reproduces the original QR pivoting on `Psi` for POD.

Slow cases (anything that trains an ESN) are skipped unless VERIFY_SLOW=1, in
keeping with the rest of tests/.
"""

import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # dynamodels and echostatenetwork import pyplot at import time

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
# POD_ESN regression: the lift must not have moved anything
# ---------------------------------------------------------------------------

def _reference_placement(case, N_sensors):
    """The original placement algorithm, pivoting explicitly on ``Psi`` on the grid."""
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
                   perform_test=False, train_ESN=False,
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
            n_epochs=30, perform_test=False)

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
                  perform_test=False)
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
# LSTM ROMs
# ---------------------------------------------------------------------------

@slow
def test_lstm_model_state_layout_and_step_contract(data):
    """An LSTM carries (h, c), so psi is longer than the ESN's [u; r] -- and
    time_step must still honour the protocol's (Nt+1, Nphi, m) shape."""
    from models.data_driven import POD_LSTM

    m = POD_LSTM(data=data, dt=0.01, n_modes=6, method="exact", random_state=0,
                 grid_shape=GRID, domain=DOMAIN, Nq=8, N_units=16, N_wash=5,
                 epochs=2, seq_len=50)

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
@pytest.mark.parametrize("name", ["POD_LSTM", "AE_LSTM"])
def test_rom_keeps_projector_and_forecaster_histories(data, name):
    """Both training histories survive construction, which overwrites `training_history`."""
    import models.data_driven as dd

    kw = dict(data=data, dt=0.01, grid_shape=GRID, domain=DOMAIN, Nq=8, N_units=16, N_wash=5,
              epochs=2, seq_len=50)
    kw |= (dict(n_modes=6, method="exact", random_state=0) if "POD" in name
           else dict(n_latent=6, n_epochs=5, seed=3))
    m = getattr(dd, name)(**kw)

    assert m.forecaster_history is not None and m.forecaster_history.n_epochs_run == 2
    if "POD" in name:
        assert m.projector_history is None
    else:
        assert m.projector_history is not None and m.projector_history.n_epochs_run == 5
        assert m.seed == 3, "the LSTM takes the projector's seed rather than resetting it"
    m.close()


@slow
@pytest.mark.parametrize("name", ["POD_LSTM", "AE_LSTM", "CAE_LSTM"])
def test_lstm_roms_place_sensors_and_forecast(data, name):
    """Every LSTM ROM places sensors and forecasts finite observables."""
    import models.data_driven as dd

    cls = getattr(dd, name)
    kw = dict(data=data, dt=0.01, grid_shape=GRID, domain=DOMAIN, Nq=8,
              N_units=16, N_wash=5, epochs=2, seq_len=50)
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
def test_lstm_model_forecasts_each_ensemble_member(data):
    """Each member of an ensemble forecasts as a single-member model would."""
    from models.data_driven import POD_LSTM

    m = POD_LSTM(data=data, dt=0.01, n_modes=6, method="exact", random_state=0,
                 grid_shape=GRID, domain=DOMAIN, Nq=8, N_units=16, N_wash=5,
                 epochs=2, seq_len=50)
    single, _ = m.time_step(Nt=5)

    u0 = m.current_state[:m.N_dim]
    members = np.hstack([u0, u0 + 0.01, u0 - 0.01])
    m.reset_forecaster(u0=members)
    ensemble, t = m.time_step(Nt=5)
    assert ensemble.shape == (6, m.Nphi, 3) and t.shape == (6,)
    np.testing.assert_allclose(ensemble[..., :1], single, rtol=0, atol=1e-12)

    averaged, _ = m.time_step(Nt=5, averaged=True)
    np.testing.assert_allclose(averaged.mean(axis=-1), ensemble[..., 0], atol=1e-12)
    m.close()


# ---------------------------------------------------------------------------
# Sensor readout and resets
# ---------------------------------------------------------------------------

def _masked(data):
    """The fixture with a solid body, so the fluid mask matters."""
    masked = data.copy()
    masked[:, :, 5:7, 2:4] = np.nan
    return masked


@slow
@pytest.mark.parametrize("name", ["POD_ESN", "AE_LSTM"])
def test_observables_read_the_decoded_field_at_the_sensors(data, name):
    """get_observables equals the decoded field on the grid at sensor_locations."""
    import models.data_driven as dd

    kw = (dict(n_modes=6, method="exact", random_state=0) if name == "POD_ESN"
          else dict(n_latent=4, n_epochs=10))
    m = getattr(dd, name)(data=_masked(data), dt=0.01, grid_shape=GRID, domain=DOMAIN,
                          Nq=6, seed=0, train_forecaster=False, **kw)
    Z = m.latent_training_trajectory[:, :5]
    Nu, Nx, Ny = GRID
    field = m._to_physical_grid(m.decode(Z)).transpose(0, 2, 3, 1).reshape(Nu * Nx * Ny, -1)

    np.testing.assert_allclose(m.get_observables(Z=Z), field[m.sensor_locations], rtol=1e-12)
    Z3 = Z.T[:, :, None]  # (Nt, N_latent, m=1)
    np.testing.assert_allclose(m.get_observables(Z=Z3)[:, :, 0], field[m.sensor_locations].T, rtol=1e-12)


@slow
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "POD_ESN",
            marks=pytest.mark.xfail(
                raises=AttributeError,
                strict=True,
                reason="dynamodels Model.reset_model re-runs Model.__init__, which cannot reassign alpha0",
            ),
        ),
        "POD_LSTM",
    ],
)
def test_reset_case_resets_the_forecaster(data, name):
    """reset_case(reset_forecaster=True) restarts from the first training snapshot."""
    import models.data_driven as dd

    kw = (dict(N_func_evals=4, N_grid=2) if name == "POD_ESN" else dict(epochs=2, seq_len=50))
    m = getattr(dd, name)(data=data, dt=0.01, n_modes=6, method="exact", random_state=0, grid_shape=GRID,
                          domain=DOMAIN, Nq=8, seed=0, N_units=16, N_wash=5, **kw)
    p, t = m.time_integrate(Nt=5)
    m.update_history(p, t)
    m.reset_case(reset_forecaster=True)
    assert np.isfinite(m.get_observable_hist()).all()
    m.close()
