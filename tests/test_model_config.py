"""Checks for `config.model_config`: saving and restoring the autoencoders and the LSTM.

A restored model must reproduce the saved one exactly, and a configuration must
hash the same whether it comes from the options or from the trained model.
"""

import os
import pathlib
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")  # dynamodels and echostatenetwork import pyplot at import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from config.model_config import (  # noqa: E402
    ModelConfig,
    auto_load_or_create,
    list_saved_configs,
    load_model_from_config,
    save_model_to_config,
)
from models.data_driven.forecasters import LSTM, LSTM_model  # noqa: E402

slow = pytest.mark.skipif(os.environ.get("VERIFY_SLOW") != "1", reason="set VERIFY_SLOW=1 to run (trains an ESN)")


def _field(n_t=80, nx=12, ny=8, seed=0):
    """(2, n_t, nx, ny) travelling waves with a solid block."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_t)[:, None, None]
    x = np.linspace(0, 2 * np.pi, nx)[:, None]
    y = np.linspace(0, np.pi, ny)[None, :]
    u = np.sin(x + t) * np.cos(y) + 0.01 * rng.standard_normal((n_t, nx, ny))
    v = np.cos(x - t) * np.sin(y) + 0.01 * rng.standard_normal((n_t, nx, ny))
    X = np.array([u, v])
    X[:, :, 4:6, 3:5] = np.nan
    return X


@pytest.fixture(scope="module")
def grid():
    return _field()


def _autoencoder(name):
    from models.data_driven import autoencoders

    small = dict(n_latent=3, epochs=4, batch_size=16, seed=1)
    arch = {
        "AE": dict(hidden=(16,)),
        "CAE": dict(channels=(4, 8)),
        "AEJax": dict(hidden=(16,)),
        "CAEJax": dict(channels=(4, 8)),
    }[name]
    return getattr(autoencoders, name), small | arch


@pytest.mark.parametrize("name", ["AE", "CAE", "AEJax", "CAEJax"])
def test_autoencoder_round_trip_is_exact(grid, tmp_path, name):
    """A restored autoencoder encodes and decodes exactly as the saved one."""
    cls, options = _autoencoder(name)
    model = cls.from_data(grid, **options)
    config, path = save_model_to_config(model, save_dir=tmp_path, training_data_filename="field")

    restored = load_model_from_config(q=path.name, load_dir=tmp_path)
    assert restored is not None and type(restored) is cls
    Z = model.encode(grid)
    np.testing.assert_array_equal(restored.encode(grid), Z)
    np.testing.assert_array_equal(restored.decode(Z), model.decode(Z))
    assert restored.training_history == model.training_history
    assert restored.training_history.n_epochs_run == 4
    assert config.to_hash() == ModelConfig.from_init_params(cls, grid, "field", **options).to_hash()


def test_lstm_round_trip_is_exact(tmp_path):
    """A restored LSTM forecasts exactly as the saved one."""
    t = np.linspace(0, 30, 500)
    data = np.stack([np.sin(t), np.cos(1.3 * t), np.sin(0.7 * t) * np.cos(t)], axis=-1)
    options: dict[str, Any] = dict(N_units=8, seed=2, epochs=2, seq_len=40, N_wash=20)
    lstm = LSTM.from_data(data, **options)
    config, path = save_model_to_config(lstm, save_dir=tmp_path)

    restored = load_model_from_config(config=config, load_dir=tmp_path)
    assert isinstance(restored, LSTM) and restored.trained
    Y, state = lstm.openLoop(data[:50])
    Y_r, state_r = restored.openLoop(data[:50])
    np.testing.assert_array_equal(Y_r, Y)
    np.testing.assert_array_equal(restored.closedLoop(data[49], 30, state_r)[0], lstm.closedLoop(data[49], 30, state)[0])
    assert restored.training_history == lstm.training_history
    assert config.to_hash() == ModelConfig.from_init_params(LSTM, data, **options).to_hash()
    assert path.name == config.to_hash()


def test_lstm_jax_matches_numpy_and_round_trips(tmp_path):
    """In float64, LSTMJax trains to the NumPy LSTM's parameters and restores exactly."""
    import jax

    from models.data_driven.forecasters import LSTMJax

    jax.config.update("jax_enable_x64", True)
    t = np.linspace(0, 40, 1200)
    data = np.stack([np.sin(t), np.cos(1.3 * t), np.sin(0.7 * t) * np.cos(t)], axis=-1)
    options: dict[str, Any] = dict(N_units=16, seed=1, epochs=3, seq_len=40, N_wash=20)
    numpy_lstm = LSTM.from_data(data, **options)
    jax_lstm = LSTMJax.from_data(data, dtype="float64", **options)

    for name, value in numpy_lstm.p.items():
        np.testing.assert_allclose(jax_lstm.p[name], value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(jax_lstm.training_history.val, numpy_lstm.training_history.val, rtol=1e-10)
    Y, state = numpy_lstm.openLoop(data[:100])
    Y_jax, state_jax = jax_lstm.openLoop(data[:100])
    np.testing.assert_allclose(Y_jax, Y, atol=1e-12)
    F, _ = numpy_lstm.closedLoop(data[99], 200, state)
    F_jax, _ = jax_lstm.closedLoop(data[99], 200, state_jax)
    np.testing.assert_allclose(F_jax, F, atol=1e-10)

    _, path = save_model_to_config(jax_lstm, save_dir=tmp_path)
    restored = load_model_from_config(q=path.name, load_dir=tmp_path)
    assert type(restored) is LSTMJax and restored.dtype == "float64"
    np.testing.assert_array_equal(restored.openLoop(data[:100])[0], Y_jax)


def test_lstm_jax_rejects_unknown_dtype():
    """dtype must name float32 or float64."""
    from models.data_driven.forecasters import LSTMJax

    with pytest.raises(ValueError, match="dtype"):
        LSTMJax(N_dim_in=2, N_units=4, dtype="float16")


def test_auto_load_or_create_trains_once(grid, tmp_path, monkeypatch):
    """The second call with the same options loads instead of training."""
    cls, options = _autoencoder("AE")
    first = auto_load_or_create(cls, grid, config_dir=tmp_path, training_data_filename="field", **options)

    def fail(*_args, **_kwargs):
        raise AssertionError("trained again")

    monkeypatch.setattr(cls, "from_data", classmethod(fail))
    second = auto_load_or_create(cls, grid, config_dir=tmp_path, training_data_filename="field", **options)
    np.testing.assert_array_equal(second.encode(grid), first.encode(grid))
    assert [row["name"] for row in list_saved_configs(tmp_path)] == [
        ModelConfig.from_init_params(cls, grid, "field", **options).to_hash()
    ]


def test_hash_depends_on_options_and_training_data():
    """Changing an option or the training data name changes the hash."""
    from models.data_driven.autoencoders import AE

    base = ModelConfig.from_init_params(AE, n_latent=3).to_hash()
    assert ModelConfig.from_init_params(AE, n_latent=3, hidden=[512, 128]).to_hash() == base
    assert ModelConfig.from_init_params(AE, n_latent=4).to_hash() != base
    assert ModelConfig.from_init_params(AE, training_data_filename="other", n_latent=3).to_hash() != base
    assert ModelConfig.from_init_params(AE, n_latent=3, device="cuda").to_hash() == base


def test_missing_config_loads_nothing(tmp_path):
    """Loading a name the store does not hold returns None."""
    assert load_model_from_config(q="0123456789abcdef", load_dir=tmp_path) is None


def test_roms_and_lstm_model_cannot_be_stored():
    """Classes whose construction needs more than their options are rejected."""
    from models.data_driven import POD_LSTM

    for cls in (LSTM_model, POD_LSTM):
        with pytest.raises(TypeError, match="cannot be stored"):
            ModelConfig.from_init_params(cls)


def test_function_activation_cannot_be_stored():
    """An activation passed as a function has no stored form."""
    import jax

    from models.data_driven.autoencoders import AEJax

    model = AEJax(n_latent=3, hidden=(16,), epochs=1, activation=jax.nn.tanh)
    with pytest.raises(ValueError, match="registered name"):
        model.config_options()


@slow
def test_esn_config_loads_instead_of_retraining(tmp_path, monkeypatch):
    """`auto_load_or_create_esn` trains once, then loads identical matrices."""
    import config.model_config as mc

    t = np.linspace(0, 60, 1500)
    data = np.stack([np.sin(t), np.cos(1.3 * t)], axis=-1)[None]
    options = dict(dt=0.04, N_units=20, seed=3, N_grid=2, N_func_evals=4, N_folds=2, N_split=2, N_wash=10)
    first = mc.auto_load_or_create_esn(config_dir=tmp_path, data=data, **options)

    esn_model = mc.ESN_model

    def load_only(**kwargs):
        assert kwargs.get("data") is None, "trained again"
        return esn_model(**kwargs)

    monkeypatch.setattr(mc, "ESN_model", load_only)
    second = mc.auto_load_or_create_esn(config_dir=tmp_path, data=data, **options)
    for name in ("Wout", "W", "Win", "norm", "shift"):
        a, b = getattr(first, name), getattr(second, name)
        a, b = (x.toarray() if hasattr(x, "toarray") else x for x in (a, b))
        np.testing.assert_array_equal(a, b)
    assert [(path / "esn_config.yaml").is_file() for path in tmp_path.iterdir()] == [True]
