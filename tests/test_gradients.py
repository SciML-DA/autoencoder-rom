"""Central-difference checks of the hand-written gradients.

Covers the JAX dense and convolutional autoencoders, whose gradients come from
`jax.value_and_grad`, and the numpy LSTM, whose `backward` is written by hand.
Every check runs in float64.
"""

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from models.data_driven.autoencoders.ae_jax import AEJaxConfig, get_ae_params, mse_loss  # noqa: E402
from models.data_driven.autoencoders.cae_jax import (  # noqa: E402
    CAEJaxConfig,
    flat_to_grid,
    get_cae_params,
    masked_mse_loss,
)
from models.data_driven.forecasters import LSTM  # noqa: E402

SEEDS = (0, 1, 2)


def jax_relative_error(loss_fn, params, x, h=1e-6):
    """Compares the gradient against a central difference on its largest entry.

    Args:
      loss_fn: Scalar loss of `(params, x)`.
      params: Parameter pytree, all float64.
      x: Input batch, float64.
      h: Finite-difference step.

    Returns:
      The relative error between the analytic and finite-difference gradients.
    """
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    assert all(a.dtype == jnp.float64 for _, a in leaves) and x.dtype == jnp.float64

    grads = jax.grad(loss_fn)(params, x)
    path, leaf = max(jax.tree_util.tree_flatten_with_path(grads)[0], key=lambda kv: float(jnp.abs(kv[1]).max()))
    idx = np.unravel_index(int(jnp.argmax(jnp.abs(leaf))), leaf.shape)
    analytic = float(leaf[idx])

    def bumped(e):
        return jax.tree_util.tree_map_with_path(lambda p, a: a.at[idx].add(e) if p == path else a, params)

    fd = (float(loss_fn(bumped(h), x)) - float(loss_fn(bumped(-h), x))) / (2 * h)
    return abs(fd - analytic) / abs(analytic)


def grid_data(seed):
    """A `(Nu, N_t, Nx, Ny)` random field with one solid point."""
    X = np.random.default_rng(seed).standard_normal((2, 40, 12, 10))
    X[:, :, 0, 0] = np.nan
    return X


@pytest.mark.parametrize("seed", SEEDS)
def test_ae_jax_gradient(seed):
    Q = np.nan_to_num(grid_data(seed)).reshape(2, 40, -1).transpose(0, 2, 1).reshape(-1, 40)
    cfg = AEJaxConfig(n_x=Q.shape[0], hidden=(16,), n_latent=4, dtype=jnp.float64)
    params = get_ae_params(jax.random.key(seed), cfg)
    x = jnp.asarray(Q.T[:16], dtype=jnp.float64)
    assert jax_relative_error(lambda p, b: mse_loss(p, b, cfg), params, x) < 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_cae_jax_gradient(seed):
    X = grid_data(seed)
    Nu, n_t, Nx, Ny = X.shape
    fluid = ~np.isnan(X[0, 0]).ravel()
    Q = X.reshape(Nu, n_t, -1)[:, :, fluid].transpose(0, 2, 1).reshape(-1, n_t)
    cfg = CAEJaxConfig(
        grid_shape=(Nu, Nx, Ny), channels=(4, 8), kernel_size=3, stride=2, pad=1, n_latent=4, dtype=jnp.float64
    )
    mask = jnp.asarray(fluid.reshape(Nx, Ny), dtype=jnp.float64)[None, None]
    G = flat_to_grid(Q, cfg, np.flatnonzero(fluid))[:8]
    params = get_cae_params(jax.random.key(seed), cfg)
    assert jax_relative_error(lambda p, g: masked_mse_loss(p, g, cfg, mask), params, G) < 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_lstm_backward(seed, n_probe=6, eps=1e-5):
    rng = np.random.default_rng(seed)
    lstm = LSTM(N_dim_in=3, N_units=8, seed=seed)
    sequences = lstm._as_sequences(rng.standard_normal((60, 3)))
    norm, shift = lstm._set_norm(sequences, method=lstm.norm_method)
    U = (sequences[0] - shift) / norm
    X, Yt = U[:-1], U[1:]

    def loss():
        Yh, caches, _ = lstm.forward(X)
        return 0.5 * np.mean((Yh - Yt) ** 2), Yh, caches

    L, Yh, caches = loss()
    grads = lstm.backward(caches, (Yh - Yt) / (Yh.shape[0] * lstm.N_dim_in))
    floor = 100.0 * np.finfo(float).eps * max(L, 1.0) / eps

    worst = 0.0
    for key, P in lstm.p.items():
        for _ in range(n_probe):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            p0 = P[idx]
            P[idx] = p0 + eps
            Lp = loss()[0]
            P[idx] = p0 - eps
            Lm = loss()[0]
            P[idx] = p0

            num, ana = (Lp - Lm) / (2 * eps), grads[key][idx]
            if abs(num) + abs(ana) >= floor:
                worst = max(worst, abs(num - ana) / (abs(num) + abs(ana)))

    assert worst < 1e-5
