from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["AdamState", "adam_init", "adam_update"]

_ACT = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "identity": lambda h: h,
}

ActSpec = str | Callable | tuple

_DTYPES = {"float32": jnp.float32, "float64": jnp.float64}


def _act_fn(a: str | Callable) -> Callable:
    return _ACT[a] if isinstance(a, str) else a


def _resolve_acts(spec: ActSpec, n_layers: int, what: str) -> tuple:
    # one activation name per ACTIVATED layer

    # same activation for every layer
    if isinstance(spec, str) or callable(spec):
        return (spec,) * n_layers

    spec = tuple(spec)
    if len(spec) != n_layers:
        raise ValueError(f"{what}: got {len(spec)} activations for {n_layers} activated layers")
    return spec


# --- learning/training ---
class AdamState(NamedTuple):
    m: dict
    v: dict
    t: jax.Array  # traced, not a python int, it lives in the state, not a closure


# initializes the moment pytrees (have same shape as params)
def adam_init(params) -> AdamState:
    z = jax.tree.map(jnp.zeros_like, params)
    return AdamState(m=z, v=z, t=jnp.zeros((), dtype=jnp.int32))


def adam_update(
    grads: dict,
    state: AdamState,
    params: dict,
    lr: float,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    wd=0.0,
):
    # to match torch.optim.Adam (uses L2 weight decay)
    t = state.t + 1
    bc1 = 1.0 - b1**t
    bc2 = 1.0 - b2**t

    if wd:
        grads = jax.tree.map(lambda g, p: g + wd * p, grads, params)

    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state.m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * g * g, state.v, grads)

    def step(p, m, v):
        # cast to the parameter's dtype, so float32 stays float32 with jax_enable_x64
        dt = p.dtype
        return p - jnp.asarray(lr, dt) * (m / bc1.astype(dt)) / (jnp.sqrt(v / bc2.astype(dt)) + eps)

    params = jax.tree.map(step, params, m, v)
    return params, AdamState(m, v, t)


def _epoch_batches(key, n: int, batch_size: int):
    """Shuffled indices reshaped to (n_batches, batch_size).

    The remainder is dropped so every step sees one shape and jit compiles once.
    The permutation is redrawn each epoch, so the dropped samples rotate and
    nothing is systematically excluded.
    """
    # we should see if this is the best way to do this, or if we could maybe by default
    # add decorrelation lag to make sure we distinct contiguous blocks rather than random?
    perm = jax.random.permutation(key, n)
    n_batches = max(n // batch_size, 1)
    return perm[: n_batches * batch_size].reshape(n_batches, -1)
