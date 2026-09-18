"""LSTM forecaster with training and rollouts compiled by JAX.

`LSTMJax` has the options, normalization, and saving of `LSTM`, and trains the
same way: a washout, truncated backpropagation over windows of `seq_len` steps
with Adam and global gradient clipping, and closed-loop validation that keeps
the best epoch. Each training segment runs as one compiled scan over its
windows, and `openLoop` and `closedLoop` run as compiled scans. The trained
parameters are stored as NumPy arrays, so every `LSTM` method works on them.

Importing this module imports JAX.

Typical usage example:

  lstm = LSTMJax(N_dim_in=3, N_units=64, epochs=40)
  lstm.train(data)
  Y, state = lstm.openLoop(data[:100])
  forecast, _ = lstm.closedLoop(data[99], 500, state)
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from ..training import TrainingHistory
from .lstm import LSTM, Array, State

__all__ = ["LSTMJax"]

#: Network parameters on device, keyed like `LSTM.p`.
JaxParams = dict[str, jax.Array]


# ── Cell and scans ────────────────────────────────────────────────────────────


def _sigmoid(z: jax.Array) -> jax.Array:
    """Applies the logistic sigmoid elementwise, as `lstm._sigmoid` does."""
    return 1.0 / (1.0 + jnp.exp(-z))


def _cell(p: JaxParams, x: jax.Array, h: jax.Array, c: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advances the LSTM one step.

    Args:
      p: The parameters.
      x: Normalized input, shape `(N_dim_in, m)`.
      h: Hidden state, shape `(N_units, m)`.
      c: Cell state, shape `(N_units, m)`.

    Returns:
      The new hidden state, the new cell state, and the readout.
    """
    f = _sigmoid(p["Wf"] @ x + p["Uf"] @ h + p["bf"])
    i = _sigmoid(p["Wi"] @ x + p["Ui"] @ h + p["bi"])
    g = jnp.tanh(p["Wg"] @ x + p["Ug"] @ h + p["bg"])
    o = _sigmoid(p["Wo"] @ x + p["Uo"] @ h + p["bo"])
    c_new = f * c + i * g
    h_new = o * jnp.tanh(c_new)
    return h_new, c_new, p["Wy"] @ h_new + p["by"]


def _teacher_forced(p: JaxParams, X: jax.Array, h: jax.Array, c: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Runs the LSTM over a sequence of true inputs.

    Args:
      p: The parameters.
      X: Normalized inputs, shape `(T, N_dim_in, m)`.
      h: Initial hidden state, shape `(N_units, m)`.
      c: Initial cell state, shape `(N_units, m)`.

    Returns:
      The readouts, shape `(T, N_dim_in, m)`, and the final hidden and cell
      states.
    """

    def step(state: tuple[jax.Array, jax.Array], x: jax.Array) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        h_new, c_new, y = _cell(p, x, *state)
        return (h_new, c_new), y

    (h, c), Y = jax.lax.scan(step, (h, c), X)
    return Y, h, c


def _closed_loop(
    p: JaxParams, x: jax.Array, h: jax.Array, c: jax.Array, n_steps: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Feeds each readout back as the next input.

    Args:
      p: The parameters.
      x: Normalized first input, shape `(N_dim_in, m)`.
      h: Hidden state, shape `(N_units, m)`.
      c: Cell state, shape `(N_units, m)`.
      n_steps: Number of steps.

    Returns:
      The readouts, shape `(n_steps, N_dim_in, m)`, and the final hidden and
      cell states.
    """

    def step(
        state: tuple[jax.Array, jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        x, h, c = state
        h_new, c_new, y = _cell(p, x, h, c)
        return (y, h_new, c_new), y

    (_, h, c), Y = jax.lax.scan(step, (x, h, c), None, length=n_steps)
    return Y, h, c


_teacher_forced_jit = jax.jit(_teacher_forced)
_closed_loop_jit = jax.jit(_closed_loop, static_argnames=("n_steps",))


# ── Training ──────────────────────────────────────────────────────────────────


def _window_loss(
    p: JaxParams, X: jax.Array, Y_true: jax.Array, h: jax.Array, c: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Computes the one-step-ahead loss over one window.

    Args:
      p: The parameters.
      X: Normalized inputs, shape `(seq_len, N_dim_in, 1)`.
      Y_true: The inputs one step later, shaped like `X`.
      h: Hidden state at the start of the window.
      c: Cell state at the start of the window.

    Returns:
      Half the mean squared error, and the hidden and cell states at the end of
      the window.
    """
    Y, h, c = _teacher_forced(p, X, h, c)
    return 0.5 * jnp.mean((Y - Y_true) ** 2), (h, c)


def _adam(
    p: JaxParams,
    grads: JaxParams,
    m: JaxParams,
    v: JaxParams,
    t: jax.Array,
    lr: jax.Array,
    clip: jax.Array,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[JaxParams, JaxParams, JaxParams]:
    """Clips the gradients by global norm and applies one Adam step.

    Args:
      p: The parameters.
      grads: Gradients, keyed like `p`.
      m: First moment estimates, keyed like `p`.
      v: Second moment estimates, keyed like `p`.
      t: Number of steps taken, including this one.
      lr: Step size.
      clip: Maximum global gradient norm.
      b1: Decay rate of the first moment.
      b2: Decay rate of the second moment.
      eps: Constant added to the denominator.

    Returns:
      The updated parameters and moment estimates.
    """
    squared = sum(jnp.sum(g**2) for g in grads.values())
    scale = clip / jnp.maximum(jnp.sqrt(squared), clip)
    new_p: JaxParams = {}
    new_m: JaxParams = {}
    new_v: JaxParams = {}
    for key, grad in grads.items():
        g = grad * scale
        new_m[key] = b1 * m[key] + (1 - b1) * g
        new_v[key] = b2 * v[key] + (1 - b2) * g * g
        m_hat = new_m[key] / (1 - b1**t)
        v_hat = new_v[key] / (1 - b2**t)
        new_p[key] = p[key] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return new_p, new_m, new_v


@partial(jax.jit, static_argnames=("n_wash", "seq_len", "n_windows"))
def _train_segment(
    p: JaxParams,
    m: JaxParams,
    v: JaxParams,
    t: jax.Array,
    U: jax.Array,
    lr: jax.Array,
    clip: jax.Array,
    n_wash: int,
    seq_len: int,
    n_windows: int,
) -> tuple[JaxParams, JaxParams, JaxParams, jax.Array, jax.Array]:
    """Trains on one segment: a washout, then one Adam step per window.

    Args:
      p: The parameters.
      m: First moment estimates.
      v: Second moment estimates.
      t: Number of Adam steps taken so far.
      U: Normalized segment, shape `(T, N_dim_in, 1)`.
      lr: Step size.
      clip: Maximum global gradient norm.
      n_wash: Washout steps.
      seq_len: Window length.
      n_windows: Number of windows, `(T - n_wash - 1) // seq_len`.

    Returns:
      The updated parameters, moment estimates, and step count, and the sum of
      the window losses.
    """
    n_units = p["Uf"].shape[0]
    zeros = jnp.zeros((n_units, 1), U.dtype)
    _, h, c = _teacher_forced(p, U[:n_wash], zeros, zeros)

    span = n_windows * seq_len
    X = U[n_wash : n_wash + span].reshape(n_windows, seq_len, *U.shape[1:])
    Y_true = U[n_wash + 1 : n_wash + 1 + span].reshape(n_windows, seq_len, *U.shape[1:])

    def window(carry: tuple[Any, ...], xy: tuple[jax.Array, jax.Array]) -> tuple[tuple[Any, ...], jax.Array]:
        p, m, v, t, h, c = carry
        (loss, (h_new, c_new)), grads = jax.value_and_grad(_window_loss, has_aux=True)(p, xy[0], xy[1], h, c)
        t = t + 1
        p, m, v = _adam(p, grads, m, v, t, lr, clip)
        return (p, m, v, t, h_new, c_new), loss

    (p, m, v, t, _, _), losses = jax.lax.scan(window, (p, m, v, t, h, c), (X, Y_true))
    return p, m, v, t, jnp.sum(losses)


@partial(jax.jit, static_argnames=("n_wash", "n_steps"))
def _closed_loop_error(p: JaxParams, U: jax.Array, n_wash: int, n_steps: int) -> jax.Array:
    """Computes the closed-loop error on one validation segment.

    Args:
      p: The parameters.
      U: Normalized segment, shape `(n_wash + n_steps + 1, N_dim_in, 1)`.
      n_wash: Washout steps.
      n_steps: Closed-loop steps after the washout.

    Returns:
      The mean absolute error, as `LSTM.compute_nRMSE` computes it.
    """
    n_units = p["Uf"].shape[0]
    zeros = jnp.zeros((n_units, 1), U.dtype)
    Y_wash, h, c = _teacher_forced(p, U[:n_wash], zeros, zeros)
    Y_closed, _, _ = _closed_loop(p, Y_wash[-1], h, c, n_steps)
    Y_pred = jnp.concatenate((Y_wash[-1:], Y_closed), axis=0)
    return jnp.mean(jnp.abs(U[n_wash:] - Y_pred))


# ── Model ─────────────────────────────────────────────────────────────────────


@dataclass(eq=False, repr=False)
class LSTMJax(LSTM):
    """An `LSTM` that trains and forecasts with JAX.

    Starts from the same initial parameters as `LSTM` for the same options and
    seed, and in float64 trains to the same parameters up to rounding.

    Attributes:
      dtype: Precision to train and forecast in, `"float32"` or `"float64"`.
        float64 needs `jax.config.update("jax_enable_x64", True)`.

    Raises:
      ValueError: If an option is out of range or names an unknown method, or
        `dtype` is not float32 or float64.
    """

    _: KW_ONLY
    dtype: str = "float32"

    def __post_init__(self) -> None:
        """Validates the options and initializes the parameters.

        Raises:
          ValueError: If an option is out of range or names an unknown method,
            or `dtype` is not float32 or float64.
        """
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {self.dtype!r}")
        super().__post_init__()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, data: npt.ArrayLike | list[npt.ArrayLike], verbose: bool = True, **overrides: Any) -> None:
        """Trains the parameters on time series.

        Replaces `training_history`.

        Args:
          data: Time series, as an array of shape `[(L) x T x N_dim_in x (1)]`
            or a list of them.
          verbose: Print the training loss and validation error each epoch.
          **overrides: New values for `N_wash`, `seq_len`, `epochs`, `lr`,
            `clip`, `val_frac`, or `norm_method`.

        Raises:
          TypeError: If an override names another attribute.
          ValueError: If an override is out of range, a training segment is
            shorter than `N_wash + seq_len + 1`, or `dtype` is float64 and
            `jax_enable_x64` is off.
        """
        self._check_precision()
        U_train, U_val = self._training_segments(data, overrides)
        dt = jnp.dtype(self.dtype)
        segments = [jnp.asarray(U, dt) for U in U_train]
        n_windows = [(len(U) - self.N_wash - 1) // self.seq_len for U in U_train]
        val_segments = [jnp.asarray(U, dt) for U in U_val]

        p = self._device_params()
        m = {k: jnp.zeros_like(a) for k, a in p.items()}
        v = {k: jnp.zeros_like(a) for k, a in p.items()}
        t = jnp.zeros((), dt)
        lr, clip = jnp.asarray(self.lr, dt), jnp.asarray(self.clip, dt)
        best_loss, best_p = np.inf, None
        history = self.training_history = TrainingHistory()

        for epoch in range(self.epochs):
            total = 0.0
            for U, n in zip(segments, n_windows, strict=True):
                p, m, v, t, loss = _train_segment(p, m, v, t, U, lr, clip, self.N_wash, self.seq_len, n)
                total += float(loss)
            train_loss = total / max(sum(n_windows), 1)
            val_loss = self._closed_loop_error(p, val_segments) if val_segments else train_loss
            history.train.append(train_loss)
            history.val.append(val_loss)
            history.lr.append(self.lr)
            if val_loss < best_loss:
                best_loss, best_p = val_loss, p
            if verbose:
                print(f"epoch {epoch:3d}  loss {train_loss:.4e}  val nRMSE {val_loss:.4e}")

        if best_p is not None:
            self.p = {k: np.asarray(a) for k, a in best_p.items()}
        self._trained = True

    def _closed_loop_error(self, p: JaxParams, segments: list[jax.Array]) -> float:
        """Computes the mean closed-loop error over validation segments.

        Args:
          p: The parameters.
          segments: Normalized segments, each beginning with `N_wash` washout
            steps.

        Returns:
          The mean error, with non-finite errors counted as 1e10.
        """
        scores: list[float] = []
        for U in segments:
            score = float(_closed_loop_error(p, U, self.N_wash, U.shape[0] - self.N_wash - 1))
            scores.append(score if np.isfinite(score) else 1e10)
        return float(np.mean(scores))

    def _check_precision(self) -> None:
        """Checks that JAX can compute in the requested precision.

        Raises:
          ValueError: If `dtype` is float64 and `jax_enable_x64` is off.
        """
        if self.dtype == "float64" and not jax.config.jax_enable_x64:
            raise ValueError('dtype="float64" needs jax.config.update("jax_enable_x64", True) before training')

    def _device_params(self) -> JaxParams:
        """Copies the parameters to the device in `dtype`."""
        dt = jnp.dtype(self.dtype)
        return {k: jnp.asarray(a, dt) for k, a in self.p.items()}

    # ── Forecasting ───────────────────────────────────────────────────────────

    def openLoop(self, X_phys: npt.ArrayLike) -> tuple[Array, State]:
        """Runs the LSTM over true physical inputs.

        Args:
          X_phys: Physical inputs, shape `(T, N_dim_in)` or `(T, N_dim_in, 1)`.

        Returns:
          The physical readouts of shape `(T, N_dim_in, 1)` and the final hidden
          and cell states.
        """
        dt = jnp.dtype(self.dtype)
        X = jnp.asarray(self.normalize_input(self._as_sequences(X_phys)[0]), dt)
        zeros = jnp.zeros((self.N_units, 1), dt)
        Y, h, c = _teacher_forced_jit(self._device_params(), X, zeros, zeros)
        return self.denormalize_output(np.asarray(Y)), (np.asarray(h), np.asarray(c))

    def closedLoop(self, x0_phys: npt.ArrayLike, Nt: int, state: State) -> tuple[Array, State]:
        """Forecasts by feeding each readout back as the next input.

        Args:
          x0_phys: First physical input, `N_dim_in` values.
          Nt: Number of steps.
          state: Hidden and cell states, such as those `openLoop` returns.

        Returns:
          The physical forecast of shape `(Nt, N_dim_in, 1)` and the final
          states.
        """
        dt = jnp.dtype(self.dtype)
        x = jnp.asarray(self.normalize_input(np.asarray(x0_phys, dtype=float).reshape(self.N_dim_in, 1)), dt)
        h, c = (jnp.asarray(s, dt) for s in state)
        Y, h, c = _closed_loop_jit(self._device_params(), x, h, c, n_steps=Nt)
        return self.denormalize_output(np.asarray(Y)), (np.asarray(h), np.asarray(c))
