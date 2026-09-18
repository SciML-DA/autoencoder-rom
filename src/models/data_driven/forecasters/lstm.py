"""LSTM forecaster, standalone and as a `dynamodels.Model`.

`LSTM` is a single-layer LSTM in NumPy, trained by truncated backpropagation
through time with Adam. `LSTM_model` wraps it as a discrete-time `Model` whose
state is `psi = [u; h; c]`: the physical state, then the hidden and cell states.

Typical usage example:

  lstm = LSTM(N_dim_in=3, N_units=64, epochs=40)
  lstm.train(data)
  Y, state = lstm.openLoop(data[:100])
  forecast, _ = lstm.closedLoop(data[99], 500, state)

  model = LSTM_model(data=segments, dt=0.01, N_units=64)
  psi, t = model.time_integrate(Nt=500)
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field, fields
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from dynamodels import DiscreteIntegrator, Model

from ..configurable import Configurable
from ..training import TrainingHistory

__all__ = ["LSTM", "LSTM_model"]

#: A real-valued array. Each parameter documents its own shape.
Array = npt.NDArray[np.floating[Any]]
#: Network parameters by name: `W*`, `U*`, and `b*` for each gate, then `Wy`, `by`.
Params = dict[str, Array]
#: Values one step saves for backpropagation: `(x, h, c, c_new, tanh(c_new), f, i, g, o)`.
Cache = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]
#: Hidden and cell states.
State = tuple[Array, Array]

#: Gate suffixes: forget, input, candidate, and output.
_GATES = "figo"
_NORM_METHODS = (None, "std", "max", "mean", "range")
_RECURRENT_INITS = ("orthogonal", "uniform")
#: Options `LSTM.train` accepts as overrides.
_TRAIN_OPTIONS = ("N_wash", "seq_len", "epochs", "lr", "clip", "val_frac", "norm_method")


def _sigmoid(z: Array) -> Array:
    """Applies the logistic sigmoid elementwise."""
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(eq=False, repr=False)
class LSTM(Configurable):
    """A single-layer LSTM trained to predict the next sample of a time series.

    Every gate computes `W x + U h + b`; the forget, input, and output gates
    apply a sigmoid and the candidate gate `tanh`. A linear readout maps the
    hidden state to the next input. Inputs are normalized per component.

    Training runs a washout of `N_wash` steps on each segment, then truncated
    backpropagation over windows of `seq_len` steps with a one-step-ahead loss,
    and keeps the parameters of the epoch with the lowest closed-loop validation
    error on the last `val_frac` of each segment.

    States and inputs are column vectors, shape `(N, m)` with one column per
    ensemble member.

    Attributes:
      N_dim_in: Number of input components.
      N_units: Number of hidden units.
      seed: Seed for the parameter initialization.
      N_wash: Washout steps before training or forecasting.
      seq_len: Truncation length of backpropagation through time.
      epochs: Number of training epochs.
      lr: Adam step size.
      clip: Maximum global gradient norm.
      val_frac: Fraction of each segment held out for validation.
      forget_bias: Initial forget gate bias.
      norm_method: Input normalization. One of `None`, `"std"`, `"max"`,
        `"mean"`, or `"range"`.
      recurrent_init: Initialization of the recurrent weights, `"orthogonal"` or
        `"uniform"`.
      p: The parameters.
      training_history: Training loss, closed-loop validation error, and step
        size of each epoch of the last `train`.

    Raises:
      ValueError: If an option is out of range or names an unknown method.
    """

    N_dim_in: int
    N_units: int
    seed: int = 0
    _: KW_ONLY
    N_wash: int = 100
    seq_len: int = 200
    epochs: int = 40
    lr: float = 3e-3
    clip: float = 5.0
    val_frac: float = 0.1
    forget_bias: float = 1.0
    norm_method: str | None = "range"
    recurrent_init: str = "orthogonal"

    p: Params = field(init=False)
    training_history: TrainingHistory = field(default_factory=TrainingHistory, init=False)
    _trained: bool = field(default=False, init=False)
    _norm: Array = field(init=False)
    _shift: Array = field(init=False)

    def __post_init__(self) -> None:
        """Validates the options and initializes the parameters.

        Raises:
          ValueError: If an option is out of range or names an unknown method.
        """
        self._validate()
        self._norm = np.ones((self.N_dim_in, 1))
        self._shift = np.zeros((self.N_dim_in, 1))

        rng = np.random.default_rng(self.seed)
        k = 1.0 / np.sqrt(self.N_units)
        self.p = {}
        for gate in _GATES:
            self.p["W" + gate] = rng.uniform(-k, k, size=(self.N_units, self.N_dim_in))
            self.p["U" + gate] = self._recurrent_matrix(rng)
            self.p["b" + gate] = np.zeros((self.N_units, 1))
        self.p["bf"] += self.forget_bias
        self.p["Wy"] = rng.uniform(-k, k, size=(self.N_dim_in, self.N_units))
        self.p["by"] = np.zeros((self.N_dim_in, 1))

    def _validate(self) -> None:
        """Checks the options.

        Raises:
          ValueError: If an option is out of range or names an unknown method.
        """
        if self.N_dim_in < 1 or self.N_units < 1:
            raise ValueError(f"N_dim_in and N_units must be >= 1, got {self.N_dim_in}, {self.N_units}")
        if self.N_wash < 0 or self.seq_len < 1 or self.epochs < 1:
            raise ValueError(
                f"need N_wash >= 0, seq_len >= 1, epochs >= 1; got {self.N_wash}, {self.seq_len}, {self.epochs}"
            )
        if self.lr <= 0 or self.clip <= 0:
            raise ValueError(f"lr and clip must be > 0, got {self.lr}, {self.clip}")
        if not 0.0 <= self.val_frac < 1.0:
            raise ValueError(f"val_frac must be in [0, 1), got {self.val_frac}")
        if self.norm_method not in _NORM_METHODS:
            raise ValueError(f"norm_method must be one of {_NORM_METHODS}, got {self.norm_method!r}")
        if self.recurrent_init not in _RECURRENT_INITS:
            raise ValueError(f"recurrent_init must be one of {_RECURRENT_INITS}, got {self.recurrent_init!r}")

    def _recurrent_matrix(self, rng: np.random.Generator) -> Array:
        """Draws one recurrent weight matrix.

        Args:
          rng: Random generator.

        Returns:
          A matrix of shape `(N_units, N_units)`: orthogonal, or uniform in
          `[-1/sqrt(N_units), 1/sqrt(N_units)]`.
        """
        if self.recurrent_init == "orthogonal":
            Q, R = np.linalg.qr(rng.normal(size=(self.N_units, self.N_units)))
            return Q * np.sign(np.diag(R))
        k = 1.0 / np.sqrt(self.N_units)
        return rng.uniform(-k, k, size=(self.N_units, self.N_units))

    def copy(self) -> Self:
        """Returns a deep copy."""
        return deepcopy(self)

    @property
    def trained(self) -> bool:
        """Whether `train` has completed."""
        return self._trained

    # ── Normalization ─────────────────────────────────────────────────────────

    @property
    def norm(self) -> Array:
        """Per-component scale inputs are divided by, shape `(N_dim_in, 1)`."""
        return self._norm

    @norm.setter
    def norm(self, value: npt.ArrayLike) -> None:
        norm = np.asarray(value, dtype=float).reshape(self.N_dim_in, 1).copy()
        norm[np.abs(norm) < 1e-12] = 1.0
        self._norm = norm

    @property
    def shift(self) -> Array:
        """Per-component offset subtracted from inputs, shape `(N_dim_in, 1)`."""
        return self._shift

    @shift.setter
    def shift(self, value: npt.ArrayLike) -> None:
        self._shift = np.asarray(value, dtype=float).reshape(self.N_dim_in, 1).copy()

    def set_norm(self, data: npt.ArrayLike | list[npt.ArrayLike]) -> None:
        """Sets `norm` and `shift` from data.

        Args:
          data: Time series in any layout `train` accepts.
        """
        self.norm, self.shift = self._set_norm(self._as_sequences(data), method=self.norm_method)

    @staticmethod
    def _set_norm(sequences: list[Array], method: str | None = "range") -> tuple[Array, Array]:
        """Computes the per-component scale and offset of segments.

        Args:
          sequences: Segments, each of shape `(T, N_dim_in, 1)`.
          method: `None` for no normalization; `"std"`, `"max"`, `"mean"`, or
            `"range"` for the standard deviation, maximum, mean absolute value,
            or range of the centered data.

        Returns:
          The scale and the offset, each of shape `(N_dim_in, 1)`.

        Raises:
          ValueError: If `method` is unknown.
        """
        U = np.concatenate(sequences, axis=0)
        if method is None:
            return np.ones(U.shape[1:]), np.zeros(U.shape[1:])

        shift: Array = U.mean(axis=0)
        centered = U - shift
        if method == "std":
            norm: Array = centered.std(axis=0)
        elif method == "max":
            norm = centered.max(axis=0)
        elif method == "mean":
            norm = np.abs(centered).mean(axis=0)
        elif method == "range":
            norm = centered.max(axis=0) - centered.min(axis=0)
        else:
            raise ValueError(f"unknown normalization method {method!r}")
        return norm, shift

    def normalize_input(self, data: Array) -> Array:
        """Normalizes physical values.

        Args:
          data: Values with components along the second-to-last axis of size
            `N_dim_in`, such as `(N_dim_in, m)`.

        Returns:
          `(data - shift) / norm`.
        """
        return (data - self.shift) / self.norm

    def denormalize_output(self, data: Array) -> Array:
        """Maps normalized values back to physical values.

        Args:
          data: Normalized values, in the layout `normalize_input` accepts.

        Returns:
          `data * norm + shift`.
        """
        return data * self.norm + self.shift

    def _as_sequences(self, data: npt.ArrayLike | list[npt.ArrayLike]) -> list[Array]:
        """Splits time series into segments of column vectors.

        Args:
          data: A list of series, or an array of shape `(T, N_dim_in)`,
            `(T, N_dim_in, 1)`, `(L, T, N_dim_in)`, or `(L, T, N_dim_in, 1)`.

        Returns:
          One array of shape `(T, N_dim_in, 1)` per segment.

        Raises:
          ValueError: If the layout does not match `N_dim_in`.
        """
        if isinstance(data, list):
            items: list[npt.ArrayLike] = data
            return [s for d in items for s in self._as_sequences(d)]

        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ..., np.newaxis]
        elif arr.ndim == 3 and arr.shape[1] == self.N_dim_in and arr.shape[-1] == 1:
            arr = arr[np.newaxis]
        elif arr.ndim == 3:
            arr = arr[..., np.newaxis]
        if arr.ndim != 4 or arr.shape[2:] != (self.N_dim_in, 1):
            raise ValueError(f"data has shape {np.shape(data)}, expected [(L) x T x {self.N_dim_in} x (1)]")
        return list(arr)

    # ── Forward and backward ──────────────────────────────────────────────────

    def step(self, x: Array, h: Array, c: Array) -> tuple[Array, Array, Array, Cache]:
        """Advances the LSTM one step.

        Args:
          x: Normalized input, shape `(N_dim_in, m)` or `(N_dim_in,)`.
          h: Hidden state, shape `(N_units, m)`.
          c: Cell state, shape `(N_units, m)`.

        Returns:
          The new hidden state, the new cell state, the normalized readout of
          shape `(N_dim_in, m)`, and the values backpropagation needs.

        Raises:
          ValueError: If a shape does not match the network.
        """
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if x.shape[0] != self.N_dim_in or h.shape[0] != self.N_units or c.shape != h.shape:
            raise ValueError(
                f"x, h, c must be ({self.N_dim_in}, m), ({self.N_units}, m), ({self.N_units}, m); "
                f"got {x.shape}, {h.shape}, {c.shape}"
            )
        return self._forward_step(x, h, c)

    def _forward_step(self, x: Array, h: Array, c: Array) -> tuple[Array, Array, Array, Cache]:
        """Advances the LSTM one step and saves the values backpropagation needs.

        Args:
          x: Normalized input, shape `(N_dim_in, m)`.
          h: Hidden state, shape `(N_units, m)`.
          c: Cell state, shape `(N_units, m)`.

        Returns:
          The new hidden state, the new cell state, the readout, and the cache.
        """
        p = self.p
        f = _sigmoid(p["Wf"] @ x + p["Uf"] @ h + p["bf"])
        i = _sigmoid(p["Wi"] @ x + p["Ui"] @ h + p["bi"])
        g = np.tanh(p["Wg"] @ x + p["Ug"] @ h + p["bg"])
        o = _sigmoid(p["Wo"] @ x + p["Uo"] @ h + p["bo"])
        c_new = f * c + i * g
        tc = np.tanh(c_new)
        h_new = o * tc
        y = p["Wy"] @ h_new + p["by"]
        return h_new, c_new, y, (x, h, c, c_new, tc, f, i, g, o)

    def _predict_step(self, x: Array, h: Array, c: Array) -> tuple[Array, Array, Array]:
        """Advances the LSTM one step without saving values for backpropagation.

        Args:
          x: Normalized input, shape `(N_dim_in, m)`.
          h: Hidden state, shape `(N_units, m)`.
          c: Cell state, shape `(N_units, m)`.

        Returns:
          The new hidden state, the new cell state, and the readout.
        """
        p = self.p
        f = _sigmoid(p["Wf"] @ x + p["Uf"] @ h + p["bf"])
        i = _sigmoid(p["Wi"] @ x + p["Ui"] @ h + p["bi"])
        g = np.tanh(p["Wg"] @ x + p["Ug"] @ h + p["bg"])
        o = _sigmoid(p["Wo"] @ x + p["Uo"] @ h + p["bo"])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new, p["Wy"] @ h_new + p["by"]

    def forward(self, X: Array, h0: Array | None = None, c0: Array | None = None) -> tuple[Array, list[Cache], State]:
        """Runs the LSTM over a sequence of true inputs.

        Args:
          X: Normalized inputs, shape `(T, N_dim_in, 1)`.
          h0: Initial hidden state, shape `(N_units, 1)`. `None` uses zeros.
          c0: Initial cell state, shape `(N_units, 1)`. `None` uses zeros.

        Returns:
          The readouts of shape `(T, N_dim_in, 1)`, one cache per step, and the
          final hidden and cell states.

        Raises:
          ValueError: If a shape does not match the network.
        """
        h = np.zeros((self.N_units, 1)) if h0 is None else h0
        c = np.zeros((self.N_units, 1)) if c0 is None else c0
        if X.shape[1:] != (self.N_dim_in, 1) or h.shape != (self.N_units, 1) or c.shape != (self.N_units, 1):
            raise ValueError(
                f"X, h0, c0 must be (T, {self.N_dim_in}, 1), ({self.N_units}, 1), ({self.N_units}, 1); "
                f"got {X.shape}, {h.shape}, {c.shape}"
            )

        Y: Array = np.empty((X.shape[0], self.N_dim_in, 1))
        caches: list[Cache] = []
        for t in range(X.shape[0]):
            h, c, Y[t], cache = self._forward_step(X[t], h, c)
            caches.append(cache)
        return Y, caches, (h, c)

    def backward(self, caches: list[Cache], dY: Array) -> Params:
        """Backpropagates through a forward pass.

        Args:
          caches: The caches `forward` returned.
          dY: Gradient of the loss with respect to each readout, shape
            `(T, N_dim_in, 1)`.

        Returns:
          The gradient of the loss with respect to each parameter.
        """
        p = self.p
        grads: Params = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next: Array = np.zeros((self.N_units, 1))
        dc_next: Array = np.zeros((self.N_units, 1))

        for t in reversed(range(len(caches))):
            x, h_prev, c_prev, _, tc, f, i, g, o = caches[t]
            grads["Wy"] += dY[t] @ (o * tc).T
            grads["by"] += dY[t]

            dh = p["Wy"].T @ dY[t] + dh_next
            do = dh * tc
            dc = dh * o * (1 - tc**2) + dc_next
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_next = dc * f

            daf = df * f * (1 - f)
            dai = di * i * (1 - i)
            dao = do * o * (1 - o)
            dag = dg * (1 - g**2)
            for gate, da in zip(_GATES, (daf, dai, dag, dao), strict=True):
                grads["W" + gate] += da @ x.T
                grads["U" + gate] += da @ h_prev.T
                grads["b" + gate] += da
            dh_next = p["Uf"].T @ daf + p["Ui"].T @ dai + p["Ug"].T @ dag + p["Uo"].T @ dao
        return grads

    def _rollout(self, x: Array, Nt: int, state: State) -> tuple[Array, State]:
        """Forecasts in normalized space by feeding each readout back as the next input.

        Args:
          x: Normalized first input, shape `(N_dim_in, 1)`.
          Nt: Number of steps.
          state: Hidden and cell states.

        Returns:
          The readouts of shape `(Nt, N_dim_in, 1)` and the final states.
        """
        h, c = state
        out: Array = np.empty((Nt, self.N_dim_in, 1))
        for t in range(Nt):
            h, c, x = self._predict_step(x, h, c)
            out[t] = x
        return out, (h, c)

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
          ValueError: If an override is out of range, or a training segment is
            shorter than `N_wash + seq_len + 1`.
        """
        U_train, U_val = self._training_segments(data, overrides)
        moments = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.p.items()}
        n_steps = 0
        best_loss, best_p = np.inf, None
        history = self.training_history = TrainingHistory()

        for epoch in range(self.epochs):
            total, n_windows = 0.0, 0
            for U in U_train:
                h: Array = np.zeros((self.N_units, 1))
                c: Array = np.zeros((self.N_units, 1))
                for t in range(self.N_wash):
                    h, c, _ = self._predict_step(U[t], h, c)

                start = self.N_wash
                while start + self.seq_len + 1 <= len(U):
                    X = U[start : start + self.seq_len]
                    Y_true = U[start + 1 : start + self.seq_len + 1]
                    Y, caches, (h, c) = self.forward(X, h, c)

                    residual = Y - Y_true
                    total += 0.5 * np.mean(residual**2)
                    n_windows += 1
                    n_steps += 1
                    self._adam_step(
                        self.backward(caches, residual / (residual.shape[0] * self.N_dim_in)), moments, n_steps
                    )

                    h, c = h.copy(), c.copy()
                    start += self.seq_len

            train_loss = total / max(n_windows, 1)
            val_loss = self._validate_closed_loop(U_val) if U_val else train_loss
            history.train.append(float(train_loss))
            history.val.append(float(val_loss))
            history.lr.append(self.lr)
            if val_loss < best_loss:
                best_loss, best_p = val_loss, {k: v.copy() for k, v in self.p.items()}
            if verbose:
                print(f"epoch {epoch:3d}  loss {train_loss:.4e}  val nRMSE {val_loss:.4e}")

        if best_p is not None:
            self.p = best_p
        self._trained = True

    def _training_segments(
        self, data: npt.ArrayLike | list[npt.ArrayLike], overrides: Mapping[str, Any]
    ) -> tuple[list[Array], list[Array]]:
        """Applies overrides, sets the normalization, and splits off validation.

        Args:
          data: Time series, in any layout `train` accepts.
          overrides: New option values, as `train` takes them.

        Returns:
          The normalized training segments, and the normalized validation
          segments, each beginning with `N_wash` washout steps.

        Raises:
          TypeError: If an override names another attribute.
          ValueError: If an override is out of range, or a training segment is
            shorter than `N_wash + seq_len + 1`.
        """
        for key, value in overrides.items():
            if key not in _TRAIN_OPTIONS:
                raise TypeError(f"train() got an unexpected option {key!r}")
            setattr(self, key, value)
        self._validate()

        sequences = self._as_sequences(data)
        n_val = [int(round(self.val_frac * len(U))) for U in sequences]
        train_raw = [U[: len(U) - n] for U, n in zip(sequences, n_val, strict=True)]
        val_raw = [
            U[len(U) - n - self.N_wash :]
            for U, n in zip(sequences, n_val, strict=True)
            if n > 0 and len(U) - n >= self.N_wash
        ]
        for index, U in enumerate(train_raw):
            if len(U) < self.N_wash + self.seq_len + 1:
                raise ValueError(
                    f"segment {index} has {len(U)} training steps, fewer than "
                    f"N_wash + seq_len + 1 = {self.N_wash + self.seq_len + 1}"
                )

        self.norm, self.shift = self._set_norm(train_raw, method=self.norm_method)
        return [self.normalize_input(U) for U in train_raw], [self.normalize_input(U) for U in val_raw]

    def _adam_step(
        self,
        grads: Params,
        moments: dict[str, list[Array]],
        n_steps: int,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Clips the gradients by global norm and applies one Adam step in place.

        Args:
          grads: Gradients, keyed like `p`.
          moments: First and second moment estimates, keyed like `p`, updated in
            place.
          n_steps: Number of steps taken, including this one.
          b1: Decay rate of the first moment.
          b2: Decay rate of the second moment.
          eps: Constant added to the denominator.
        """
        squared: float = sum(float((g**2).sum()) for g in grads.values())
        grad_norm = np.sqrt(squared)
        scale = self.clip / max(grad_norm, self.clip)
        for key, grad in grads.items():
            g = grad * scale
            m, v = moments[key]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1**n_steps)
            v_hat = v / (1 - b2**n_steps)
            self.p[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def _validate_closed_loop(self, sequences: list[Array]) -> float:
        """Computes the mean closed-loop error over validation segments.

        Args:
          sequences: Normalized segments, each beginning with `N_wash` washout
            steps.

        Returns:
          The mean of `compute_nRMSE` over segments, with non-finite errors
          counted as 1e10.
        """
        scores: list[float] = []
        for U in sequences:
            Y_wash, _, state = self.forward(U[: self.N_wash])
            Y_closed, _ = self._rollout(Y_wash[-1], len(U) - self.N_wash - 1, state)
            score = self.compute_nRMSE(U[self.N_wash :], np.concatenate((Y_wash[-1:], Y_closed), axis=0))
            scores.append(score if np.isfinite(score) else 1e10)
        return float(np.mean(scores))

    def compute_nRMSE(self, Y_true: Array, Y_pred: Array, norm: float = 1.0) -> float:
        """Computes the error `EchoStateNetwork.compute_nRMSE` computes.

        Args:
          Y_true: True values.
          Y_pred: Predicted values, shaped like `Y_true`.
          norm: Normalization constant.

        Returns:
          The mean absolute error divided by `abs(norm)`.
        """
        return float(np.mean(np.sqrt((Y_true - Y_pred) ** 2)) / np.mean(np.sqrt(np.asarray(norm, dtype=float) ** 2)))

    # ── Saving and restoring ──────────────────────────────────────────────────

    @classmethod
    def resolve_options(cls, data: Any, **options: Any) -> dict[str, Any]:
        """Computes the options an LSTM trained on `data` reports.

        Args:
          data: Time series, shape `(T, N_dim_in)` or `(L, T, N_dim_in)`.
          **options: Constructor options. `N_dim_in` defaults to the last axis
            of `data` when `data` is given.

        Returns:
          The options, as `config_options` reports them.
        """
        if data is not None:
            options.setdefault("N_dim_in", np.shape(data)[-1])
        return super().resolve_options(data, **options)

    @classmethod
    def from_data(cls, data: Any, **options: Any) -> Self:
        """Builds and trains an LSTM without printing progress.

        Args:
          data: Time series, shape `(T, N_dim_in)` or `(L, T, N_dim_in)`.
          **options: Constructor options. `N_dim_in` defaults to the last axis
            of `data`.

        Returns:
          The trained LSTM.
        """
        options.setdefault("N_dim_in", np.shape(data)[-1])
        model = cls(**options)
        model.train(data, verbose=False)
        return model

    def trained_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects the parameters, normalization, and history.

        Returns:
          Each parameter as `p.<name>`, then `norm`, `shift`, and the history.

        Raises:
          RuntimeError: If the LSTM is not trained.
        """
        if not self.trained:
            raise RuntimeError("LSTM is not trained")
        params = {f"p.{name}": value for name, value in self.p.items()}
        return params | {"norm": self.norm, "shift": self.shift} | self.training_history.to_arrays()

    @classmethod
    def from_trained(cls, options: Mapping[str, Any], arrays: Mapping[str, npt.NDArray[Any]], **overrides: Any) -> Self:
        """Rebuilds a trained LSTM.

        Args:
          options: Options from `config_options`. Options the class no longer
            defines are ignored.
          arrays: Arrays from `trained_arrays`.
          **overrides: Options to use instead of the stored ones.

        Returns:
          The trained LSTM.
        """
        model = cls(**cls._known_options(options, overrides))
        model.p = {name: np.array(arrays[f"p.{name}"]) for name in model.p}
        model.norm, model.shift = arrays["norm"], arrays["shift"]
        model.training_history = TrainingHistory.from_arrays(arrays)
        model._trained = True
        return model

    # ── Forecasting ───────────────────────────────────────────────────────────

    def openLoop(self, X_phys: npt.ArrayLike) -> tuple[Array, State]:
        """Runs the LSTM over true physical inputs.

        Args:
          X_phys: Physical inputs, shape `(T, N_dim_in)` or `(T, N_dim_in, 1)`.

        Returns:
          The physical readouts of shape `(T, N_dim_in, 1)` and the final hidden
          and cell states.
        """
        Y, _, state = self.forward(self.normalize_input(self._as_sequences(X_phys)[0]))
        return self.denormalize_output(Y), state

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
        x = np.asarray(x0_phys, dtype=float).reshape(self.N_dim_in, 1)
        out, state = self._rollout(self.normalize_input(x), Nt, state)
        return self.denormalize_output(out), state


class LSTM_model(LSTM, Model):
    """An LSTM as a discrete-time `dynamodels.Model`.

    The state is `psi = [u; h; c]`, of size `N_dim + 2 * N_units`. Supports
    ensembles, one member per column. Construction trains the LSTM, so
    `config.model_config` stores `LSTM` rather than this class.

    Attributes:
      update_state: Include the physical state in `state_labels`.
      update_memory: Include the hidden and cell states in `state_labels`.
      t_train: Training time. `None` uses 80% of the data.
      t_val: Validation time. `None` uses 20% of `t_train`.
      t_test: Test time.
      N_dim: Number of physical components.
    """

    update_state: bool = True
    update_memory: bool = True
    t_train: float | None = None
    t_val: float | None = None
    t_test: float = 0.0
    N_units: int = 50
    N_wash: int = 5
    extra_print_params = ["N_units", "N_wash", "seq_len", "epochs", "lr"]
    configurable = False

    #: Options set on the model before the LSTM is built.
    MODEL_OPTIONS = (
        "update_state",
        "update_memory",
        "t_train",
        "t_val",
        "t_test",
        "N_units",
        "N_wash",
        "extra_print_params",
    )

    def __init__(
        self,
        data: npt.ArrayLike,
        dt: float = 1.0,
        y0: npt.ArrayLike | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Trains the LSTM and initializes the model at `y0`.

        Args:
          data: Training trajectories, shape `(L, Nt, N_dim)` or `(Nt, N_dim)`.
          dt: Time step of the data.
          y0: Initial physical state. `None` uses the first sample.
          verbose: Print the training loss each epoch.
          **kwargs: Options in `MODEL_OPTIONS`, then `LSTM` options, then
            options of `Model.__init__`. `seed` defaults to the current `seed`.

        Raises:
          ValueError: If `data` is not two- or three-dimensional, or an LSTM
            option is invalid.
        """
        for key in [k for k in kwargs if k in self.MODEL_OPTIONS]:
            setattr(self, key, kwargs.pop(key))

        segments = np.asarray(data, dtype=float)
        if segments.ndim == 2:
            segments = segments[np.newaxis]
        if segments.ndim != 3:
            raise ValueError(f"data must be (L, Nt, N_dim) or (Nt, N_dim), got shape {segments.shape}")
        n_dim = segments.shape[-1]

        lstm_options = {k: kwargs.pop(k) for k in list(kwargs) if k in _LSTM_OPTIONS}
        lstm_options.setdefault("N_wash", self.N_wash)
        LSTM.__init__(self, n_dim, self.N_units, seed=kwargs.pop("seed", self.seed), **lstm_options)
        self.train(segments, verbose=verbose)

        t_total = segments.shape[1] * dt
        t_train = self.t_train or t_total * 0.8
        t_val = self.t_val or t_train * 0.2
        self.t_train, self.t_val = t_train, t_val
        self.t_transient = t_train + t_val + self.t_test
        self.t_CR = 10 * dt

        self.N_dim = n_dim
        if not hasattr(self, "Nq"):
            self.Nq = n_dim
        u0 = segments[0, 0] if y0 is None else np.asarray(y0, dtype=float)
        psi0 = self.build_psi(u=u0.reshape(n_dim, 1), h=np.zeros((self.N_units, 1)), c=np.zeros((self.N_units, 1)))
        Model.__init__(self, dt=dt, psi0=psi0, integrator_class=DiscreteIntegrator, **kwargs)

    # ── Model interface ───────────────────────────────────────────────────────

    @property
    def dt_step(self) -> float:
        """Time step of one forecast step."""
        return self.dt

    @property
    def obs_labels(self) -> list[str]:
        """Labels of the physical components."""
        return [f"$u_{j + 1}$" for j in range(self.N_dim)]

    @property
    def forecaster_state_labels(self) -> list[str]:
        """Labels of the hidden and cell states, or none when `update_memory` is off."""
        if not self.update_memory:
            return []
        return [f"$h_{j + 1}$" for j in range(self.N_units)] + [f"$c_{j + 1}$" for j in range(self.N_units)]

    @property
    def state_labels(self) -> list[str]:
        """Labels of the state vector."""
        physical = [f"$u_{j + 1}$" for j in range(self.N_dim)] if self.update_state else []
        return physical + self.forecaster_state_labels

    def get_observables(self, Nt: int = 1, **kwargs: Any) -> Array:
        """Reads the latest physical states from the history.

        Args:
          Nt: Number of most recent time steps.
          **kwargs: Ignored.

        Returns:
          Physical states, shape `(N_dim, m)` for `Nt=1`, otherwise
          `(Nt, N_dim, m)`.
        """
        del kwargs
        if Nt == 1:
            return self.hist[-1, : self.N_dim]
        return self.hist[-Nt:, : self.N_dim]

    def build_psi(self, u: Array, h: Array, c: Array) -> Array:
        """Stacks physical, hidden, and cell states into `psi`.

        Args:
          u: Physical states, shape `(N_dim, m)` or `(Nt, N_dim, m)`.
          h: Hidden states, in the matching layout.
          c: Cell states, in the matching layout.

        Returns:
          The state vector, stacked along the component axis.

        Raises:
          ValueError: If `u` is neither two- nor three-dimensional.
        """
        if u.ndim not in (2, 3):
            raise ValueError(f"u must be 2- or 3-dimensional, got {u.ndim}")
        return np.concatenate((u, h, c), axis=u.ndim - 2)

    def unbuild_psi(self, psi: Array | None = None) -> tuple[Array, Array, Array]:
        """Splits `psi` into physical, hidden, and cell states.

        Args:
          psi: State vector, shape `(Nphi, m)` or `(Nt, Nphi, m)`. `None` uses
            the current state.

        Returns:
          The physical, hidden, and cell states.
        """
        state: Array = self.current_state if psi is None else psi
        n, k = self.N_dim, self.N_units
        blocks = state if state.ndim == 2 else state.transpose(1, 0, 2)
        u, h, c = blocks[:n], blocks[n : n + k], blocks[n + k : n + 2 * k]
        if state.ndim == 2:
            return u, h, c
        return u.transpose(1, 0, 2), h.transpose(1, 0, 2), c.transpose(1, 0, 2)

    def time_step(self, Nt: int = 10, averaged: bool = False) -> tuple[Array, Array]:
        """Forecasts `Nt` steps from the current state.

        Args:
          Nt: Number of steps.
          averaged: Forecast the ensemble mean and keep each member's initial
            deviation from it, instead of forecasting each member.

        Returns:
          The states, shape `(Nt + 1, Nphi, m)`, including the current state, and
          their times.

        Raises:
          RuntimeError: If the LSTM is not trained.
        """
        if not self.trained:
            raise RuntimeError("LSTM model is not trained")

        t: Array = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt, self.precision_t)
        u0, h0, c0 = self.unbuild_psi(self.current_state)
        m = u0.shape[-1]
        u: Array = np.empty((Nt + 1, self.N_dim, m))
        h: Array = np.empty((Nt + 1, self.N_units, m))
        c: Array = np.empty((Nt + 1, self.N_units, m))
        u[0], h[0], c[0] = u0, h0, c0

        if averaged:
            u_mean, h_mean, c_mean = (a.mean(axis=-1, keepdims=True) for a in (u0, h0, c0))
            u_dev, h_dev, c_dev = u0 - u_mean, h0 - h_mean, c0 - c_mean
            for i in range(Nt):
                h_mean, c_mean, y = self._predict_step(self.normalize_input(u_mean), h_mean, c_mean)
                u_mean = self.denormalize_output(y)
                u[i + 1], h[i + 1], c[i + 1] = u_mean + u_dev, h_mean + h_dev, c_mean + c_dev
        else:
            for i in range(Nt):
                h[i + 1], c[i + 1], y = self._predict_step(self.normalize_input(u[i]), h[i], c[i])
                u[i + 1] = self.denormalize_output(y)

        return self.build_psi(u=u, h=h, c=c), t

    def reset_forecaster(self, data: Array | None = None, u0: npt.ArrayLike | None = None, **kwargs: Any) -> None:
        """Restarts the model from `u0` with zero hidden and cell states.

        Keeps the trained parameters.

        Args:
          data: Ignored; accepted for the `ESN_model.reset_forecaster` signature.
          u0: Initial physical state, `N_dim` values or shape `(N_dim, m)`.
            `None` keeps the current physical state.
          **kwargs: Ignored.
        """
        del data, kwargs
        if u0 is None:
            u: Array = self.current_state[: self.N_dim]
        else:
            u = np.asarray(u0, dtype=float).reshape(self.N_dim, -1)
        zeros: Array = np.zeros((self.N_units, u.shape[1]))
        psi = self.build_psi(u=u, h=zeros, c=zeros.copy())
        self.update_history(psi[np.newaxis], t=np.array([self.current_time]), reset=True)


#: `LSTM` options `LSTM_model` passes on, besides the dimensions and seed it sets itself.
_LSTM_OPTIONS = frozenset(f.name for f in fields(LSTM) if f.init) - {"N_dim_in", "N_units", "seed"}
