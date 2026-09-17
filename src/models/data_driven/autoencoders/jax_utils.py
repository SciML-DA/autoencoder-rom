"""Base class, optimizer, and training loop shared by the JAX autoencoders.

Typical usage example:

  @dataclass(eq=False, repr=False)
  class MyAE(JaxAutoencoder[MyParams]):
      def fit(self, X):
          Q, _ = self._prepare_fit(X)
          self._configure()
          self.weights, history = self._train(data, self._init_params, step, loss)
          self._finish_fit(history)
          return self
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.typing import DTypeLike

from ..training import TrainingHistory
from .base import Autoencoder

__all__ = [
    "ACTIVATIONS",
    "Activation",
    "ActivationSpec",
    "AdamState",
    "JaxAutoencoder",
    "Layer",
    "activation_fn",
    "adam_init",
    "adam_update",
    "epoch_batches",
    "register_activation",
    "resolve_activations",
    "uniform",
]

#: An elementwise activation.
type Activation = Callable[[jax.Array], jax.Array]
#: One activation for every layer, or one per layer, by name or as a function.
type ActivationSpec = str | Activation | Sequence[str | Activation]
#: Runs one optimizer step on the batch at a row of the batch index array.
type StepFn[P] = Callable[[P, AdamState[P], jax.Array, jax.Array, int, float], tuple[P, AdamState[P], jax.Array]]
#: Computes the loss over a block of snapshots.
type LossFn[P] = Callable[[P, jax.Array], jax.Array]


class Layer(TypedDict):
    """Weights and bias of one dense or convolutional layer."""

    W: jax.Array
    b: jax.Array


# ── Activations ───────────────────────────────────────────────────────────────


def _identity(x: jax.Array) -> jax.Array:
    """Returns the input unchanged."""
    return x


#: Activations available by name.
ACTIVATIONS: dict[str, Activation] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "identity": _identity,
}


def register_activation(name: str, fn: Activation) -> None:
    """Makes an activation available by name.

    A model whose activations are all named can be saved.

    Args:
      name: The name options refer to it by.
      fn: The activation.
    """
    ACTIVATIONS[name] = fn


def activation_fn(a: str | Activation) -> Activation:
    """Looks up an activation.

    Args:
      a: An activation name or function.

    Returns:
      The activation function.
    """
    return ACTIVATIONS[a] if isinstance(a, str) else a


def resolve_activations(spec: ActivationSpec, n_layers: int, what: str) -> tuple[str | Activation, ...]:
    """Expands an activation option to one entry per activated layer.

    Args:
      spec: One activation for every layer, or a sequence with one per layer.
      n_layers: Number of activated layers.
      what: Option name, for error messages.

    Returns:
      One activation name or function per layer.

    Raises:
      ValueError: If a sequence has the wrong length or names an unknown
        activation.
      TypeError: If an entry is neither a name nor a callable.
    """
    acts: tuple[str | Activation, ...] = (spec,) * n_layers if isinstance(spec, str) or callable(spec) else tuple(spec)
    if len(acts) != n_layers:
        raise ValueError(f"{what}: got {len(acts)} activations for {n_layers} activated layers")
    for a in acts:
        if isinstance(a, str):
            if a not in ACTIVATIONS:
                raise ValueError(f"{what}: unknown activation {a!r}; choose from {sorted(ACTIVATIONS)}")
        elif not callable(a):
            raise TypeError(f"{what}: activation must be a name or a callable, got {a!r}")
    return acts


# ── Optimizer ─────────────────────────────────────────────────────────────────


class AdamState[P](NamedTuple):
    """Moment estimates and step count for Adam.

    Attributes:
      m: First-moment estimates, shaped like the parameters.
      v: Second-moment estimates, shaped like the parameters.
      t: Number of steps taken.
    """

    m: P
    v: P
    t: jax.Array


def adam_init[P](params: P) -> AdamState[P]:
    """Creates a zeroed Adam state.

    Args:
      params: Parameter tree.

    Returns:
      Zero moment estimates shaped like `params`, and a step count of 0.
    """

    def zeros_like(x: jax.Array) -> jax.Array:
        """Returns zeros shaped like one parameter."""
        return jnp.zeros_like(x)

    zeros: P = jax.tree.map(zeros_like, params)
    return AdamState(m=zeros, v=zeros, t=jnp.zeros((), dtype=jnp.int32))


def adam_update[P](
    grads: P,
    state: AdamState[P],
    params: P,
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    wd: float = 0.0,
) -> tuple[P, AdamState[P]]:
    """Applies one bias-corrected Adam step with L2 weight decay.

    Each parameter stays in its own dtype.

    Args:
      grads: Gradients, shaped like `params`.
      state: Adam state from the previous step.
      params: Parameter tree.
      lr: Learning rate.
      b1: Decay rate of the first-moment estimate.
      b2: Decay rate of the second-moment estimate.
      eps: Constant added to the denominator.
      wd: L2 penalty coefficient, added to the gradients.

    Returns:
      The updated parameters and Adam state.
    """
    t = state.t + 1
    bc1 = 1.0 - b1**t
    bc2 = 1.0 - b2**t

    def decay(g: jax.Array, p: jax.Array) -> jax.Array:
        """Adds the L2 penalty gradient."""
        return g + wd * p

    def first_moment(m: jax.Array, g: jax.Array) -> jax.Array:
        """Updates one first-moment estimate."""
        return b1 * m + (1.0 - b1) * g

    def second_moment(v: jax.Array, g: jax.Array) -> jax.Array:
        """Updates one second-moment estimate."""
        return b2 * v + (1.0 - b2) * g * g

    def step(p: jax.Array, m: jax.Array, v: jax.Array) -> jax.Array:
        """Applies the bias-corrected update to one parameter."""
        dt = p.dtype
        return p - jnp.asarray(lr, dt) * (m / bc1.astype(dt)) / (jnp.sqrt(v / bc2.astype(dt)) + eps)

    if wd:
        grads = jax.tree.map(decay, grads, params)
    m: P = jax.tree.map(first_moment, state.m, grads)
    v: P = jax.tree.map(second_moment, state.v, grads)
    updated: P = jax.tree.map(step, params, m, v)
    return updated, AdamState(m, v, t)


# ── Initialization and batching ───────────────────────────────────────────────


def uniform(key: jax.Array, shape: tuple[int, ...], fan_in: int, dtype: DTypeLike) -> jax.Array:
    """Draws weights uniformly from `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`.

    Args:
      key: A `jax.random` key.
      shape: Shape of the weights.
      fan_in: Number of inputs to each unit.
      dtype: Scalar type of the weights.

    Returns:
      The weights.
    """
    k = 1.0 / np.sqrt(fan_in)
    return jax.random.uniform(key, shape, dtype=dtype, minval=-k, maxval=k)


def epoch_batches(key: jax.Array, n: int, batch_size: int) -> jax.Array:
    """Shuffles `n` indices into equal batches.

    Drops the final partial batch, so every batch has the same shape.

    Args:
      key: A `jax.random` key.
      n: Number of indices.
      batch_size: Indices per batch.

    Returns:
      Shuffled indices, shape `(n_batches, batch_size)`, or one batch of all `n`
      indices when `n` is below `batch_size`.
    """
    perm = jax.random.permutation(key, n)
    n_batches = max(n // batch_size, 1)
    return perm[: n_batches * batch_size].reshape(n_batches, -1)


# ── Base class ────────────────────────────────────────────────────────────────


@dataclass(eq=False, repr=False)
class JaxAutoencoder[P](Autoencoder):
    """Options, training loop, and weight saving shared by `AEJax` and `CAEJax`.

    Attributes:
      activation: Activation of the encoder's hidden layers, as a name or
        function, or a sequence with one per layer.
      layer_activations: Activation of the decoder's hidden layers, in the same
        form. `None` uses `activation`.
      dtype: Precision of the parameters, `"float32"` or `"float64"`.
      weights: The trained parameters, or `None` before `fit`.

    Raises:
      ValueError: If an option is out of range or `dtype` is not float32 or
        float64.
    """

    activation: ActivationSpec = "tanh"
    layer_activations: ActivationSpec | None = None
    dtype: DTypeLike = "float32"

    weights: P | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range or `dtype` is not float32 or
            float64.
        """
        super().__post_init__()
        self.dtype = np.dtype(self.dtype)
        if self.dtype.name not in ("float32", "float64"):
            raise ValueError(f"dtype must be float32 or float64, got {self.dtype.name}")
        if not isinstance(self.activation, str) and not callable(self.activation):
            self.activation = tuple(self.activation)
        spec = self.layer_activations
        if spec is not None and not isinstance(spec, str) and not callable(spec):
            self.layer_activations = tuple(spec)

    @property
    def n_params(self) -> int:
        """Number of trainable parameters, or 0 before `fit`."""
        if self.weights is None:
            return 0
        leaves: list[jax.Array] = jax.tree.leaves(self.weights)
        return int(sum(a.size for a in leaves))

    @abstractmethod
    def _configure(self) -> None:
        """Builds the static architecture from the recorded grid geometry."""

    @abstractmethod
    def _init_params(self, key: jax.Array) -> P:
        """Draws initial parameters.

        Args:
          key: A `jax.random` key.

        Returns:
          The parameters.
        """

    def _trained_params(self) -> P:
        """Returns the trained parameters.

        Raises:
          RuntimeError: If `fit` has not completed.
        """
        self._check_fitted()
        if self.weights is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        return self.weights

    def _check_precision(self) -> None:
        """Checks that JAX can compute in the requested precision.

        Raises:
          ValueError: If `dtype` is float64 and `jax_enable_x64` is off.
        """
        if np.dtype(self.dtype).name == "float64" and not jax.config.jax_enable_x64:
            raise ValueError('dtype="float64" needs jax.config.update("jax_enable_x64", True) before fitting')

    def _train(
        self, data: jax.Array, init: Callable[[jax.Array], P], step: StepFn[P], loss: LossFn[P]
    ) -> tuple[P, TrainingHistory]:
        """Trains parameters with Adam, a plateau schedule, and early stopping.

        Args:
          data: Scaled snapshots, one per entry along the first axis. The last
            `val_fraction` are held out for validation.
          init: Draws initial parameters from a key.
          step: Runs one optimizer step on the batch at a row of the batch index
            array.
          loss: Computes the loss over a block of snapshots.

        Returns:
          The parameters with the lowest validation loss, or the final
          parameters without a validation block, and the training history.
        """
        n_t = data.shape[0]
        n_val = int(round(self.val_fraction * n_t))
        X_tr, X_val = data[: n_t - n_val], data[n_t - n_val :]

        lr = self.learning_rate
        key = jax.random.key(self.seed)
        key, init_key = jax.random.split(key)
        params = init(init_key)
        opt_state = adam_init(params)

        best_params, best_val, wait = params, np.inf, 0
        sched_best, sched_bad = np.inf, 0
        history = TrainingHistory()

        for _ in range(self.n_epochs):
            key, shuffle_key = jax.random.split(key)
            batches = epoch_batches(shuffle_key, X_tr.shape[0], self.batch_size)
            total = jnp.zeros((), dtype=data.dtype)
            for i in range(batches.shape[0]):
                params, opt_state, batch_loss = step(params, opt_state, X_tr, batches, i, lr)
                total = total + batch_loss
            history.train.append(float(total) / batches.shape[0])
            history.lr.append(lr)
            if n_val == 0:
                continue

            v = float(loss(params, X_val))
            history.val.append(v)
            if v < sched_best * (1.0 - self.threshold):
                sched_best, sched_bad = v, 0
            else:
                sched_bad += 1
                if sched_bad > self.lr_patience:
                    lr = max(lr * self.lr_factor, self.min_lr)
                    sched_bad = 0
            if v < best_val * (1.0 - self.threshold):
                best_val, wait, best_params = v, 0, params
            else:
                wait += 1
                if wait >= self.patience:
                    break

        return (best_params if n_val > 0 else params), history

    # ── Saving and restoring ──────────────────────────────────────────────────

    def _weight_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects the trained parameters.

        Returns:
          The parameter leaves, in `jax.tree.leaves` order, named `weights.<i>`.

        Raises:
          RuntimeError: If `fit` has not completed.
        """
        leaves: list[jax.Array] = jax.tree.leaves(self._trained_params())
        return {f"weights.{i}": np.asarray(a) for i, a in enumerate(leaves)}

    def _load_weight_arrays(self, arrays: Mapping[str, npt.NDArray[Any]]) -> None:
        """Builds the architecture and loads the trained parameters.

        Args:
          arrays: Arrays holding the entries `_weight_arrays` returned.
        """
        self._configure()
        skeleton = self._init_params(jax.random.key(0))
        n_leaves = len(jax.tree.leaves(skeleton))
        stored = iter([jnp.asarray(arrays[f"weights.{i}"]) for i in range(n_leaves)])

        def restore(_: jax.Array) -> jax.Array:
            """Returns the next stored parameter array."""
            return next(stored)

        self.weights = jax.tree.map(restore, skeleton)
