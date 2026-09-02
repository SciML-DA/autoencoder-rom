"""The interface a temporal forecaster must satisfy to drive a latent ROM.

NOTE ON PROVENANCE
------------------
This protocol is **ours**, not upstream. There is no equivalent in
``dynamodels``, ``echostatenetwork`` or ``romda``; it is a proposal, and is on
the agenda to discuss with A. Nóvoa rather than something her structure already
prescribes.

What it names, though, is not speculative. `EchoStateNetwork` (the
``echostatenetwork`` package) and `LSTM` (``lstm_core.py``, here) were
written independently and converge on a ten-member public surface -- the members below,
obtained by intersecting ``dir()`` on the two classes, not by assertion.
Writing that overlap down is what lets `LatentROMMixin` be projector- *and*
forecaster-agnostic, so the projector x forecaster matrix costs new leaf classes
rather than a redesign.

Deliberately absent: free-running. The two disagree there -- `EchoStateNetwork`
offers ``run_test``, `LSTM` offers ``openLoop``/``closedLoop`` -- and closing the
loop is the *model* wrapper's job (`ESN_model.time_step`), not the reservoir's.
Requiring it here would be inventing agreement that does not exist.

It is a ``typing.Protocol``: structural, so nothing inherits from it and no
runtime behaviour depends on it. Deleting this file is a one-import change.

NOT TO BE CONFUSED WITH ``LatentForecaster``
--------------------------------------------
``tools/branched_ae.py`` defines a concrete class called `LatentForecaster` -- a
GRU over latent trajectories used by the sparse-sensor nowcast. It does **not**
implement this protocol and is unrelated to it; the names are close enough to
mislead, so they are kept apart deliberately.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["Forecaster"]


@runtime_checkable
class Forecaster(Protocol):
    """A trainable temporal model over a latent trajectory.

    Satisfied as-is by `echostatenetwork.EchoStateNetwork` and by
    `lstm_core.LSTM`.
    """

    #: whether `train` has been called and the readout is usable
    trained: bool
    #: open-loop washout length, in steps
    N_wash: int
    #: per-channel normalisation applied to inputs before stepping
    norm: np.ndarray
    #: per-channel offset applied to inputs before stepping
    shift: np.ndarray
    #: which normalisation `set_norm` applies ('range', 'std', ...)
    norm_method: str

    def train(self, train_data: np.ndarray, **kwargs: Any) -> Any:
        """Fit the forecaster on ``(L, Nt, N_dim)`` trajectories."""
        ...

    def step(self, u: np.ndarray, r: np.ndarray) -> Any:
        """Advance one step from input ``u`` and internal state ``r``."""
        ...

    def normalize_input(self, data: np.ndarray) -> np.ndarray:
        """Apply ``norm``/``shift`` to raw physical inputs."""
        ...

    def compute_nRMSE(self, Y_true: np.ndarray, Y_pred: np.ndarray,
                      norm: float = 1.0) -> Any:
        """Normalised RMSE between truth and prediction."""
        ...

    def copy(self) -> "Forecaster":
        """A deep copy of the forecaster."""
        ...
