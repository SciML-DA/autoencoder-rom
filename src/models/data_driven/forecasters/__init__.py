"""Temporal forecasters: the other half of a latent ROM.

    `ESN_model`     echo state network as a Model            [esn.py]
    `LSTM`          single-layer LSTM, truncated BPTT       [lstm.py]
    `LSTMJax`       that LSTM trained with JAX              [lstm_jax.py]
    `LSTM_model`    that LSTM as a Model                    [lstm.py]

`LSTMJax` imports JAX, so it loads on first access.
"""

from typing import Any

from .esn import ESN_model, phi_to_esn_layout
from .lstm import LSTM, LSTM_model

__all__ = ["ESN_model", "phi_to_esn_layout", "LSTM", "LSTMJax", "LSTM_model"]


def __getattr__(name: str) -> Any:
    """Loads `LSTMJax` the first time it is accessed.

    Args:
      name: Attribute name.

    Returns:
      The `LSTMJax` class.

    Raises:
      AttributeError: If the package exports no attribute called `name`.
    """
    if name == "LSTMJax":
        from .lstm_jax import LSTMJax

        return LSTMJax
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
