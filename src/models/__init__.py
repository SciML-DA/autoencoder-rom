"""The model layer.

The modelling core -- ``Model``, ``HistoryTracker``, the integrators and the
physical models -- lives in the standalone ``dynamodels`` package and is
re-exported here; this subpackage keeps the data-driven models built on top
of it (``ESN_model``, ``POD_ESN``).

``dynamodels`` was split out of A. Nóvoa's real-time bias-aware DA repository
so the modelling layer could be reused on its own. It used to be vendored here
as ``models/{model,history,integrator}.py``; those copies had drifted from
upstream and were removed in favour of the released package.

No ``sys.modules`` aliases are set for the old module paths: nothing in this
repository pickles model instances. ``config/model_config.py`` saves plain arrays
via ``np.savez_compressed`` and records class paths under the current module
names only.
"""

import dynamodels
from dynamodels import (
    ConstantIntegrator,
    DiscreteIntegrator,
    HistoryTracker,
    Integrator,
    IVPIntegrator,
    Model,
    physical,
)

# bind as package attributes too, so `models.model.Model` works and not only
# `from models import Model`
model = dynamodels.model
history = dynamodels.history
integrator = dynamodels.integrator

from . import data_driven  # noqa: E402  (needs the re-exports in place)

__all__ = [
    "Model",
    "HistoryTracker",
    "Integrator",
    "IVPIntegrator",
    "DiscreteIntegrator",
    "ConstantIntegrator",
    "physical",
    "data_driven",
]
