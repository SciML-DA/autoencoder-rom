"""ROM-side figures: POD modes, coefficients, spectra, RMS fields.

The sparse-sensor plots that used to live here moved to
``field_estimation/plots.py``, so that package is self-contained and this one is
only about the reduced-order models.
"""

from . import figures
from . import pod

__all__ = [
    "figures",
    "pod",
]
