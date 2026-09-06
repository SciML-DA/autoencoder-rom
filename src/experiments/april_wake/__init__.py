"""The April porous-disc campaign: PIV snapshots paired with load-cell forces.

`case_reader` reads the RDS directory layout into `Case` objects;
`data_preprocessing` turns those into the arrays the estimators consume --
argument plumbing, force re-timing, band limiting and splitting.
"""

from . import case_reader, data_preprocessing

__all__ = ["case_reader", "data_preprocessing"]
