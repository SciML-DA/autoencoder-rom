"""Experiment-specific code: one subpackage per measurement campaign.

Everything here knows about a particular rig -- its sampling rates, channel
count, run names and file layout. Nothing outside this package may import from
it, which is what keeps `datasets`, `models`, `field_estimation` and `plotting`
reusable across experiments.
"""
