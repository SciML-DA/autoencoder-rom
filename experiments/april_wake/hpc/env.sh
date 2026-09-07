#!/usr/bin/env bash
# Environment specific to the April porous-disc campaign.
#
# Sourced by this campaign's job scripts, right after hpc/lib.sh and *before*
# `hpc_init` -- `hpc_report` prints RDS_ROOT in the job banner, so setting it
# afterwards would log `unset` on every run. It is
# separate from hpc/lib.sh on purpose: lib.sh is shared by every campaign and
# must not know where any one of them keeps its data. The default below used to
# live there, which meant the generic library carried one rig's project path.
#
# `${VAR:-default}` throughout, so an explicitly set value always wins -- the
# test suite points RDS_ROOT at a synthetic fixture, and a laptop copy of the
# data needs nothing but an export.

# Read in place from RDS project space, not copied. Mirrors the default in
# experiments/april_wake/case_reader.py, which is what resolves it in python.
export RDS_ROOT="${RDS_ROOT:-/rds/general/project/immanuel/live/Seagate/april_experiment}"
