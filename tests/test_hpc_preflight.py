"""
Preflight for the PBS job scripts.

Extracts the command each job actually submits and runs it against a synthetic
fixture. Catches the failures that only appear on the cluster: argument drift
between a script and its caller, flags that exist in one entry point but not
another, unbound variables under `set -u`, and paths that resolve differently
inside a job.

Every one of those has reached the queue at least once. Running the code locally
with hand-written arguments does not catch them, because the arguments in the
job script are the ones that break.

    pytest tests/test_hpc_preflight.py -q
"""

from __future__ import annotations

import glob
import os
import re
import shlex
import subprocess
import sys
import tempfile

import pytest

from test_sparse_sensors import _write_fixture  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Jobs live in two places: hpc/ for the ones not tied to a rig, and
# experiments/<campaign>/hpc/ for the ones that are. Paths here are
# repo-relative so a job can move between them without editing this list twice.
AW = "experiments/april_wake/hpc"

# Scripts that load experiment data. convergence.pbs uses a different dataset and
# sparse_sensors_crossyaw.pbs needs several runs, so both are covered elsewhere.
JOBS = [f"{AW}/{j}" for j in
        ["lowrank.pbs", "sensors.pbs", "diagnose.pbs", "sparse_sensors.pbs",
         "sparse_sensor_sweep.pbs", "spectra.pbs",
         "convergence_diagnosis.pbs", "loss_curves.pbs", "hyper_search.pbs"]]

# Overrides that make a cluster-sized job finish in seconds. Applied last, so
# they win over whatever the job script sets.
SMALL = {
    "sparse_sensor_sweep.py": ["--quick", "--stages", "A", "--latents-sweep", "4",
                               "--latents", "pod", "--branches", "linear",
                               "--no-video", "--device", "cpu"],
    "sparse_sensor_study.py": ["--n", "300", "--r-field", "4", "--delays", "1",
                               "--latents", "pod", "--branches", "--no-plots",
                               "--device", "cpu"],
    "diagnose_sensors.py": ["--n", "300", "--r-field", "4", "--n-delays", "3",
                            "--skip", "spectrum", "leak", "lag"],
    "convergence_diagnosis.py": ["--quick", "--n", "300", "--device", "cpu",
                                "--branches", "linear", "--latents", "pod"],
    "loss_curves.py": ["--n", "300", "--r-field", "4", "--n-delays", "5",
                       "--epochs", "10", "--patience", "5", "--ae-epochs", "10",
                       "--branches", "linear", "--latents", "pod",
                       "--device", "cpu"],
    # needs a record long enough to survive make_split's warm-up plus
    # hyper_search's own validation guard band
    "hyper_search.py": ["--n", "900", "--delays", "25", "--stages", "screen",
                        "--limit", "2", "--device", "cpu"],
    "spectra.py": ["--n", "300", "--r-field", "4", "--n-modes-coh", "2",
                   "--nperseg", "64", "--nperseg-force", "512",
                   "--max-lag", "20"],
    "inspect_experiment.py": [],
}


def extract_command(job: str) -> list[str]:
    """
    Returns the python argv the job script builds, by shadowing `uv`.

    Shadows a shell function rather than a PATH entry: the job scripts prepend
    $HOME/.local/bin to PATH themselves, which shadows a stub binary right back.
    """
    probe = (
        'uv() { printf "PYARGS:%s\\n" "${*:3}"; }\n'
        f'cd {shlex.quote(REPO)}\n'
        f'PBS_O_WORKDIR={shlex.quote(REPO)} PBS_JOBID=preflight.0 '
        'PBS_JOBNAME=preflight NCPUS=2\n'
        f'source {shlex.quote(os.path.join(REPO, job))}\n'
    )
    p = subprocess.run(["bash", "-c", probe], capture_output=True, text=True,
                       cwd=REPO, timeout=300)
    lines = [ln[len("PYARGS:"):] for ln in p.stdout.splitlines()
             if ln.startswith("PYARGS:")]
    assert lines, (
        f"{job} produced no command (exit {p.returncode}).\n"
        f"stderr: {p.stderr[-800:]}"
    )
    return shlex.split(lines[-1])


@pytest.mark.parametrize("job", JOBS)
def test_job_script_command_runs(job, tmp_path):
    argv = extract_command(job)
    script = os.path.basename(argv[0])
    if script not in SMALL:
        pytest.skip(f"no small-run profile for {script}")

    root = tempfile.mkdtemp()
    run, _ = _write_fixture(root, n_t=300)

    # drop the job's own --out/--tag/--run/--cache-dir, then apply the small profile
    #
    # Resolved from the job's own argv rather than by assuming a directory: the
    # rig-specific entry points live under experiments/<campaign>/scripts/
    # and the generic ones under scripts/, and this should not need editing
    # again the next time one moves.
    path = os.path.join(REPO, argv[0])
    argv = _strip(argv, {"--out", "--tag", "--run", "--cache-dir"})
    extra = ["--run", run, "--out", str(tmp_path)]
    if "--tag" in _help(path):
        extra += ["--tag", "pf"]
    cmd = [sys.executable, path] + argv[1:] + SMALL[script] + extra

    env = {**os.environ, "RDS_ROOT": root, "MPLBACKEND": "Agg",
           "OMP_NUM_THREADS": "2"}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    assert p.returncode == 0, (
        f"{job} -> {script} failed (exit {p.returncode})\n"
        f"cmd: {' '.join(cmd[1:])}\n"
        f"stdout tail:\n{p.stdout[-1500:]}\n"
        f"stderr tail:\n{p.stderr[-1500:]}"
    )


def _help(script_path: str) -> str:
    """Returns a script's --help text, for checking a flag exists before using it."""
    return subprocess.run([sys.executable, script_path, "--help"],
                          capture_output=True, text=True, timeout=120).stdout


def _strip(argv: list[str], flags: set[str]) -> list[str]:
    """Removes each flag in `flags` and its value from an argv list."""
    out, skip = [], False
    for a in argv:
        if skip and not a.startswith("--"):
            continue
        skip = False
        if a in flags:
            skip = True
            continue
        out.append(a)
    return out


ALL_JOBS = sorted(
    os.path.relpath(f, REPO)
    for pattern in ("hpc/*.pbs", "experiments/*/hpc/*.pbs")
    for f in glob.glob(os.path.join(REPO, pattern))
)


@pytest.mark.parametrize("job", ALL_JOBS)
def test_job_script_uses_the_shared_setup(job):
    """Every job sources hpc/lib.sh rather than carrying its own copy.

    Five job scripts once held five copies of the thread block, and they drifted:
    one of them had lost the `unset` that clears a stale OMP_NUM_THREADS
    inherited through `qsub -V`, so that job ran single-threaded on eight cores
    and said `threads: 1 (NCPUS=1, nproc=1)` in a log nobody reread.
    """
    src = open(os.path.join(REPO, job)).read()
    assert "hpc/lib.sh" in src, f"{job} does not source hpc/lib.sh"
    assert "unset OMP_NUM_THREADS" not in src, (
        f"{job} has its own thread block again -- it belongs in hpc/lib.sh"
    )


def _threads(**env) -> int:
    """OMP_NUM_THREADS as hpc_threads actually resolves it, in a clean shell."""
    probe = (f"set -euo pipefail\nsource {shlex.quote(os.path.join(REPO, 'hpc', 'lib.sh'))}\n"
             "hpc_threads >/dev/null\necho \"$OMP_NUM_THREADS\"\n")
    base = {k: v for k, v in os.environ.items()
            if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NCPUS", "HPC_MAX_THREADS",
                         "HPC_TEST_MASK", "HPC_TEST_MACHINE")}
    p = subprocess.run(["bash", "-c", probe], capture_output=True, text=True,
                       env={**base, **{k: str(v) for k, v in env.items()}})
    assert p.returncode == 0, p.stderr
    return int(p.stdout.strip())


def test_stale_omp_num_threads_is_overridden():
    """The failure that actually happened: OMP_NUM_THREADS=1 through `qsub -V`.

    `nproc` reports min(affinity, $OMP_NUM_THREADS), so an inherited 1 makes
    every source agree on 1 and the job silently runs serial. hpc_threads must
    clear the variable before it probes anything.
    """
    assert _threads(OMP_NUM_THREADS=1, NCPUS=8) == 8


def test_cgroup_wins_when_the_node_is_cpuset_confined():
    """The cx3 case, from job 3936942 on cx3-5-17.

    PBS reported NCPUS=1 for a job whose select line asked for 8, while the
    cgroup handed the job an 8-CPU cpuset on a 128-core node. Believing NCPUS
    there ran every SVD single-threaded on eight allocated cores, and the whole
    verify job took 16 minutes at 36% CPU.

    A mask narrower than the machine means a cgroup is enforcing a real
    allocation, and it is then the authority: it is what the kernel will
    schedule on regardless of what PBS reports.
    """
    assert _threads(NCPUS=1, HPC_TEST_MASK=8, HPC_TEST_MACHINE=128) == 8


def test_allocation_wins_when_there_is_no_cpuset():
    """PBS Pro only cpuset-confines a job when the cgroup hook is enabled.

    Where it is not, the affinity mask is the whole node and `nproc` returns the
    machine size for a job that asked for two cores. Taking the mask there
    oversubscribes a shared node by an order of magnitude, so the allocation has
    to win.
    """
    assert _threads(NCPUS=2, HPC_TEST_MASK=128, HPC_TEST_MACHINE=128) == 2


def test_never_exceeds_the_cpuset():
    """Whatever PBS claims, the mask is a hard upper bound."""
    assert _threads(NCPUS=64, HPC_TEST_MASK=8, HPC_TEST_MACHINE=128) == 8


def test_falls_back_and_caps():
    assert _threads() >= 1                                  # no PBS at all
    assert _threads(NCPUS=16, HPC_MAX_THREADS=3) == 3       # explicit override


@pytest.mark.parametrize("job", ALL_JOBS)
def test_job_script_is_valid_bash(job):
    p = subprocess.run(["bash", "-n", os.path.join(REPO, job)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
