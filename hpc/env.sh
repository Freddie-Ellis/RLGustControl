#!/bin/bash
# Environment shared by the CFD scripts on Iridis 6. Source it, then call the functions you need:
#     source hpc/env.sh
#     load_openfoam      # OpenFOAM (v2506) for the Slurm jobs
#     load_python        # the .venv-hpc Python environment made by hpc/setup_python.sh
# Also sets repo_root to the repository's top folder.

OF_VERSION="v2506"
repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

load_openfoam() {
    module purge
    # OpenFOAM's bashrc treats any arguments it sees as extra settings files, and references unset
    # variables. Inside this function it sees no arguments; strict mode is switched off while it loads.
    local saved_flags="$-"
    set +eu
    source "/iridisfs/i6software/openfoam/${OF_VERSION}/OpenFOAM-${OF_VERSION}/etc/bashrc"
    [[ "${saved_flags}" == *e* ]] && set -e
    [[ "${saved_flags}" == *u* ]] && set -u
    if [ -z "${WM_PROJECT_VERSION:-}" ]; then
        echo "OpenFOAM ${OF_VERSION} did not load" >&2
        return 1
    fi
}

load_python() {
    local venv="${repo_root}/.venv-hpc"
    if [ ! -f "${venv}/python_module" ]; then
        echo "No Python environment at ${venv}; run 'bash hpc/setup_python.sh' on the login node first." >&2
        return 1
    fi
    module load "$(cat "${venv}/python_module")"
    source "${venv}/bin/activate"
}

# Run one step, logging to log.<name> in the current folder; print the log's end and stop on failure.
run_step() {
    local name="$1"
    shift
    echo "--- ${name}"
    local start=${SECONDS}
    if ! "$@" > "log.${name}" 2>&1; then
        echo "FAILED: ${name}"
        tail -n 40 "log.${name}"
        exit 1
    fi
    echo "    done in $(( SECONDS - start )) s"
}

print_environment() {
    echo "=== Environment ==="
    echo "Job ID:        ${SLURM_JOB_ID:-none} on ${SLURM_JOB_NODELIST:-$(hostname)} (${SLURM_NTASKS:-1} tasks)"
    echo "OpenFOAM:      ${WM_PROJECT_VERSION:-not loaded}"
    echo "Python:        $(command -v python) ($(python --version 2>&1))"
    echo "Commit:        $(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
}
