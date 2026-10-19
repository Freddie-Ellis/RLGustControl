#!/bin/bash
# One-off setup of the Python environment the CFD jobs use on Iridis 6.
#
# Run on the LOGIN node (it downloads packages; compute nodes may have no internet):
#     bash hpc/setup_python.sh
# Re-run it to rebuild the environment, e.g. after a dependency changes.
#
# Creates <repo>/.venv-hpc with only what src/cfd needs (torch CPU build and pydantic), rather than the full
# project environment, which includes Windows-only and GUI packages. It records which Python module it was
# built with in .venv-hpc/python_module, so the Slurm jobs load the same one.

set -euo pipefail

# Make the `module` command available when this is run as a plain script rather than in a login shell.
if ! command -v module > /dev/null 2>&1; then
    source /etc/profile > /dev/null 2>&1 || true
fi

repo_root="$(git rev-parse --show-toplevel)"
venv="${repo_root}/.venv-hpc"

module purge
# Pick the newest Python 3.14 module (list them with: module avail python).
python_module="$(module -t avail python 2>&1 | grep -E '^python/3\.14' | sort -V | tail -n 1)"
if [ -z "${python_module}" ]; then
    echo "No python/3.14 module found; check 'module avail python' and edit this script." >&2
    exit 1
fi
module load "${python_module}"
echo "Using ${python_module}: $(command -v python3) ($(python3 --version))"

rm -rf "${venv}"
python3 -m venv "${venv}"
source "${venv}/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "pydantic>=2.13"
pip install numpy   # not used by src/cfd, but torch warns on import without it
echo "${python_module}" > "${venv}/python_module"

python -c "import torch, pydantic; print(f'torch {torch.__version__}, pydantic {pydantic.__version__}: OK')"
echo "Environment ready: ${venv}"
