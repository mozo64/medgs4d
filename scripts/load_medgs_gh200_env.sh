#!/usr/bin/env bash

# This file must be sourced so that module and environment changes
# remain active in the current shell:
#
#   source scripts/load_medgs_gh200_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: this script must be sourced, not executed."
    echo "Use: source ${BASH_SOURCE[0]}"
    exit 1
fi

_medgs4d_fail() {
    echo "MedGS4D environment error: $*" >&2
    return 1
}

if [[ "$(uname -m)" != "aarch64" ]]; then
    _medgs4d_fail         "GH200 environment requires an aarch64 compute node. Current architecture: $(uname -m). Start a Slurm GH200 job first."
    return 1
fi

if ! command -v module >/dev/null 2>&1; then
    _medgs4d_fail "The environment-modules command is unavailable."
    return 1
fi

if [[ -z "${PLG_GROUPS_STORAGE:-}" ]]; then
    _medgs4d_fail "PLG_GROUPS_STORAGE is not defined."
    return 1
fi

if [[ -z "${SCRATCH:-}" ]]; then
    _medgs4d_fail "SCRATCH is not defined."
    return 1
fi

export MEDGS4D_GROUP="${MEDGS4D_GROUP:-plggtriplane}"
export MEDGS4D_ROOT="${MEDGS4D_ROOT:-${PLG_GROUPS_STORAGE}/${MEDGS4D_GROUP}/${USER}/medgs4d}"
export MEDGS4D_VENV="${MEDGS4D_VENV:-${MEDGS4D_ROOT}/envs/medgs-gh200}"

if [[ ! -f "${MEDGS4D_VENV}/bin/activate" ]]; then
    _medgs4d_fail "Virtual environment not found: ${MEDGS4D_VENV}"
    return 1
fi

# Start from a reproducible module set for the GH200 ARM compute nodes.
module purge
module load Python/3.11.5
module load CUDA/12.8.0

# Activate the persistent ARM64 Python environment.
# shellcheck disable=SC1091
source "${MEDGS4D_VENV}/bin/activate"

# Keep large caches and temporary build files outside the small home directory.
export PIP_CACHE_DIR="${MEDGS4D_ROOT}/cache/pip"
export TMPDIR="${SCRATCH}/medgs4d/tmp"
export PYTHONNOUSERSITE=1

# Build native PyTorch extensions specifically for NVIDIA Hopper / GH200.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

if ! command -v python >/dev/null 2>&1; then
    _medgs4d_fail "Python is unavailable after activating the environment."
    return 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    _medgs4d_fail "nvcc is unavailable after loading CUDA/12.8.0."
    return 1
fi

echo "MedGS4D GH200 environment loaded"
echo "  Slurm job:      ${SLURM_JOB_ID:-not-set}"
echo "  Node:           $(hostname)"
echo "  Architecture:   $(uname -m)"
echo "  Project root:   ${MEDGS4D_ROOT}"
echo "  Virtual env:    ${VIRTUAL_ENV}"
echo "  Python:         $(python --version 2>&1)"
echo "  CUDA toolkit:   $(nvcc --version | awk '/release/ {print $5}' | tr -d ',')"
echo "  CUDA_HOME:      ${CUDA_HOME:-not-set}"
echo "  GPU arch list:  ${TORCH_CUDA_ARCH_LIST}"
echo "  Build workers:  ${MAX_JOBS}"

unset -f _medgs4d_fail
