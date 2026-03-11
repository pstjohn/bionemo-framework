#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install DeepEP (hybrid-ep branch) with NVSHMEM support.
#
# Environment expectations:
#   - CUDA toolkit (nvcc) available via $CUDA_HOME or /usr/local/cuda
#   - PyTorch pre-installed (NGC container or equivalent)
#   - GPU with Blackwell architecture (sm_100 / sm_120) recommended
#
# Usage:
#   bash install_hybridep.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE_DIR="$(mktemp -d)"
trap 'rm -rf "$CLONE_DIR"' EXIT

echo "============================================"
echo " DeepEP (hybrid-ep) + NVSHMEM Installation"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Install NVSHMEM via pip (headers + libs + static archives)
# ---------------------------------------------------------------------------
echo "[1/5] Installing NVSHMEM ..."

# Pick the pip package that matches the CUDA major version so we don't pull
# in a cu12 package on a CUDA-13 system (or vice-versa).
CUDA_MAJOR="$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])")"
echo "       Detected CUDA major version: ${CUDA_MAJOR}"

pip install --no-cache-dir "nvidia-nvshmem-cu${CUDA_MAJOR}" 2>&1 | tail -3
echo ""

# Verify the pip package landed where we expect.
NVSHMEM_DIR="$(python3 -c "
import importlib.util, sys
spec = importlib.util.find_spec('nvidia.nvshmem')
if spec is None:
    print('ERROR: nvidia.nvshmem not found after pip install', file=sys.stderr)
    sys.exit(1)
print(spec.submodule_search_locations[0])
")"
echo "       NVSHMEM_DIR=${NVSHMEM_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 2. Install pynvml (DeepEP runtime dependency)
# ---------------------------------------------------------------------------
echo "[2/5] Installing pynvml ..."
pip install --no-cache-dir pynvml 2>&1 | tail -2
echo ""

# ---------------------------------------------------------------------------
# 3. Detect target GPU architecture
# ---------------------------------------------------------------------------
echo "[3/5] Detecting GPU architecture ..."

GPU_ARCH="$(python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('10.0')                       # safe default (Blackwell datacenter)
    sys.exit(0)
cap = torch.cuda.get_device_capability(0)
arch = f'{cap[0]}.{cap[1]}'
# DeepEP hybrid-ep kernels target Blackwell (10.0+).  If the detected arch
# is older than 10.0 we still try 10.0 and rely on PTX forward-compat.
major, minor = cap
if major < 10:
    arch = '10.0'
print(arch)
")"
echo "       Target TORCH_CUDA_ARCH_LIST=${GPU_ARCH}"
echo ""

# ---------------------------------------------------------------------------
# 4. Clone & build DeepEP (hybrid-ep branch)
# ---------------------------------------------------------------------------
echo "[4/5] Cloning and building DeepEP ..."
git clone --branch hybrid-ep --depth 1 \
    https://github.com/deepseek-ai/DeepEP.git \
    "${CLONE_DIR}/DeepEP" 2>&1 | tail -3

pushd "${CLONE_DIR}/DeepEP" > /dev/null

# Export build knobs.
# - NVSHMEM_DIR: auto-discovered by setup.py from the pip package, but we set
#   it explicitly for clarity and to avoid the importlib fallback during build.
# - TORCH_CUDA_ARCH_LIST: compile for the detected (or default) arch.
# - DISABLE_AGGRESSIVE_PTX_INSTRS: automatically set to 1 by setup.py for
#   any arch != 9.0, so no need to export it here.
export NVSHMEM_DIR
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"

echo "       Building wheel (this may take several minutes) ..."
python3 setup.py bdist_wheel 2>&1 | tail -5
echo ""

echo "       Installing wheel ..."
pip install --no-cache-dir --no-deps dist/*.whl 2>&1 | tail -3

popd > /dev/null
echo ""

# ---------------------------------------------------------------------------
# 5. Verify imports
# ---------------------------------------------------------------------------
echo "[5/5] Verifying imports ..."
python3 -c "
import sys

failures = []

# -- deep_ep (Python package) --
try:
    import deep_ep
    print('  OK  deep_ep')
except Exception as e:
    print(f'  FAIL deep_ep: {e}')
    failures.append('deep_ep')

# -- deep_ep_cpp (C++/CUDA extension) --
try:
    import deep_ep_cpp
    print('  OK  deep_ep_cpp')
except Exception as e:
    print(f'  FAIL deep_ep_cpp: {e}')
    failures.append('deep_ep_cpp')

# -- hybrid_ep_cpp (C++/CUDA extension) --
try:
    import hybrid_ep_cpp
    print('  OK  hybrid_ep_cpp')
except Exception as e:
    print(f'  FAIL hybrid_ep_cpp: {e}')
    failures.append('hybrid_ep_cpp')

# -- nvidia.nvshmem --
try:
    import nvidia.nvshmem
    print('  OK  nvidia.nvshmem')
except Exception as e:
    print(f'  FAIL nvidia.nvshmem: {e}')
    failures.append('nvidia.nvshmem')

if failures:
    print(f'\nERROR: {len(failures)} import(s) failed: {failures}', file=sys.stderr)
    sys.exit(1)
else:
    print('\nAll imports succeeded.')
"

echo ""
echo "============================================"
echo " Installation complete"
echo "============================================"
