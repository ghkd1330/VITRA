#!/usr/bin/env bash
# Install PyTorch + torchvision + torchaudio built with CUDA 12.8 (cu128).
# Required for NVIDIA Blackwell GPUs (compute capability sm_120), e.g. RTX 5090.
# Run AFTER: conda activate vitra && pip install -e .
#
# This replaces older PyTorch wheels with official cu128 binaries from
# https://pytorch.org/get-started/locally/ (Linux + CUDA 12.8).
# Model weights and numerics are unchanged; only the GPU runtime matches the GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Upgrading PyTorch stack to CUDA 12.8 wheels (cu128)..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo
echo "Verifying installation:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

# --- PyTorch3D: must be built with nvcc + CUDA headers (pip nvidia-* packages). ---
# Without nvcc, the wheel is CPU-only ("Not compiled with GPU support").
# Without CPATH to site-packages/nvidia/*/include, build fails on cusparse.h.

rebuild_pytorch3d_gpu() {
  echo "Rebuilding PyTorch3D with CUDA (several minutes)..."
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "Installing nvcc via conda (required for PyTorch3D GPU kernels)..."
    conda install -y -c nvidia cuda-nvcc=12.8
  fi

  export CUDA_HOME="${CONDA_PREFIX}"
  export CUB_HOME="${CONDA_PREFIX}/targets/x86_64-linux"
  # CUDA 12.8 nvcc rejects GCC 14.x from conda; use system compiler if available.
  export CC="${CC:-/usr/bin/gcc}"
  export CXX="${CXX:-/usr/bin/g++}"
  SITE_PACKAGES="$(python -c "import site; print(site.getsitepackages()[0])")"
  NV_INCLUDES="$(find "$SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name include 2>/dev/null | paste -sd: -)"
  export CPATH="${NV_INCLUDES}${CPATH:+:$CPATH}"
  export FORCE_CUDA=1
  export PATH="${CONDA_PREFIX}/bin:${PATH}"

  pip uninstall -y pytorch3d
  pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
}

echo
if pip show pytorch3d >/dev/null 2>&1; then
  NEED=0
  if ! python -c "import pytorch3d" 2>/dev/null; then
    NEED=1
  else
    set +e
    python "$SCRIPT_DIR/check_pytorch3d_gpu.py"
    RC=$?
    set -e
    if [ "$RC" -eq 1 ]; then
      NEED=1
    fi
  fi
  if [ "$NEED" -eq 1 ]; then
    rebuild_pytorch3d_gpu
  fi
  set +e
  python "$SCRIPT_DIR/check_pytorch3d_gpu.py"
  P3D_RC=$?
  set -e
  if [ "$P3D_RC" -eq 0 ]; then
    echo "PyTorch3D: CUDA rasterization OK."
  elif [ "$P3D_RC" -eq 2 ]; then
    echo "PyTorch3D: CUDA check skipped (no GPU in this session)."
  fi
else
  echo "PyTorch3D not installed. For visualization: pip install -e .[visulization]"
fi

echo
echo "If other CUDA extensions report undefined symbols, reinstall them against this torch build."
