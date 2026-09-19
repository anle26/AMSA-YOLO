#!/usr/bin/env bash
# ==============================================================================
# AMSA-YOLO: Offline Kaggle Bootstrap & Verification Script
# ==============================================================================
# Targets: Kaggle Linux x86_64 container with Internet OFF
# Primary GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
# ==============================================================================

set -euo pipefail

echo "================================================================================"
echo " Starting AMSA-YOLO Offline Kaggle Bootstrap"
echo "================================================================================"

# 1. Require uv strictly (do not fall back to pip)
if ! command -v uv >/dev/null 2>&1; then
    echo "[!] CRITICAL ERROR: 'uv' is not installed or not in PATH."
    echo "    This bootstrap script requires uv for deterministic offline dependency management."
    exit 1
fi
echo "[+] Detected uv executable: $(command -v uv) (version: $(uv --version))"

# 2. Locate the mounted wheelhouse directory
if [ -z "${WHEELHOUSE_DIR:-}" ]; then
    echo "[*] WHEELHOUSE_DIR not set. Searching /kaggle/input and current directory..."
    DETECTED_WH=$(find /kaggle/input . -maxdepth 3 -type d -name "wheelhouse" 2>/dev/null | head -n 1 || true)
    if [ -n "${DETECTED_WH}" ] && [ -d "${DETECTED_WH}" ]; then
        WHEELHOUSE_DIR="${DETECTED_WH}"
        echo "[+] Discovered wheelhouse at: ${WHEELHOUSE_DIR}"
    else
        echo "[!] ERROR: Could not locate 'wheelhouse' directory."
        echo "    Please export WHEELHOUSE_DIR=/path/to/wheelhouse and re-run."
        exit 1
    fi
else
    echo "[+] Using specified WHEELHOUSE_DIR: ${WHEELHOUSE_DIR}"
fi

# Verify wheel files exist in wheelhouse
WHL_COUNT=$(find "${WHEELHOUSE_DIR}" -maxdepth 1 -name "*.whl" | wc -l)
if [ "${WHL_COUNT}" -eq 0 ]; then
    echo "[!] ERROR: No .whl binary wheels found in ${WHEELHOUSE_DIR}"
    exit 1
fi
echo "[+] Found ${WHL_COUNT} binary wheel(s) in wheelhouse."

# 3. FORBIDDEN PACKAGE GUARD (Part B)
# Scan wheelhouse files for forbidden base-stack packages
echo "[*] Running safety guard on wheelhouse filenames..."
FORBIDDEN_WHLS=$(find "${WHEELHOUSE_DIR}" -maxdepth 1 -type f \( -name "torch-*" -o -name "torchvision-*" -o -name "nvidia_*" -o -name "triton-*" \) 2>/dev/null || true)
if [ -n "${FORBIDDEN_WHLS}" ]; then
    echo "[!] CRITICAL SAFETY GUARD FAILURE: Forbidden base-stack wheels detected in wheelhouse:"
    echo "${FORBIDDEN_WHLS}"
    echo "[!] Base packages (torch, torchvision, triton, nvidia-*) must NOT be bundled in the wheelhouse."
    exit 1
fi
echo "[+] Wheelhouse file scan passed: Zero forbidden base-stack wheels detected."

# Locate requirements file
if [ -f "${WHEELHOUSE_DIR}/../requirements/wheelhouse.txt" ]; then
    REQ_FILE="${WHEELHOUSE_DIR}/../requirements/wheelhouse.txt"
elif [ -f "./requirements/wheelhouse.txt" ]; then
    REQ_FILE="./requirements/wheelhouse.txt"
elif [ -f "${WHEELHOUSE_DIR}/wheelhouse.txt" ]; then
    REQ_FILE="${WHEELHOUSE_DIR}/wheelhouse.txt"
else
    REQ_FILE=""
fi

# Scan requirements file if found
if [ -n "${REQ_FILE}" ] && [ -f "${REQ_FILE}" ]; then
    echo "[*] Scanning requirements specification: ${REQ_FILE}..."
    FORBIDDEN_REQS=$(grep -E '^(torch|torchvision|triton|nvidia[-_])' "${REQ_FILE}" || true)
    if [ -n "${FORBIDDEN_REQS}" ]; then
        echo "[!] CRITICAL SAFETY GUARD FAILURE: Forbidden packages specified in ${REQ_FILE}:"
        echo "${FORBIDDEN_REQS}"
        echo "[!] Must not request reinstallation of base stack."
        exit 1
    fi
    echo "[+] Requirements specification scan passed: No forbidden entries in ${REQ_FILE}."
fi

# 4. Pre-installation Base Stack Verification (Part A)
echo "[*] Checking preinstalled Kaggle base PyTorch stack via /usr/bin/python3..."
/usr/bin/python3 -c "
import sys

try:
    import torch
    print(f'[+] Detected base torch      : {torch.__version__} (CUDA: {torch.version.cuda})')
    # Verify compatibility with 2.10.x baseline
    if not torch.__version__.startswith('2.10'):
        print(f'[!] WARNING: Detected torch version {torch.__version__} differs from expected 2.10.x baseline.')
except ImportError as e:
    print(f'[!] CRITICAL ERROR: Base torch is not installed: {e}')
    sys.exit(1)

try:
    import torchvision
    print(f'[+] Detected base torchvision: {torchvision.__version__}')
    # Verify compatibility with 0.25.x baseline
    if not torchvision.__version__.startswith('0.25'):
        print(f'[!] WARNING: Detected torchvision version {torchvision.__version__} differs from expected 0.25.x baseline.')
except ImportError as e:
    print(f'[!] CRITICAL ERROR: Base torchvision is not installed: {e}')
    sys.exit(1)
"

# 5. Create /opt/venv with access to Kaggle base site-packages using uv
VENV_PATH="/opt/venv"
VENV_PYTHON="${VENV_PATH}/bin/python"

echo "[*] Initializing isolated virtual environment at ${VENV_PATH}..."
uv venv "${VENV_PATH}" \
  --python /usr/bin/python3 \
  --system-site-packages \
  --no-managed-python

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "[!] ERROR: Virtual environment Python not executable at ${VENV_PYTHON}"
    exit 1
fi
echo "[+] Isolated environment ready: $(${VENV_PYTHON} --version)"

# 6. Install wheelhouse dependencies using uv pip install (never pip)
# NOTE ON --no-deps INTENTIONALITY:
# - Kaggle's verified base image already provides PyTorch (2.10.0+cu128) and torchvision (0.25.0+cu128).
# - /opt/venv was created with --system-site-packages to inherit this vendor-optimized base stack.
# - requirements/wheelhouse.txt represents an explicitly pinned, complete non-base dependency layer.
# - The offline wheelhouse intentionally excludes torch, torchvision, and CUDA runtime wheels.
# - Allowing uv to resolve dependencies with --no-index would fail because ultralytics declares
#   torch>=1.8.0 as a dependency, which uv's offline resolver would attempt to locate within the wheelhouse.
# - Using --no-deps bypasses index-based dependency resolution while installing the exact pinned wheels.
# - Complete dependency consistency is verified immediately afterward via `uv pip check`.
echo "[*] Installing offline wheelhouse dependencies via uv pip install (--no-deps)..."
if [ -n "${REQ_FILE}" ] && [ -f "${REQ_FILE}" ]; then
    uv pip install \
      --python "${VENV_PYTHON}" \
      --offline \
      --no-index \
      --no-deps \
      --find-links "${WHEELHOUSE_DIR}" \
      -r "${REQ_FILE}"
else
    uv pip install \
      --python "${VENV_PYTHON}" \
      --offline \
      --no-index \
      --no-deps \
      --find-links "${WHEELHOUSE_DIR}" \
      $(find "${WHEELHOUSE_DIR}" -maxdepth 1 -name "*.whl")
fi
echo "[+] Offline installation complete."

# 7. Dependency Consistency Verification
# Verify that all installed packages and inherited base packages satisfy all dependency requirements
echo "[*] Verifying dependency consistency across virtual environment and base stack via uv pip check..."
uv pip check --python "${VENV_PYTHON}"
echo "[+] Dependency consistency verified: All requirements satisfied."

# 8. Comprehensive Runtime Verification (Parts C & D)
echo ""
echo "================================================================================"
echo " Executing Post-Bootstrap Runtime Verification"
echo "================================================================================"

"${VENV_PYTHON}" -c "
import sys
import os

print(f'Python Executable    : {sys.executable}')
print(f'Python Version       : {sys.version.split()[0]}')

# Check PyTorch stack
import torch
import torchvision
print(f'torch.__version__    : {torch.__version__}')
print(f'torch.version.cuda   : {torch.version.cuda}')
print(f'torchvision.__ver__  : {torchvision.__version__}')
print(f'CUDA Available       : {torch.cuda.is_available()}')

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / (1024 ** 3)
    print(f'GPU Name             : {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability   : {props.major}.{props.minor}')
    print(f'VRAM Total           : {vram_gib:.2f} GiB')
else:
    print('[!] ERROR: CUDA is not available in PyTorch.')
    sys.exit(1)

# Import non-torch packages
packages = ['numpy', 'scipy', 'pandas', 'cv2', 'PIL', 'yaml', 'tqdm', 'psutil']
for p in packages:
    try:
        mod = __import__(p)
        ver = getattr(mod, '__version__', 'OK')
        print(f'{p:<20} : OK (v{ver})')
    except Exception as e:
        print(f'{p:<20} : FAILED ({e})')
        sys.exit(1)

# Ultralytics verification
import ultralytics
print(f'ultralytics.__ver__  : {ultralytics.__version__}')

# Part D: Verify Ultralytics without weight downloads
try:
    from ultralytics import YOLO
    print('[+] from ultralytics import YOLO: SUCCESS')
    # Instantiate from architecture YAML only (randomly initialized, no .pt download)
    model = YOLO('yolov8s.yaml')
    print(f'[+] YOLO(\"yolov8s.yaml\") instantiated successfully: {type(model).__name__}')
except Exception as e:
    print(f'[!] CRITICAL ERROR in YOLO architecture instantiation: {e}')
    sys.exit(1)

# Part C: Minimal CUDA Tensor Matrix Multiplication Test
print('[*] Running minimal CUDA tensor computation...')
a = torch.randn(128, 128, device='cuda', dtype=torch.float32)
b = torch.randn(128, 128, device='cuda', dtype=torch.float32)
c = torch.matmul(a, b)
torch.cuda.synchronize()
assert c.shape == (128, 128)
print('[+] CUDA Tensor Matmul Test: SUCCESS')
"

echo "================================================================================"
echo " AMSA-YOLO Bootstrap Successfully Completed at ${VENV_PATH}"
echo "================================================================================"
