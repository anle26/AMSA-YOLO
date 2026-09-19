# Environment Notes & Constraints: AMSA-YOLO Reproducibility

These technical notes document non-negotiable architectural boundaries and environment constraints governing the reproduction of AMSA-YOLO.

---

## 1. System Python Isolation (Ubuntu 26.04 WSL2)
- **Status:** The system Python in this Ubuntu 26.04 installation is **Python 3.14.4** (`/usr/bin/python3`).
- **Rule:** **System Python 3.14.4 must not be used as the project runtime yet.**
- **Rationale:** Deep learning ecosystems (PyTorch, torchvision, ultralytics, opencv, scipy) do not have stable, prebuilt, CUDA-enabled binary wheels for Python 3.14. Modifying or replacing `/usr/bin/python3` via `apt` or creating symlinks can break the host OS distribution tooling.
- **Enforcement:** All project runtimes will be installed and managed strictly in user-space using `uv` (e.g. `uv python install <version>` and `.venv`), leaving system Python untouched.

---

## 2. Unknown Kaggle Python ABI
- **Status:** The exact Python minor version and CPython ABI tag (`cp310`, `cp311`, etc.) on the target Kaggle environment is currently unconfirmed.
- **Rule:** **No virtual environment or wheel downloads may be locked until the Kaggle Python ABI is probed.**
- **Rationale:** Wheelhouse binaries (`.whl`) are compiled against specific CPython ABI tags (e.g., `cp310-cp310-manylinux_2_28_x86_64`). Downloading wheels locally for the wrong Python version will cause binary mismatch errors when deploying offline (`Internet OFF`) to Kaggle.

---

## 3. PyTorch & CUDA Compatibility Constraints
- **Status:** The target Kaggle accelerator is an **NVIDIA RTX PRO 6000** (Ada Generation, ~48 GB VRAM, Compute Capability 8.9).
- **Rule:** **PyTorch and CUDA versions must not be selected until the Kaggle RTX PRO 6000 runtime is probed.**
- **Rationale:** The Ada Lovelace architecture requires a PyTorch build with CUDA 11.8 minimum (preferably CUDA 12.1 or CUDA 12.4) to leverage native Tensor Cores, FlashAttention, and optimal cuDNN kernels. Selecting an arbitrary PyTorch wheel prematurely risks runtime incompatibilities or missing kernel symbols on Kaggle.

---

## 4. NVIDIA Driver CUDA API vs. PyTorch CUDA Runtime
- **Technical Clarification:** **The NVIDIA driver CUDA version reported by `nvidia-smi` is not the same thing as the CUDA runtime bundled with PyTorch wheels.**
- **Details:**
  - `nvidia-smi` displays the **CUDA Driver API version** supported by the installed graphics driver (e.g. Driver `581.29` supporting CUDA `13.0`).
  - PyTorch wheels are compiled against a specific **CUDA Runtime API (libcudart)** version (e.g., `torch-2.4.0+cu124` uses CUDA Runtime 12.4).
  - The driver API is backward-compatible with any runtime API version equal to or lower than the driver limit (Driver CUDA 13.0 $\ge$ Runtime CUDA 12.4 $\ge$ 11.8).
  - A system-wide CUDA Toolkit (`nvcc`) is **not** required unless compiling custom C++/CUDA extensions from source. Standard binary PyTorch wheels embed the necessary runtime libraries (`libcudart`, `libcublas`, `libcudnn`, etc.).
