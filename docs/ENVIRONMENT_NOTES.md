# Environment Notes & Constraints: AMSA-YOLO Reproducibility

These technical notes document non-negotiable architectural boundaries and environment constraints governing the reproduction of AMSA-YOLO.

> [!IMPORTANT]
> `docs/KAGGLE_RUNTIME_LOCK.md` is the authoritative runtime record measured directly on the target platform with Internet OFF.

---

## 1. System Python Isolation (Ubuntu 26.04 WSL2)
- **Status:** The system Python in this Ubuntu 26.04 installation is **Python 3.14.4** (`/usr/bin/python3`).
- **Rule:** **System Python 3.14.4 must not be used as the project runtime.**
- **Rationale:** Deep learning ecosystems (PyTorch, torchvision, ultralytics, opencv, scipy) do not have stable, prebuilt, CUDA-enabled binary wheels for Python 3.14. Modifying or replacing `/usr/bin/python3` via `apt` or creating symlinks risks breaking host OS distribution tooling.
- **Enforcement:** All project runtimes are installed and managed strictly in user-space using `uv` (CPython `3.12.13` at `/home/an/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12` and `.venv`), leaving system Python untouched.

---

## 2. Kaggle Python ABI & Runtime (Verified)
- **Historical Pre-Probe Assumption (OBSOLETE):** Prior to Phase 3 probing, the Kaggle Python minor version and CPython ABI tag (`cp310`, `cp311`, etc.) were unconfirmed.
- **Verified Kaggle Runtime:** Probed and locked in `docs/KAGGLE_RUNTIME_LOCK.md`:
  - **Python Version:** `3.12.13`
  - **Python SOABI:** `cpython-312-x86_64-linux-gnu` (`cp312`)
  - **Architecture:** `x86_64`
- **Enforcement:** Local `.venv` and all offline wheels in `wheelhouse/` are aligned strictly to `cp312`, guaranteeing 100% binary compatibility on Kaggle.

---

## 3. Kaggle Accelerator & Deep Learning Stack (Verified)
- **Historical Pre-Probe Assumption (OBSOLETE):** Prior to Phase 3 probing, the target GPU was hypothesized as an *NVIDIA RTX PRO 6000 Ada Generation (~48 GB VRAM, Compute Capability 8.9)*.
- **Verified Target Accelerator:** Probed and locked in `docs/KAGGLE_RUNTIME_LOCK.md`:
  - **Accelerator:** **NVIDIA RTX PRO 6000 Blackwell Server Edition**
  - **Compute Capability:** `12.0` (Blackwell Architecture, `sm_120`)
  - **Total VRAM:** `94.97 GiB`
  - **Preinstalled PyTorch:** `2.10.0+cu128` (CUDA Runtime `12.8`, cuDNN `91002`)
  - **Preinstalled torchvision:** `0.25.0+cu128`
- **Rule:** Kaggle's preinstalled `torch` and `torchvision` builds are compiled natively for Blackwell and must be preserved via `/opt/venv --system-site-packages`. They must **NEVER** be overwritten or packaged into the offline wheelhouse.

---

## 4. NVIDIA Driver CUDA API vs. PyTorch CUDA Runtime
- **Technical Clarification:** **The NVIDIA driver CUDA version reported by `nvidia-smi` is not the same thing as the CUDA runtime bundled with PyTorch wheels.**
- **Details:**
  - `nvidia-smi` on Kaggle displays `CUDA Version: 13.0` (supported by host driver `580.159.04`).
  - PyTorch wheels execute against embedded **CUDA Runtime API (libcudart) 12.8** (`torch.version.cuda`).
  - The driver API is backward-compatible with any runtime API version equal to or lower than the driver limit (Driver CUDA 13.0 $\ge$ Runtime CUDA 12.8).
  - Standard binary PyTorch wheels embed necessary runtime libraries (`libcudart`, `libcublas`, `libcudnn`, etc.), so a system-wide `nvcc` compiler is not required for inference or training.
