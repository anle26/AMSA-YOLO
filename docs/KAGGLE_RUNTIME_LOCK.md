# Kaggle Runtime Lock & Architectural Decision Record

**Document Date:** 2026-09-19  
**Status:** LOCKED & VERIFIED  
**Target Platform:** Kaggle Notebook (Internet OFF)  
**Target Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition  

---

## 1. Verified Kaggle Environment Specifications

The following hardware and runtime metrics were measured directly on Kaggle with Internet OFF and confirmed:

| Parameter | Measured Value | Technical Meaning / Role |
| :--- | :--- | :--- |
| **Python Version** | `3.12.13` | Exact CPython interpreter version on worker node |
| **Python SOABI** | `cpython-312-x86_64-linux-gnu` | CPython 3.12 ABI tag (`cp312`) for binary wheel resolution |
| **Architecture** | `x86_64` | Target CPU ISA for compiled wheels |
| **PyTorch Version** | `2.10.0+cu128` | Preinstalled PyTorch with CUDA 12.8 runtime support |
| **torchvision Version** | `0.25.0+cu128` | Preinstalled torchvision compiled against torch 2.10.0 |
| **PyTorch CUDA Runtime** | `12.8` (`torch.version.cuda`) | Active CUDA runtime embedded in PyTorch binaries |
| **cuDNN Version** | `91002` (cuDNN 9.1.0) | Deep learning primitive library used by PyTorch |
| **GPU Name** | `NVIDIA RTX PRO 6000 Blackwell Server Edition` | Primary accelerator on worker node |
| **Compute Capability** | `12.0` (Blackwell Architecture) | Hardware microarchitecture revision |
| **GPU VRAM** | `94.97 GiB` | Available high-bandwidth GPU memory |
| **Host Driver Version** | `580.159.04` | NVIDIA host kernel display driver |
| **NVIDIA-SMI CUDA Cap** | `13.0` | Maximum CUDA API capability supported by driver |

---

## 2. Technical Clarifications & Core Decisions

### Driver CUDA Capability (13.0) vs. PyTorch CUDA Runtime (12.8)
- `nvidia-smi` reports `CUDA Version: 13.0`. This denotes the maximum CUDA Driver API level supported by the host display driver (`580.159.04`).
- `torch.version.cuda` reports `12.8`. This is the CUDA Runtime API version statically linked/bundled into the preinstalled PyTorch distribution (`2.10.0+cu128`).
- The driver capability is backward-compatible with CUDA 12.8 runtime calls, providing optimal Blackwell Tensor Core execution.

### Preserving Kaggle Base PyTorch & Torchvision
- **Decision:** PyTorch (`2.10.0+cu128`) and torchvision (`0.25.0+cu128`) are already compiled for Blackwell (sm_120) and baked into the Kaggle base image.
- **Action:** We **MUST NOT** download, package, or overwrite `torch`, `torchvision`, CUDA toolkits, cuDNN, or NVIDIA runtime libraries in the primary wheelhouse.
- **Benefit:** Avoids transferring massive multi-gigabyte wheels (often >2.5 GB) into Kaggle Datasets and eliminates the risk of overwriting vendor-optimized Blackwell kernels with generic or incompatible community wheels.

### Local Alignment via `uv`
- Local Ubuntu WSL system Python (`3.14.4`) remains completely untouched.
- A hermetic CPython `3.12.13` has been provisioned via `uv` at `/home/an/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`.
- Project virtual environment `.venv` is anchored strictly to Python `3.12.13`, ensuring 100% ABI compatibility (`cp312`) during dependency resolution and offline wheel collection.
