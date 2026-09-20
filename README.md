# AMSA-YOLO: Reproducible Offline Kaggle Implementation

## Project Goal
The goal of this project is to build a hermetic, fully reproducible, offline execution environment for **AMSA-YOLO** (an advanced object detection architecture enhancing YOLOv8 with Adaptive Multi-Scale Attention mechanisms) and deploy it to **Kaggle** under **Internet OFF** constraints targeting an **NVIDIA RTX PRO 6000** GPU.

---

## AMSA-YOLO Reproduction
AMSA-YOLO enhances standard YOLOv8 by introducing:
1. **Adaptive Multi-Scale Attention (AMSA)** modules designed to capture cross-scale feature interactions and improve small/dense object detection. The paper establishes that AMSA modules are applied at **P3, P4, P5** feature levels in the backbone before feature-pyramid fusion.
2. **Enhanced Feature Aggregation** in the neck network to mitigate information loss during downsampling.
3. **Optimized Loss and Anchor Assignment** tailored for multi-scale feature alignment.

The reproduction strategy begins with a **paper-faithful standalone implementation** before deciding YOLO integration:
- `src/amsa/`:
  - `scale_aware.py`: Scale-Aware Module (initial scale embedding dimension $D = 64$).
  - `spatial_attention.py`: Adaptive Spatial Attention.
  - `channel_attention.py`: Adaptive Channel Attention.
  - `fusion.py`: Multi-scale Feature Fusion.
  - `amsa.py`: Top-level AMSA module orchestrating scale-aware, spatial, channel, and fusion paths.
- `tests/`: Comprehensive unit tests verifying tensor shapes ($P_3=80\times80, P_4=40\times40, P_5=20\times20$ for $640\times640$ inputs), numerical stability, and gradient flow before YOLO integration.
- `configs/`: Model architectures (YAML) and training hyperparameter presets (once standalone modules pass testing).
- `src/training/`: Training loops, loss wrappers, metric callbacks, and validation routines.

---

## Verified Kaggle Environment (Locked)
The Kaggle execution environment has been probed and locked (see `docs/KAGGLE_RUNTIME_LOCK.md`):
- **Operating Platform:** Kaggle Container Linux `x86_64` (Internet OFF)
- **Primary Accelerator:** **NVIDIA RTX PRO 6000 Blackwell Server Edition**
  - **VRAM:** 94.97 GiB
  - **Compute Capability:** 12.0 (Blackwell Architecture)
  - **Host Driver:** `580.159.04` (Supports CUDA 13.0 Driver API)
- **Preinstalled Deep Learning Stack:**
  - **Python:** `3.12.13` (SOABI: `cpython-312-x86_64-linux-gnu`)
  - **PyTorch:** `2.10.0+cu128` (CUDA Runtime: `12.8`, cuDNN: `91002`)
  - **torchvision:** `0.25.0+cu128`
- **Core Policy:** Kaggle's preinstalled `torch` and `torchvision` builds are preserved to exploit Blackwell hardware optimizations and avoid multi-gigabyte wheel transfers.

---

## Local WSL Environment
Local prototyping and validation are conducted in a controlled WSL2 environment:
- **Operating System:** Ubuntu 26.04.1 LTS (`resolute`)
- **Kernel / Architecture:** `6.18.33.2-microsoft-standard-WSL2` (`x86_64`)
- **System Python:** Python 3.14.4 (`/usr/bin/python3`) ? *isolated and not modified*.
- **Project Python:** Hermetic CPython `3.12.13` managed via `uv` at `/home/an/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`.
- **Virtual Environment:** `.venv` (Python 3.12.13, matching Kaggle ABI `cp312`).
- **Local Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
- **Host Graphics Driver:** Version `581.29` (providing CUDA 13.0 driver API passthrough).
- **Constraints:** Batch size and image resolutions locally must be throttled to fit within 6GB VRAM during smoke tests.

---

## Offline Wheelhouse Strategy
Because Kaggle execution runs with network access disabled, all non-base dependencies are packaged offline:
1. **ABI-Matched Wheelhouse:** A complete collection of 28 pre-downloaded `.whl` files matching Kaggle's Linux `x86_64` Python 3.12 (`cp312`) ABI is housed in `wheelhouse/` (see `wheelhouse/MANIFEST.txt`).
2. **Exclusion of Base Stack:** `torch`, `torchvision`, and low-level NVIDIA runtime libraries are excluded from the primary wheelhouse to prevent overwriting Kaggle's Blackwell-optimized binaries.
3. **Self-Contained Installation:** The offline Kaggle bootstrap script executes:
   ```bash
   uv pip install \
     --python /opt/venv/bin/python \
     --offline \
     --no-index \
     --no-deps \
     --find-links=wheelhouse/ \
     -r requirements/wheelhouse.txt
   ```
4. **No External Network Calls:** Auto-downloading of pretrained weights is disabled; model weights and assets are bundled directly.

---

## Planned Reproduction Stages
1. **Stage 1: Environment Audit & Safe Baseline Setup (Completed)**
   - Audit Ubuntu WSL system properties, GPU passthrough, and driver status (`docs/ENVIRONMENT_NOTES.md`).
2. **Stage 2: Tooling & Clean Project Foundation (Completed)**
   - Install `uv` safely in user space without touching system Python (`0.12.17`).
   - Scaffold structured repository and research-grade `.gitignore`.
   - Document environment policies in `docs/ENVIRONMENT_NOTES.md`.
3. **Stage 3: Kaggle Runtime Probing (Completed)**
   - Developed and executed runtime probe on Kaggle (`kaggle/probe_runtime.py`).
   - Verified Python 3.12.13, RTX PRO 6000 Blackwell (94.97 GiB), PyTorch 2.10.0+cu128.
   - Locked specifications in `docs/KAGGLE_RUNTIME_LOCK.md`.
4. **Stage 4: Environment Alignment & Dependency Inputs (Completed)**
   - Provisioned matching CPython `3.12.13` locally using `uv`.
   - Initialized project virtual environment `.venv` using Python 3.12.
   - Prepared dependency inputs (`requirements/base.in`, `requirements/kaggle-base-constraints.txt`).
5. **Stage 5: Baseline Selection, Dependency Locking & Wheelhouse Assembly (Completed)**
   - Formulated Ultralytics baseline decision (`ultralytics==8.2.103` in `docs/ULTRALYTICS_VERSION_DECISION.md`).
   - Resolved full dependency graph in `requirements/base.lock`.
   - Downloaded 28 verified binary wheels (`cp312`) to `wheelhouse/` (~151 MB, torch/cuda excluded).
   - Generated `wheelhouse/MANIFEST.txt` and `wheelhouse/SHA256SUMS.txt`.
   - Validated offline installation in `.venv-offline-test`.
   - Created Kaggle bootstrap script (`kaggle/bootstrap_offline.sh`) and usage guide (`kaggle/BOOTSTRAP_README.md`).
6. **Stage 6: Kaggle Offline Bootstrap Validation (Completed)**
   - Uploaded wheelhouse dataset to Kaggle and verified `bootstrap_offline.sh` under Internet OFF on NVIDIA RTX PRO 6000 Blackwell Server Edition.
   - Verified Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, ultralytics 8.2.103, YOLO architecture instantiation without downloads, and CUDA matmul.
   - Finalized targeted post-bootstrap runtime verification checks in `kaggle/bootstrap_offline.sh` and `kaggle/validate_offline_bootstrap_notebook.py`.
7. **Stage 7: Paper-Faithful Standalone AMSA Module Implementation & Unit Tests**
   - Implement standalone AMSA modules applied at P3, P4, P5 levels (`scale_aware.py`, `spatial_attention.py`, `channel_attention.py`, `fusion.py`, `amsa.py`).
   - Write comprehensive unit tests in `tests/` (`test_scale_aware.py`, `test_spatial_attention.py`, `test_channel_attention.py`, `test_fusion.py`, `test_amsa.py`) to verify tensor shapes, numerical stability, and gradient flow before YOLO integration.
8. **Stage 8: Local Smoke Training on RTX 3050**
   - Run short sanity training on local RTX 3050 with throttled batch/image resolution.
9. **Stage 9: Kaggle Offline Deployment & Full-Scale Benchmark Replication**
   - Upload dataset and execute full benchmark reproduction on Kaggle RTX PRO 6000 with Internet OFF.

---

## Environment Status
- [x] Phase 1: Ubuntu WSL audit completed.
- [x] Phase 2: Tooling & clean workspace scaffold completed (`uv 0.12.17`, repo structure, `.gitignore`).
- [x] Phase 3: Kaggle runtime probed & locked (Python 3.12.13, RTX PRO 6000 Blackwell 94.97 GiB, PyTorch 2.10.0+cu128).
- [x] Phase 4: Local environment aligned (`uv` CPython 3.12.13, `.venv` created).
- [x] Phase 5: Dependency locked (`base.lock`), 28 binary wheels built into `wheelhouse/`, manifest generated, offline tested.
- [x] Phase 6: Kaggle offline bootstrap validation completed (Internet OFF, RTX PRO 6000 Blackwell).
- [ ] Phase 7: Standalone AMSA module implementation & unit tests.
- [ ] Phase 8: Local smoke training on RTX 3050.
- [ ] Phase 9: Full-scale benchmark replication on Kaggle RTX PRO 6000 (Internet OFF).
