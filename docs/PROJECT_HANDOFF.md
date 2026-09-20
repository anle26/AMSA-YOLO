# AMSA-YOLO Reproduction: Project Handoff & Architecture Context

**Document Version:** 1.1.0  
**Last Updated:** 2026-09-20  
**Target Repository:** `/home/an/research/AMSA-YOLO`  
**Handoff Document Path:** `/home/an/research/AMSA-YOLO/docs/PROJECT_HANDOFF.md`  
**Target Platform:** Kaggle Container Linux `x86_64` (Internet OFF)  
**Target Accelerator:** NVIDIA RTX PRO 6000 Blackwell Server Edition  

---

## 1. Project Goal
The primary objective of this project is to construct a hermetic, fully reproducible, offline execution environment to implement and evaluate **AMSA-YOLO** (*Adaptive Multi-Scale Attention YOLO*), reproducing its architectural enhancements and benchmark claims under strict **Kaggle Internet OFF** constraints.

Reference Publication:
- **Title:** *AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism*
- **Authors:** Canjin Wang, Peng Sun, Chunhui Yang, Xianglong Teng, Rijun Wang
- **Publication:** *Neural Networks*, Volume 197, Article 108545 (January 2026)
- **Comparative Baseline:** YOLOv8 Small (`yolov8s`)

---

## 2. Verified Environments

### A. Target Remote Environment: Kaggle (Internet OFF) ? LOCKED & VERIFIED
Probed directly on Kaggle with network disabled and verified via `docs/KAGGLE_RUNTIME_LOCK.md`:
- **Operating Platform:** Kaggle Container Linux `x86_64` (Internet OFF)
- **Python Version:** `3.12.13`
- **Python SOABI:** `cpython-312-x86_64-linux-gnu` (`cp312`)
- **Architecture:** `x86_64`
- **Primary Accelerator:** `NVIDIA RTX PRO 6000 Blackwell Server Edition`
  - **Compute Capability:** `12.0` (Blackwell Architecture, `sm_120`)
  - **VRAM Total:** `94.97 GiB`
- **Host Display Driver:** `580.159.04` (Supports CUDA 13.0 Driver API)
- **Preinstalled Deep Learning Base Stack:**
  - `torch==2.10.0+cu128` (Embedded CUDA Runtime: `12.8`, cuDNN: `91002`)
  - `torchvision==0.25.0+cu128`
- **CRITICAL ARCHITECTURAL DISTINCTION:**
  - `nvidia-smi` reports `CUDA Version: 13.0` (host display driver API limit).
  - PyTorch internally executes against embedded CUDA Runtime `12.8` (`torch.version.cuda`).
  - The driver API is backward-compatible with CUDA 12.8.
  - The Kaggle base PyTorch stack is pre-compiled for Blackwell (`sm_120`) and **must be preserved**.

### B. Local Development Environment: Ubuntu WSL2
Probed and configured locally:
- **Operating System:** Ubuntu 26.04.1 LTS (`resolute`) on Windows 11 WSL2
- **Kernel / Architecture:** `6.18.33.2-microsoft-standard-WSL2` (`x86_64`)
- **System Python:** Python `3.14.4` (`/usr/bin/python3`) ? *Strictly isolated; NEVER modified*.
- **Tooling Python:** Hermetic CPython `3.12.13` managed via `uv` (`0.12.17`) at `/home/an/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`.
- **Virtual Environment:** `.venv` (Python 3.12.13, matching Kaggle ABI `cp312`).
- **Local Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, Driver `581.29`).
- **Local Role:** Prototyping, module unit testing, shape verification, and throttled smoke tests.

---

## 3. Repository Structure & Layout

```
/home/an/research/AMSA-YOLO/
??? .gitignore                                # Git ignore rules (weights, cache, wheels, archives)
??? pyproject.toml                            # Build definition (Hatchling) & dev dependencies
??? README.md                                 # High-level overview & project roadmap
??? amsa-yolo-offline-kaggle.zip              # Pre-packaged archive for Kaggle dataset (ignored)
??? configs/                                  # Model YAML definitions & training presets
?   ??? .gitkeep
??? docs/                                     # Architectural & technical documentation
?   ??? ENVIRONMENT_NOTES.md                  # Isolation principles & verified runtime notes
?   ??? KAGGLE_RUNTIME_LOCK.md                # Locked runtime specifications of Kaggle node
?   ??? ULTRALYTICS_VERSION_DECISION.md       # Decision record for Ultralytics 8.2.103
?   ??? PROJECT_HANDOFF.md                    # THIS RECOVERY & HANDOFF DOCUMENT
??? kaggle/                                   # Kaggle probe & bootstrap scripts
?   ??? BOOTSTRAP_README.md                   # Step-by-step Kaggle offline bootstrap guide
?   ??? PROBE_INSTRUCTIONS.md                 # Initial probe execution guide
?   ??? probe_runtime.py                      # Standalone runtime probe script
?   ??? probe_runtime_notebook.py             # Single-cell notebook probe script
?   ??? bootstrap_offline.sh                  # Hermetic offline bootstrap shell script
?   ??? validate_offline_bootstrap_notebook.py # Complete notebook validation cell
??? requirements/                             # Dependency specifications & lockfiles
?   ??? README.md                             # Architectural rationale for wheelhouse
?   ??? base.in                               # Direct user-space dependency inputs
?   ??? base.lock                             # Resolved dependency lockfile (via uv)
?   ??? kaggle-base-constraints.txt           # Kaggle base image constraints (torch/torchvision)
?   ??? wheelhouse.txt                        # Pinned requirements for non-torch wheelhouse layer
??? src/                                      # Source implementation
?   ??? amsa/                                 # Standalone AMSA module implementation
?   ?   ??? scale_aware.py                    # Scale-Aware Module (dim D=64)
?   ?   ??? spatial_attention.py              # Adaptive Spatial Attention
?   ?   ??? channel_attention.py              # Adaptive Channel Attention
?   ?   ??? fusion.py                         # Feature Fusion
?   ?   ??? amsa.py                           # Full AMSA top-level layer
?   ??? training/                             # Training orchestration, loss & metric wrappers
?       ??? .gitkeep
??? tests/                                    # Unit tests (tensor shapes, numerical stability, gradients)
?   ??? test_scale_aware.py
?   ??? test_spatial_attention.py
?   ??? test_channel_attention.py
?   ??? test_fusion.py
?   ??? test_amsa.py
??? wheelhouse/                               # 28 binary Linux x86_64 cp312 wheels (~151 MB)
    ??? MANIFEST.txt                          # Detailed wheel manifest and tags
    ??? SHA256SUMS.txt                        # Cryptographic SHA256 checksums of all wheels
    ??? *.whl                                 # Pre-downloaded binary wheels (torch/CUDA excluded)
```

---

## 4. Dependency Strategy & Wheelhouse Architecture

### Two-Tier Layered Architecture
1. **Kaggle Base Platform Layer (System Site-Packages):**
   - Provides `torch 2.10.0+cu128`, `torchvision 0.25.0+cu128`, CUDA runtime `12.8`, cuDNN `91002`, and low-level NVIDIA runtime libraries.
   - Preserved to guarantee native hardware acceleration on the Blackwell architecture without transferring multi-gigabyte wheels.
2. **Offline Wheelhouse Layer (Isolated Venv):**
   - Provides all 28 non-base direct and transitive dependencies pinned in `requirements/wheelhouse.txt`.
   - Total archive size: ~151 MB.
   - All wheels match `cp312` ABI or `py3-none-any`.

### Forbidden Wheels Guard
The wheelhouse strictly **MUST NOT** contain:
- `torch`
- `torchvision`
- `triton`
- `cuda-*` or `nvidia-*` runtime wheels

Both `kaggle/bootstrap_offline.sh` and repository safety checks enforce this via automatic glob guards (`torch-*`, `torchvision-*`, `nvidia_*`, `triton-*`).

---

## 5. Ultralytics Baseline Decision: `ultralytics==8.2.103`

- **Selection:** `ultralytics==8.2.103` (Released September 28, 2024).
- **CRITICAL SCIENTIFIC DISTINCTION:**
  - The published AMSA-YOLO paper (*Neural Networks*, 2026) does **NOT** disclose an Ultralytics package version or commit SHA, nor is there an official public repository.
  - `8.2.103` is a **project-selected controlled reproduction baseline**.
  - **Do NOT present `8.2.103` as an author-stated fact or verified paper fact.**
- **Rationale for `8.2.103`:**
  - It is the final stable release of the YOLOv8 series immediately preceding Ultralytics `8.3.0` (which introduced YOLO11 and overhauled internal registries and YAML parser semantics).
  - Preserves intact YOLOv8 anchor-free loss modules, head structures, and C2f building blocks.
  - Fully compatible with Python 3.12 and PyTorch 2.10.0 without unpinned dependency conflicts.

---

## 6. Kaggle Bootstrap Design & Execution Flow

The bootstrap process in `kaggle/bootstrap_offline.sh` and `kaggle/validate_offline_bootstrap_notebook.py` is engineered as follows:

```
[Kaggle Environment (Internet OFF)]
   |
   +--> 1. Detect mounted wheelhouse directory (e.g. /kaggle/input/**/wheelhouse)
   |
   +--> 2. Run forbidden-wheel safety guard (scans for torch/nvidia/triton wheels)
   |
   +--> 3. Inspect preinstalled base torch/torchvision via /usr/bin/python3
   |
   +--> 4. Initialize isolated virtual environment:
   |      uv venv /opt/venv \
   |        --clear \
   |        --python /usr/bin/python3 \
   |        --system-site-packages \
   |        --no-managed-python
   |
   +--> 5. Install offline wheels:
   |      uv pip install \
   |        --python /opt/venv/bin/python \
   |        --offline \
   |        --no-index \
   |        --no-deps \
   |        --find-links "${WHEELHOUSE_DIR}" \
   |        -r requirements/wheelhouse.txt
   |
   +--> 6. Execute targeted runtime verification via /opt/venv/bin/python
```

### Rationale for `--no-deps`:
- `ultralytics` declares `torch>=1.8.0` in its distribution metadata.
- `torch` and `torchvision` are already present in Kaggle's container and inherited into `/opt/venv` via `--system-site-packages`.
- However, if `uv pip install` runs with index dependency resolution enabled, `uv`'s resolver searches `--find-links` for a `torch` wheel matching `torch>=1.8.0`. Because `torch` is deliberately absent from `wheelhouse/`, resolution aborts with a missing package error.
- Passing `--no-deps` bypasses this resolver conflict, directly installing the verified, pre-locked non-base wheels into `/opt/venv`.

---

## 7. Validation History & Solved Issues

### Manual Kaggle Validation Passed:
The bootstrap flow was executed and manually verified directly on an active Kaggle notebook instance with Internet OFF targeting the **NVIDIA RTX PRO 6000 Blackwell Server Edition**:
- **Python:** `3.12.13`
- **PyTorch:** `2.10.0+cu128` (CUDA runtime `12.8`)
- **Torchvision:** `0.25.0+cu128`
- **Ultralytics:** `8.2.103` (located at `/opt/venv/lib/python3.12/site-packages`)
- **CUDA Matmul Test:** `SUCCESS` (GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GiB VRAM, sm_120)
- **Ultralytics Architecture Test:** `from ultralytics import YOLO; model = YOLO("yolov8s.yaml")` `SUCCESS` (zero network downloads)

### Key Issue Solved: False Failure with `uv pip check`
- In earlier iterations, `uv pip check` was executed post-install.
- `uv pip check` emitted false errors because it did not treat distributions installed in the underlying system site-packages (`torch`, `torchvision`) as satisfying the dependencies of packages installed directly inside `/opt/venv`.
- **Solution:** `uv pip check` was removed as a fatal gate (commit `afb48f6`). It was replaced with **Targeted Runtime Verification**:
  1. Direct imports of `torch`, `torchvision`, and `ultralytics`.
  2. Semantic version constraint verification via `packaging.version`:
     - `torch >= 1.8.0`
     - `torchvision >= 0.9.0`
     - `ultralytics == 8.2.103`
  3. Tensor computation sanity check on GPU (`torch.matmul` + `torch.cuda.synchronize()`).
  4. Model initialization check (`YOLO("yolov8s.yaml")`).
  5. Verification of core dependencies (`numpy`, `scipy`, `pandas`, `cv2`, `PIL`, `yaml`, `tqdm`, `psutil`).

---

## 8. Current Project Phase Status

| Phase | Description | Status | Repository Evidence |
| :--- | :--- | :--- | :--- |
| **Phase 1** | WSL environment audit | **COMPLETE** | Documented in `docs/ENVIRONMENT_NOTES.md` and `README.md`. |
| **Phase 2** | Project & tooling setup | **COMPLETE** | `uv 0.12.17`, `pyproject.toml`, `.gitignore`, directory scaffold (commit `5f3a841`). |
| **Phase 3** | Kaggle runtime probe | **COMPLETE** | `kaggle/probe_runtime.py`, specifications locked in `docs/KAGGLE_RUNTIME_LOCK.md`. |
| **Phase 4** | Python/ABI alignment | **COMPLETE** | CPython `3.12.13` (`cp312`) provisioned in local `.venv`, matching Kaggle SOABI. |
| **Phase 5** | Dependency lock & wheelhouse | **COMPLETE** | `base.lock`, 28 wheels in `wheelhouse/`, `MANIFEST.txt`, `SHA256SUMS.txt`, local offline test. |
| **Phase 6** | Kaggle offline bootstrap validation | **COMPLETE** | Validated on Kaggle RTX PRO 6000 with Internet OFF; commit `afb48f6` finalized scripts; `README.md` updated. |
| **Phase 7** | AMSA implementation (standalone) | **NOT STARTED** | Standalone architecture and unit test plan defined below. |
| **Phase 8** | Local smoke training | **NOT STARTED** | `src/training/`, `configs/`, `scripts/` contain only `.gitkeep`. |
| **Phase 9** | Kaggle full benchmark reproduction | **NOT STARTED** | Dataset training and benchmark evaluation not yet initiated. |

---

## 9. Paper-Faithful AMSA Architecture Constraints (Phase 7)

The AMSA-YOLO paper (*Neural Networks*, Wang et al., 2026) specifies that AMSA modules are applied at the **P3, P4, P5** feature levels of the backbone **prior to feature-pyramid fusion**.

### Mathematical & Dimensional Constraints:
- **Input Tensor:** $X \in \mathbb{R}^{B \times C \times H \times W}$
- **Scale Feature Levels (for standard $640 \times 640$ input):**
  - **P3:** $80 \times 80$ ($H/8 \times W/8$)
  - **P4:** $40 \times 40$ ($H/16 \times W/16$)
  - **P5:** $20 \times 20$ ($H/32 \times W/32$)
- **Initial Scale Embedding Dimension:** $D = 64$
- **AMSA Four-Stage Composition:**
  1. **Scale-Aware Module:** Generates scale-adaptive descriptor embeddings across receptive field branches.
  2. **Adaptive Spatial Attention:** Captures fine-grained positional dependencies across multi-scale spatial regions.
  3. **Adaptive Channel Attention:** Dynamically recalibrates cross-channel feature responses conditioned on scale context.
  4. **Feature Fusion:** Synthesizes spatial, channel, and scale-aware representations via residual gating.

### Non-Negotiable Boundary Rules for Initial Implementation:
- **NO P2** level integration during the initial implementation.
- **NO neck redesign** or premature PANet/FPN restructuring.
- **NO loss modification** or anchor reassignment changes.
- **NO C2f redesign or `c2f_amsa.py`** modules yet.
- **NO changes to pretrained-weight loading behavior.**
- **NO decision on the exact YOLO insertion mechanism** until the standalone AMSA module has passed shape, numerical, and gradient tests.

---

## 10. Phase 7 Execution Plan: Standalone AMSA & Unit Tests

Phase 7 begins strictly with a clean, modular, standalone implementation:

### Source Modules (`src/amsa/`):
- `scale_aware.py`: Scale-Aware Module computing multi-scale descriptor tensors ($D=64$).
- `spatial_attention.py`: Adaptive Spatial Attention map generator.
- `channel_attention.py`: Adaptive Channel Attention weight generator.
- `fusion.py`: Adaptive feature fusion layer.
- `amsa.py`: Unified `AMSA` module taking $X \in \mathbb{R}^{B \times C \times H \times W}$ and outputting recalibrated feature maps of identical shape.

### Verification Unit Tests (`tests/`):
- `test_scale_aware.py`: Shape check across varying batch sizes and spatial sizes ($80\times80, 40\times40, 20\times20$).
- `test_spatial_attention.py`: Attention map bounds ($[0, 1]$), spatial broadcast invariance.
- `test_channel_attention.py`: Channel recalibration dimensions, non-saturation checks.
- `test_fusion.py`: Shape consistency, residual connection verification.
- `test_amsa.py`: Full end-to-end forward pass, backward gradient propagation (checking for non-zero gradients and zero NaNs/Infs).

---

## 11. Invariant Rules: What Must NOT Be Changed Without Explicit Approval

1. **DO NOT modify the system Python (`/usr/bin/python3`) in Ubuntu WSL.** Always execute via `.venv` powered by `uv` CPython 3.12.13.
2. **DO NOT download or add `torch`, `torchvision`, `triton`, or `nvidia-*` wheels to the wheelhouse.** Kaggle's base container provides the vendor-optimized Blackwell stack.
3. **DO NOT remove `--no-deps` from the offline `uv pip install` invocation in the bootstrap scripts.** Doing so causes `uv` to search for `torch` in the wheelhouse and fail.
4. **DO NOT restore `uv pip check` as a fatal exit condition during bootstrap.** It produces false-negative failures with `--system-site-packages`.
5. **DO NOT upgrade or arbitrarily change `ultralytics==8.2.103`** without an explicit version decision record.
6. **DO NOT claim `ultralytics==8.2.103` is the paper's original author version.** It is a controlled engineering reproduction baseline.
7. **DO NOT introduce online weight downloading calls (`YOLO('yolov8s.pt')`) in offline bootstrap or evaluation routines.** Always initialize models from architecture specifications (`YOLO('yolov8s.yaml')`) or offline pre-staged checkpoints.
8. **DO NOT assume custom C2f blocks (`c2f_amsa.py`) or premature YOLO modifications** before standalone AMSA passes all shape, numerical, and gradient unit tests.
