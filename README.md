# AMSA-YOLO: Reproducible Offline Kaggle Implementation

## Project Goal
The goal of this project is to build a hermetic, fully reproducible, offline execution environment for **AMSA-YOLO** (an advanced object detection architecture enhancing YOLOv8 with Adaptive Multi-Scale Attention mechanisms) and deploy it to **Kaggle** under **Internet OFF** constraints targeting an **NVIDIA RTX PRO 6000** GPU.

---

## AMSA-YOLO Reproduction
AMSA-YOLO enhances standard YOLOv8 by introducing:
1. **Adaptive Multi-Scale Attention (AMSA)** modules designed to capture cross-scale feature interactions and improve small/dense object detection.
2. **Enhanced Feature Aggregation** in the neck network to mitigate information loss during downsampling.
3. **Optimized Loss and Anchor Assignment** tailored for multi-scale feature alignment.

The reproduction strategy focuses on clean modularization:
- `src/amsa/`: Custom attention layers, multi-scale blocks, and modified head/neck components.
- `src/training/`: Training loops, loss wrappers, metric callbacks, and validation routines.
- `configs/`: Model architectures (YAML) and training hyperparameter presets.
- `tests/`: Unit tests verifying tensor shapes, gradient flow, and parameter counts.

---

## Local WSL Environment
Local prototyping and validation are conducted in a controlled WSL2 environment:
- **Operating System:** Ubuntu 26.04.1 LTS (`resolute`)
- **Kernel / Architecture:** `6.18.33.2-microsoft-standard-WSL2` (`x86_64`)
- **System Python:** Python 3.14.4 (`/usr/bin/python3`) — *isolated and not modified*.
- **Package & Runtime Manager:** `uv` installed in user space (`~/.local/bin/uv`).
- **Local Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
- **Host Graphics Driver:** Version `581.29` (providing CUDA 13.0 driver API passthrough).
- **Constraints:** Batch size and image resolutions locally must be throttled to fit within 6GB VRAM during smoke tests.

---

## Target Kaggle Environment
Production runs and benchmark reproduction will execute on Kaggle notebooks:
- **Connectivity:** **Internet OFF** (mandatory strict offline execution).
- **Primary Accelerator:** **NVIDIA RTX PRO 6000** (Ada Generation, ~48 GB VRAM) or fallback accelerators.
- **Python Runtime:** Standard Kaggle container (typically Python 3.10 or 3.11 ABI — to be confirmed in Phase 3 probing).
- **Storage Constraints:** Kaggle `/kaggle/working` (ephemeral disk) and read-only Kaggle Datasets for dependencies and weights.

---

## Offline Wheelhouse Strategy
Because Kaggle execution runs with network access disabled, all dependencies must be resolved and packaged offline:
1. **ABI-Matched Wheelhouse:** A complete collection of pre-downloaded `.whl` files matching Kaggle's Linux `x86_64` Python version and compatible CUDA runtime will be placed into `wheelhouse/`.
2. **Self-Contained Installation:** The offline Kaggle bootstrap script executes:
   ```bash
   pip install --no-index --find-links=wheelhouse/ -r requirements/kaggle.txt
   ```
3. **No External Network Calls:** Code and training scripts will disable auto-downloading of pretrained weights (e.g., `yolov8n.pt` / `yolov8s.pt` will be bundled locally).

---

## Planned Reproduction Stages
1. **Stage 1: Environment Audit & Safe Baseline Setup (Completed)**
   - Audit Ubuntu WSL system properties, GPU passthrough, and driver status.
   - Record findings in `ENVIRONMENT_AUDIT.md`.
2. **Stage 2: Tooling & Clean Project Foundation (Completed)**
   - Install `uv` safely in user space without touching system Python.
   - Scaffold structured repository and research-grade `.gitignore`.
   - Document environment policies in `docs/ENVIRONMENT_NOTES.md`.
3. **Stage 3: Kaggle Runtime Probing (Current - Runtime probe pending execution on Kaggle)**
   - Provide standalone and notebook-friendly probe scripts (`kaggle/probe_runtime.py`, `kaggle/probe_runtime_notebook.py`).
   - Probe exact Python version, glibc version, CUDA runtime, and preinstalled packages on Kaggle.
4. **Stage 4: Hermetic Environment Setup with `uv`**
   - Provision matching Python version via `uv python install <version>`.
   - Create isolated project virtual environment (`.venv`).
5. **Stage 5: AMSA-YOLO Architecture Implementation & Unit Tests**
   - Implement attention modules, integrate into YOLOv8 architecture, and write shape/gradient tests.
6. **Stage 6: Offline Wheelhouse Preparation & Validation**
   - Resolve, download, and verify binary wheels for offline Kaggle deployment.
7. **Stage 7: Local Smoke Training & Verification**
   - Run short sanity training on local RTX 3050.
8. **Stage 8: Kaggle Offline Deployment & Benchmark Replication**
   - Upload dataset and wheelhouse, run full training on Kaggle RTX PRO 6000 with Internet OFF.

---

## Environment Status
- [x] Phase 1: Ubuntu WSL audit completed.
- [x] Phase 2: Tooling & clean workspace scaffold completed (`uv 0.12.17`, repo structure, `.gitignore`).
- [ ] Phase 3: Runtime probe pending execution on Kaggle.
- [ ] Phase 4: Project virtual environment created (pending Kaggle runtime probe).
- [ ] Phase 5: PyTorch / Ultralytics dependencies installed (deferred until Kaggle ABI match).
