# AMSA-YOLO: Dependency Resolution & Wheelhouse Architecture

This document details the dependency strategy, ABI compatibility model, and offline packaging principles for deploying AMSA-YOLO to Kaggle with **Internet OFF**.

---

## 1. Local and Kaggle ABI Parity (`cp312`)
- **Kaggle Target ABI:** `cpython-312-x86_64-linux-gnu` (CPython `3.12.13`, Linux `x86_64`).
- **Local WSL ABI:** `cpython-312-x86_64-linux-gnu` (provisioned via `uv` CPython `3.12.13` and `.venv`).
- **Significance:** Binary C-extensions and CPython wheels targeting `cp312-cp312-manylinux*` or `cp312-abi3-manylinux*` built or resolved locally are bit-for-bit compatible with the Kaggle environment.

---

## 2. Preinstalled Base Runtime: PyTorch & Torchvision
- Kaggle's verified environment contains:
  - `torch==2.10.0+cu128`
  - `torchvision==0.25.0+cu128`
  - CUDA Runtime: `12.8`
  - cuDNN: `91002`
- These packages are pre-compiled for the **NVIDIA RTX PRO 6000 Blackwell Server Edition** (sm_120, Compute Capability 12.0) with vendor-tuned GEMM and attention kernels.
- **Principle:** We leverage the base image's preinstalled `torch` and `torchvision` rather than re-installing them.

---

## 3. Rationale for Excluding PyTorch from the Primary Wheelhouse
1. **Size & Bandwidth Constraints:** PyTorch with CUDA 12.8 wheels exceeds 2.5 GB. Bundling torch into a Kaggle Dataset creates unnecessary upload/download friction.
2. **Architecture-Specific Optimizations:** Overwriting the Kaggle base PyTorch wheel with a generic wheel from PyPI risks installing kernels that lack Blackwell (sm_120) optimization or contain mismatched CUDA runtime versions.
3. **Reproducibility:** Reusing the platform's native torch ensures maximum runtime stability and zero risk of clobbering GPU driver interface bindings.

---

## 4. Fallback Wheelhouse Strategy (Contingency Plan)
- If Kaggle's base container updates or changes its default PyTorch version, or if an experimental custom operator requires rebuilding:
  - A fallback wheelhouse (`wheelhouse-fallback/`) can be compiled with explicit `--extra-index-url https://download.pytorch.org/whl/cu128`.
  - The fallback mechanism would download the exact matching CUDA-enabled torch/torchvision wheels as an isolated dataset layer.

---

## 5. Isolation of Local System Python (`3.14.4`)
- Ubuntu 26.04 WSL2 provides system Python `3.14.4` (`/usr/bin/python3`).
- Python 3.14 lacks mature, prebuilt scientific and deep learning wheels (PyTorch, torchvision, and many C-extensions are not yet released for 3.14).
- To prevent polluting or breaking host operating system utilities, system Python remains untouched. All development, resolution, and validation occur strictly inside `.venv` powered by uv-managed CPython `3.12.13`.

---

## 6. Offline Wheel Compatibility Validation Plan
When binary wheels are collected into `wheelhouse/`:
1. **Filename Tag Inspection:** Ensure every `.whl` matches either `py3-none-any.whl`, `cp312-abi3-manylinux*.whl`, or `cp312-cp312-manylinux*.whl`.
2. **No External Network Flag:** Verify installation readiness using:
   ```bash
   pip install --no-index --find-links=wheelhouse/ -r requirements/base.txt
   ```
3. **Import & Smoke Check:** Ensure all packages import cleanly without network access and without modifying existing `torch` or `torchvision` installations.
