# Kaggle Offline Bootstrap Guide for AMSA-YOLO

This guide explains how to upload the verified offline wheelhouse as a Kaggle Dataset and run `bootstrap_offline.sh` inside an **Internet OFF** Kaggle notebook.

---

## Step 1: Upload the Wheelhouse to Kaggle Datasets

1. In your local WSL environment, the binary wheels and manifest are stored at:
   `/home/an/research/AMSA-YOLO/wheelhouse/`
2. Create a Kaggle Dataset containing the `wheelhouse/` folder:
   - On Kaggle, go to **Datasets** -> **New Dataset**.
   - Title: `amsa-yolo-wheelhouse` (or any custom name).
   - Drag and drop or upload all `.whl` files from `wheelhouse/` (or compress `wheelhouse/` as `wheelhouse.zip` and upload).
   - Click **Create**.

---

## Step 2: Attach Dataset to Your Kaggle Notebook

1. Open your Kaggle notebook configured with:
   - **Accelerator:** NVIDIA RTX PRO 6000 (Blackwell Server Edition)
   - **Internet:** **OFF**
2. In the right-hand panel, click **+ Add Data** (or **+ Add Input**).
3. Search for your uploaded dataset (e.g. `amsa-yolo-wheelhouse`) and click **Add**.
4. The wheels will be mounted at:
   `/kaggle/input/<dataset-slug>/wheelhouse` (or `/kaggle/input/<dataset-slug>/`)

---

## Step 3: Run the Bootstrap Script in a Notebook Cell

Paste and execute the following snippet in your first notebook cell:

```bash
%%bash
# Set path to the mounted dataset wheelhouse (or let auto-detection find it)
export WHEELHOUSE_DIR=$(find /kaggle/input -type d -name "wheelhouse" | head -n 1)

# If bootstrap_offline.sh is uploaded as a utility script:
bash /kaggle/input/<code-dataset>/bootstrap_offline.sh

# Alternatively, if pasting inline or running locally cloned repo:
# bash kaggle/bootstrap_offline.sh
```

---

## Step 4: What the Bootstrap Script Does

1. **Auto-detects the Wheelhouse:** Discovers `.whl` files in `/kaggle/input/` without hardcoding dataset slugs.
2. **Preserves Base PyTorch Stack:** Creates `/opt/venv` with `--system-site-packages`, preserving Kaggle's preinstalled:
   - `torch==2.10.0+cu128`
   - `torchvision==0.25.0+cu128`
   - CUDA Runtime `12.8` with native Blackwell (`sm_120`) kernel acceleration.
3. **Installs Offline Wheels via `--no-deps`:** Installs all 28 non-torch packages (`ultralytics==8.2.103`, `numpy`, `scipy`, `pandas`, `opencv-python`, etc.) using:
   ```bash
   uv pip install \
     --python /opt/venv/bin/python \
     --offline \
     --no-index \
     --no-deps \
     --find-links "${WHEELHOUSE_DIR}" \
     -r "${REQ_FILE}"
   ```
   **Why `--no-deps` is intentional:**
   - Kaggle's verified base image already provides PyTorch (`2.10.0+cu128`) and torchvision (`0.25.0+cu128`).
   - `/opt/venv` is created with `--system-site-packages` to inherit this vendor-optimized stack directly.
   - `requirements/wheelhouse.txt` represents an explicitly pinned, complete non-base dependency layer.
   - Torch, torchvision, and NVIDIA runtime wheels are deliberately omitted from the wheelhouse.
   - Allowing `uv` to resolve dependencies normally with `--offline --no-index --find-links` would incorrectly demand `torch` from the offline wheelhouse candidates because `ultralytics` declares `torch>=1.8.0`.
   - Using `--no-deps` installs the exact, pre-pinned wheels without triggering the index-based resolver.
4. **Verifies Dependency Consistency (`uv pip check`):**
   Immediately after installation, runs:
   ```bash
   uv pip check --python /opt/venv/bin/python
   ```
   This validates that all installed packages and the inherited base stack form a fully satisfied, consistent dependency graph.
5. **Verifies Runtime Functionality:**
   Runs comprehensive verification validating:
   - `import torch` and `import torchvision`
   - `import ultralytics` and `from ultralytics import YOLO`
   - Architecture initialization: `YOLO("yolov8s.yaml")` (zero network download)
   - CUDA availability, VRAM (94.97 GiB), compute capability (12.0)
   - CUDA tensor matrix multiplication test
   - All core package imports (`numpy`, `scipy`, `pandas`, `cv2`, `PIL`, `yaml`, `tqdm`, `psutil`)

