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
3. **Installs Offline Wheels:** Installs all 28 non-torch packages (`ultralytics==8.2.103`, `numpy`, `scipy`, `pandas`, `opencv-python`, etc.) via `--no-index --find-links`.
4. **Verifies Stack:** Runs an automated test validating device count, VRAM (94.97 GiB), compute capability (12.0), and package imports.
