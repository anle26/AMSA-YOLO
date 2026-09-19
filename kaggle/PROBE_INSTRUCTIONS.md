# Kaggle Runtime Probing Instructions: AMSA-YOLO

This guide explains how to inspect and extract the exact hardware, Python ABI, and CUDA runtime parameters from a Kaggle environment to ensure 100% offline reproducibility locally.

---

## Step 1: Create a Kaggle Notebook

1. Navigate to [Kaggle](https://www.kaggle.com/) and log in.
2. Click **Create** -> **New Notebook** (or go to [kaggle.com/code](https://www.kaggle.com/code) and select **New Notebook**).
3. Set the notebook language to **Python**.

---

## Step 2: Configure Accelerator & Disable Internet

1. Look at the right-hand panel under **Notebook options**:
   - **Accelerator**: Expand the dropdown and select **GPU** -> Choose **NVIDIA RTX PRO 6000** (or the target accelerator assigned to your account/competition).
   - **Internet**: Toggle the switch to **OFF**.  
     *(This probe is completely offline and requires zero external internet access).*
2. Wait a few seconds for the Kaggle interactive session to start and connect to the GPU worker.

---

## Step 3: Run the Probe Script

There are two equally simple ways to run the probe:

### Option A: Single-Cell Paste (Fastest)
1. Open the local file `kaggle/probe_runtime_notebook.py`.
2. Copy the entire contents of the file.
3. Paste it directly into the first code cell of your Kaggle notebook.
4. Press **Shift + Enter** to execute the cell.

### Option B: Upload and Execute
1. In the Kaggle notebook menu, select **File** -> **Upload data** or add `probe_runtime.py` as a script utility.
2. In a code cell, run:
   ```bash
   !python probe_runtime.py
   ```

---

## Step 4: Capture and Return the Output

1. After the cell finishes executing, copy the **entire standard output** displayed below the cell.
2. In particular, ensure the final **`DECISION INPUTS`** block is completely captured, for example:
   ```text
   ================================================================================
   DECISION INPUTS
   ================================================================================
   PYTHON_VERSION=3.10.12
   PYTHON_SOABI=cpython-310-x86_64-linux-gnu
   ARCH=x86_64
   TORCH_VERSION=...
   TORCH_CUDA=...
   CUDNN_VERSION=...
   GPU_NAME=...
   COMPUTE_CAPABILITY=...
   GPU_VRAM_GIB=...
   NVIDIA_DRIVER=...
   NVIDIA_SMI_CUDA=...
   ================================================================================
   ```
3. Return to this conversation and paste the output into our chat (or save it into `kaggle/PROBE_RESULTS.txt` inside your project directory).

---

## What Happens Next?
Once the exact `DECISION INPUTS` are provided:
- We will provision the matching hermetic Python version locally using `uv python install <version>`.
- We will match the PyTorch and CUDA runtime wheels (`cu121`/`cu124`).
- We will assemble the offline wheelhouse (`wheelhouse/*.whl`) guaranteed to install without network access on Kaggle.
