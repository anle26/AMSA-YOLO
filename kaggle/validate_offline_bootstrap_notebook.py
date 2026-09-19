# ==============================================================================
# AMSA-YOLO: Kaggle Offline Bootstrap & Validation Notebook Cell
# ==============================================================================
# INSTRUCTIONS:
# 1. Ensure Kaggle Notebook Settings:
#    - Accelerator: GPU (NVIDIA RTX PRO 6000 Blackwell Server Edition)
#    - Internet: OFF
# 2. Attach the dataset containing your wheelhouse directory.
# 3. Paste and run this ENTIRE block in a code cell.
# ==============================================================================

import os
import sys
import subprocess
import glob
import re
import json

print("=" * 80)
print(" AMSA-YOLO OFFLINE BOOTSTRAP VALIDATION")
print("=" * 80)

# 1. Define WHEELHOUSE_DIR
wheelhouse_dir = os.environ.get("WHEELHOUSE_DIR")
if not wheelhouse_dir:
    matches = glob.glob("/kaggle/input/**/wheelhouse", recursive=True)
    if matches:
        wheelhouse_dir = matches[0]
    else:
        if os.path.isdir("./wheelhouse"):
            wheelhouse_dir = os.path.abspath("./wheelhouse")
        else:
            whl_matches = glob.glob("/kaggle/input/**/*.whl", recursive=True)
            if whl_matches:
                wheelhouse_dir = os.path.dirname(whl_matches[0])

if not wheelhouse_dir or not os.path.isdir(wheelhouse_dir):
    print("[!] ERROR: Could not locate wheelhouse directory. Please set WHEELHOUSE_DIR.")
    sys.exit(1)

os.environ["WHEELHOUSE_DIR"] = wheelhouse_dir
print(f"[1] WHEELHOUSE_DIR = {wheelhouse_dir}")
whl_count = len(glob.glob(os.path.join(wheelhouse_dir, "*.whl")))
print(f"    Found {whl_count} wheels in wheelhouse.")

# 2. Show GPU via nvidia-smi
print("\n[2] Hardware Status (nvidia-smi):")
try:
    smi_res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    if smi_res.returncode == 0:
        lines = smi_res.stdout.strip().splitlines()
        print("\n".join(lines[:15]))
    else:
        print(f"nvidia-smi returned code {smi_res.returncode}: {smi_res.stderr}")
except Exception as e:
    print(f"nvidia-smi execution notice: {e}")

# 3. Run bootstrap_offline.sh
print("\n[3] Executing bootstrap_offline.sh...")
bootstrap_script = None
for candidate in [
    os.path.join(os.path.dirname(wheelhouse_dir), "kaggle", "bootstrap_offline.sh"),
    "./kaggle/bootstrap_offline.sh",
    "./bootstrap_offline.sh",
]:
    if os.path.isfile(candidate):
        bootstrap_script = candidate
        break

if not bootstrap_script:
    matches = glob.glob("/kaggle/input/**/bootstrap_offline.sh", recursive=True)
    if matches:
        bootstrap_script = matches[0]

bootstrap_returncode = -1
if bootstrap_script and os.path.isfile(bootstrap_script):
    print(f"    Running script at: {bootstrap_script}")
    subprocess.run(["chmod", "+x", bootstrap_script], check=False)
    boot_proc = subprocess.run(["bash", bootstrap_script], check=False)
    bootstrap_returncode = boot_proc.returncode
else:
    print("    bootstrap_offline.sh not found as standalone file. Running inline bootstrap with uv...")
    # NOTE: Using --no-deps because Kaggle base provides torch/torchvision via --system-site-packages;
    # wheelhouse represents the complete non-base layer and normal resolution would fail without torch wheel.
    boot_commands = f"""
    set -euo pipefail
    if ! command -v uv >/dev/null 2>&1; then
        echo "[!] uv is required but not installed."
        exit 1
    fi
    uv venv /opt/venv --python /usr/bin/python3 --system-site-packages --no-managed-python
    uv pip install --python /opt/venv/bin/python --offline --no-index --no-deps --find-links "{wheelhouse_dir}" $(find "{wheelhouse_dir}" -maxdepth 1 -name "*.whl")
    uv pip check --python /opt/venv/bin/python
    """
    boot_proc = subprocess.run(["bash", "-c", boot_commands], check=False)
    bootstrap_returncode = boot_proc.returncode

print(f"    Bootstrap exit code: {bootstrap_returncode}")

# 4. Run runtime verification using /opt/venv/bin/python
print("\n[4] Running Runtime Verification via /opt/venv/bin/python...")
venv_python = "/opt/venv/bin/python"

verify_code = """
import sys
import os
import json

res = {}

# Python
res["PYTHON_VERSION"] = sys.version.split()[0]

# PyTorch
try:
    import torch
    res["TORCH_VERSION"] = str(torch.__version__)
    res["TORCH_CUDA"] = str(torch.version.cuda)
    res["CUDA_AVAILABLE"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        res["GPU_NAME"] = str(torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        res["COMPUTE_CAPABILITY"] = f"{props.major}.{props.minor}"
        
        # Part C: Minimal CUDA Tensor Matrix Multiplication Test
        a = torch.randn(64, 64, device='cuda')
        b = torch.randn(64, 64, device='cuda')
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        res["CUDA_MATMUL"] = "SUCCESS"
    else:
        res["GPU_NAME"] = "NONE"
        res["COMPUTE_CAPABILITY"] = "NONE"
        res["CUDA_MATMUL"] = "NO_CUDA"
except Exception as e:
    res["TORCH_VERSION"] = f"ERROR: {e}"
    res["TORCH_CUDA"] = "N/A"
    res["CUDA_AVAILABLE"] = "False"
    res["GPU_NAME"] = "N/A"
    res["COMPUTE_CAPABILITY"] = "N/A"
    res["CUDA_MATMUL"] = f"ERROR: {e}"

# Torchvision
try:
    import torchvision
    res["TORCHVISION_VERSION"] = str(torchvision.__version__)
except Exception as e:
    res["TORCHVISION_VERSION"] = f"ERROR: {e}"

# Ultralytics & YOLO (Part D)
try:
    import ultralytics
    res["ULTRALYTICS_VERSION"] = str(ultralytics.__version__)
    from ultralytics import YOLO
    # Instantiate from architecture YAML only (random initial weights, zero download)
    model = YOLO("yolov8s.yaml")
    res["YOLO_IMPORT"] = "SUCCESS"
except Exception as e:
    res["ULTRALYTICS_VERSION"] = getattr(ultralytics, "__version__", f"ERROR: {e}")
    res["YOLO_IMPORT"] = f"ERROR: {e}"

# Core packages
for p in ['numpy', 'scipy', 'pandas', 'cv2', 'PIL', 'yaml', 'tqdm', 'psutil']:
    try:
        mod = __import__(p)
        print(f"  [+] {p:<15}: OK (v{getattr(mod, '__version__', 'N/A')})")
    except Exception as e:
        print(f"  [!] {p:<15}: FAILED ({e})")

print("___JSON_START___" + json.dumps(res) + "___JSON_END___")
"""

out_proc = subprocess.run([venv_python, "-c", verify_code], capture_output=True, text=True, check=False)
stdout = out_proc.stdout
stderr = out_proc.stderr
print(stdout)
if stderr:
    print(f"Verification stderr:\n{stderr}")

# 5. Parse and Print Compact Result Block
metrics = {}
match = re.search(r"___JSON_START___(.*?)___JSON_END___", stdout)
if match:
    try:
        metrics = json.loads(match.group(1))
    except Exception:
        pass

bootstrap_ok = (bootstrap_returncode == 0) and (metrics.get("CUDA_MATMUL") == "SUCCESS") and (metrics.get("YOLO_IMPORT") == "SUCCESS")

print("\n" + "=" * 80)
print("BOOTSTRAP VALIDATION RESULT")
print("=" * 80)
print(f"BOOTSTRAP_RESULT={'SUCCESS' if bootstrap_ok else 'FAILED'}")
print(f"PYTHON_VERSION={metrics.get('PYTHON_VERSION', 'UNKNOWN')}")
print(f"TORCH_VERSION={metrics.get('TORCH_VERSION', 'UNKNOWN')}")
print(f"TORCH_CUDA={metrics.get('TORCH_CUDA', 'UNKNOWN')}")
print(f"TORCHVISION_VERSION={metrics.get('TORCHVISION_VERSION', 'UNKNOWN')}")
print(f"ULTRALYTICS_VERSION={metrics.get('ULTRALYTICS_VERSION', 'UNKNOWN')}")
print(f"GPU_NAME={metrics.get('GPU_NAME', 'UNKNOWN')}")
print(f"COMPUTE_CAPABILITY={metrics.get('COMPUTE_CAPABILITY', 'UNKNOWN')}")
print(f"CUDA_AVAILABLE={metrics.get('CUDA_AVAILABLE', 'UNKNOWN')}")
print(f"YOLO_IMPORT={metrics.get('YOLO_IMPORT', 'UNKNOWN')}")
print(f"OFFLINE_INSTALL={'SUCCESS' if bootstrap_returncode == 0 else 'FAILED'}")
print("=" * 80)
