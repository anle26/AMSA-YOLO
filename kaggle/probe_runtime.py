#!/usr/bin/env python3
"""
Kaggle Runtime Probe for AMSA-YOLO Reproducibility
===================================================
Inspects and prints hardware, OS, Python ABI, PyTorch, CUDA,
and package management state in a Kaggle notebook environment.

Guarantees:
- Strictly read-only: does not install or download anything.
- Fully offline: zero network requests.
- Fail-safe: gracefully handles missing packages or unavailable GPUs.
"""

import os
import sys
import platform
import subprocess
import sysconfig
import re

def safe_run(cmd):
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return res.stdout.strip(), res.stderr.strip(), res.returncode
    except Exception as e:
        return "", str(e), -1

def main():
    print("=" * 80)
    print(" AMSA-YOLO KAGGLE RUNTIME PROBE")
    print("=" * 80)

    # 1. SYSTEM
    print("\n[1. SYSTEM]")
    print(f"sys.version         : {sys.version.replace(chr(10), ' ')}")
    print(f"sys.executable      : {sys.executable}")
    print(f"platform.platform() : {platform.platform()}")
    print(f"platform.machine()  : {platform.machine()}")
    print(f"os.cpu_count()      : {os.cpu_count()}")

    # 2. PYTHON ABI
    print("\n[2. PYTHON ABI]")
    py_version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    soabi = sysconfig.get_config_var("SOABI")
    impl_name = getattr(sys, "implementation", None)
    impl_str = impl_name.name if impl_name else platform.python_implementation()
    print(f"Python Version      : {py_version_str}")
    print(f"SOABI               : {soabi}")
    print(f"Implementation      : {impl_str}")

    # 3. PYTORCH
    print("\n[3. PYTORCH]")
    torch_installed = False
    torch_version = "NOT INSTALLED"
    torch_cuda = "NOT INSTALLED"
    cudnn_version = "NOT INSTALLED"
    cuda_is_available = False

    try:
        import torch
        torch_installed = True
        torch_version = str(torch.__version__)
        torch_cuda = str(torch.version.cuda)
        try:
            cudnn_val = torch.backends.cudnn.version()
            cudnn_version = str(cudnn_val) if cudnn_val is not None else "None"
        except Exception as e:
            cudnn_version = f"Error: {e}"
        cuda_is_available = bool(torch.cuda.is_available())
        print(f"torch.__version__            : {torch_version}")
        print(f"torch.version.cuda           : {torch_cuda}")
        print(f"torch.backends.cudnn.version : {cudnn_version}")
        print(f"torch.cuda.is_available()   : {cuda_is_available}")
    except ImportError:
        print("torch.__version__            : NOT INSTALLED")
        print("torch.version.cuda           : NOT INSTALLED")
        print("torch.backends.cudnn.version : NOT INSTALLED")
        print("torch.cuda.is_available()   : False")

    # 4. GPU DETAILS
    print("\n[4. GPU DETAILS]")
    gpu_names = []
    compute_caps = []
    vram_gib_list = []
    
    if torch_installed and cuda_is_available:
        device_count = torch.cuda.device_count()
        print(f"torch.cuda.device_count()    : {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            cap_str = f"{cap[0]}.{cap[1]}"
            props = torch.cuda.get_device_properties(i)
            vram_gib = props.total_memory / (1024 ** 3)
            mp_count = getattr(props, "multi_processor_count", "N/A")
            
            gpu_names.append(name)
            compute_caps.append(cap_str)
            vram_gib_list.append(f"{vram_gib:.2f}")

            print(f"--- Device {i} ---")
            print(f"  Name                     : {name}")
            print(f"  Compute Capability       : {cap_str}")
            print(f"  Total VRAM (GiB)         : {vram_gib:.2f} GiB ({props.total_memory} bytes)")
            print(f"  Multiprocessor Count     : {mp_count}")
    else:
        print("CUDA acceleration is NOT available in PyTorch.")

    # 5. TORCHVISION
    print("\n[5. TORCHVISION]")
    torchvision_version = "NOT INSTALLED"
    try:
        import torchvision
        torchvision_version = str(torchvision.__version__)
        print(f"torchvision.__version__      : {torchvision_version}")
    except ImportError:
        print("torchvision.__version__      : NOT INSTALLED")

    # 6. ULTRALYTICS
    print("\n[6. ULTRALYTICS]")
    ultralytics_version = "NOT INSTALLED"
    try:
        import ultralytics
        ultralytics_version = str(ultralytics.__version__)
        print(f"ultralytics.__version__      : {ultralytics_version}")
    except ImportError:
        print("ultralytics.__version__      : NOT INSTALLED")

    # 7. NVIDIA DRIVER
    print("\n[7. NVIDIA DRIVER (via nvidia-smi)]")
    print("NOTE: Driver CUDA version indicates the maximum CUDA API supported by the host driver,")
    print("      which is distinct from the CUDA runtime bundled inside PyTorch wheels.")
    
    driver_version = "UNKNOWN"
    driver_smi_cuda = "UNKNOWN"
    
    out, err, code = safe_run(["nvidia-smi"])
    if code == 0 and out:
        driver_match = re.search(r"Driver Version:\s*([0-9.]+)", out)
        cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        if driver_match:
            driver_version = driver_match.group(1)
        if cuda_match:
            driver_smi_cuda = cuda_match.group(1)
        print(f"Driver Version               : {driver_version}")
        print(f"CUDA Version (nvidia-smi)    : {driver_smi_cuda}")
        print("\n--- Full nvidia-smi Header ---")
        lines = out.splitlines()[:15]
        print("\n".join(lines))
    else:
        print(f"nvidia-smi command returned error code {code}")
        if err:
            print(f"nvidia-smi stderr: {err}")

    # 8. PACKAGE ENVIRONMENT
    print("\n[8. PACKAGE ENVIRONMENT (Read-only check)]")
    pip_out, _, pip_code = safe_run([sys.executable, "-m", "pip", "--version"])
    if pip_code == 0:
        print(f"pip version                  : {pip_out}")
    else:
        print("pip                          : NOT AVAILABLE via sys.executable")

    uv_out, _, uv_code = safe_run(["uv", "--version"])
    if uv_code == 0:
        print(f"uv version                   : {uv_out}")
    else:
        print("uv                           : NOT INSTALLED on host")

    # 9. DECISION INPUTS
    primary_gpu_name = gpu_names[0] if gpu_names else "NONE"
    primary_cc = compute_caps[0] if compute_caps else "NONE"
    primary_vram = vram_gib_list[0] if vram_gib_list else "NONE"

    print("\n" + "=" * 80)
    print("DECISION INPUTS")
    print("=" * 80)
    print(f"PYTHON_VERSION={py_version_str}")
    print(f"PYTHON_SOABI={soabi if soabi else 'UNKNOWN'}")
    print(f"ARCH={platform.machine()}")
    print(f"TORCH_VERSION={torch_version}")
    print(f"TORCH_CUDA={torch_cuda}")
    print(f"CUDNN_VERSION={cudnn_version}")
    print(f"GPU_NAME={primary_gpu_name}")
    print(f"COMPUTE_CAPABILITY={primary_cc}")
    print(f"GPU_VRAM_GIB={primary_vram}")
    print(f"NVIDIA_DRIVER={driver_version}")
    print(f"NVIDIA_SMI_CUDA={driver_smi_cuda}")
    print("=" * 80)

if __name__ == "__main__":
    main()
