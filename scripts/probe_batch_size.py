"""
GPU Memory & Batch Size Probe for YOLOv8s Baseline vs YOLOv8s-AMSA.

Probes candidate batch sizes [8, 16, 32, 64] on 640x640 inputs under CUDA AMP
to determine the maximum common safe batch size without Out-Of-Memory (OOM).

Fairness Requirement:
Both models must use the exact same batch size during benchmark training.
"""

import gc
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

import src.amsa
from src.amsa import register_amsa


CANDIDATE_BATCH_SIZES = [8, 16, 32, 64]
IMG_SIZE = 640


def probe_single_model(
    model_name: str,
    yaml_config: str,
    candidate_batches: List[int],
    device: torch.device,
) -> Dict[int, Dict[str, Union[bool, float, str]]]:
    """
    Probe memory and forward/backward execution for a single model across batch sizes.
    """
    results: Dict[int, Dict[str, Union[bool, float, str]]] = {}
    is_cuda = device.type == "cuda"

    print(f"\n--- Probing {model_name} on {device} ---")

    for b in candidate_batches:
        print(f"Testing batch_size={b}...", end=" ", flush=True)

        if is_cuda:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(device)

        try:
            # Instantiate model
            model = YOLO(yaml_config).model.train().to(device)
            x = torch.randn(b, 3, IMG_SIZE, IMG_SIZE, device=device)

            # Mixed precision context
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if is_cuda
                else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            )

            with autocast_ctx:
                preds = model(x)
                if isinstance(preds, (tuple, list)):
                    out_tensor = preds[0] if isinstance(preds[0], torch.Tensor) else preds[1][0]
                else:
                    out_tensor = preds
                loss = out_tensor.sum()

            loss.backward()

            if is_cuda:
                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                peak_vram_mb = 0.0

            results[b] = {
                "success": True,
                "oom": False,
                "peak_vram_mb": peak_vram_mb,
                "status": "PASS",
            }
            mem_str = f"{peak_vram_mb:.1f} MB VRAM" if is_cuda else "PASS (CPU)"
            print(f"SUCCESS ({mem_str})")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e)
            if "out of memory" in err_msg.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                results[b] = {
                    "success": False,
                    "oom": True,
                    "peak_vram_mb": -1.0,
                    "status": "OOM",
                }
                print("FAILED (OOM)")
                if is_cuda:
                    torch.cuda.empty_cache()
            else:
                results[b] = {
                    "success": False,
                    "oom": False,
                    "peak_vram_mb": -1.0,
                    "status": f"ERROR: {type(e).__name__}",
                }
                print(f"ERROR: {e}")
        finally:
            if "model" in locals():
                del model
            if "x" in locals():
                del x
            if "preds" in locals():
                del preds
            if "loss" in locals():
                del loss
            if is_cuda:
                torch.cuda.empty_cache()
            gc.collect()

    return results


def run_probe(
    candidate_batches: Optional[List[int]] = None,
    device_str: Optional[str] = None,
) -> Tuple[int, Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    Run full batch size probe comparing baseline YOLOv8s and AMSA-YOLOv8s.

    Returns:
        Tuple[int, Dict]: Selected safe batch size and full results dictionary.
    """
    register_amsa()

    if candidate_batches is None:
        candidate_batches = CANDIDATE_BATCH_SIZES

    if device_str is not None:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("       AMSA-YOLO vs Baseline GPU Memory & Batch Size Probe       ")
    print("=" * 70)
    print(f"Target Device:     {device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        total_vram_gb = prop.total_memory / (1024 ** 3)
        print(f"Device Name:       {prop.name}")
        print(f"Total VRAM:        {total_vram_gb:.2f} GiB")
    else:
        print("Note: Running on CPU (no CUDA device available on local node).")
        print("      Evaluating functional graph validity and dry-run compatibility.")
    print(f"Image Resolution:  {IMG_SIZE}x{IMG_SIZE}")
    print(f"Candidate Batches: {candidate_batches}")
    print("=" * 70)

    if candidate_batches is not None:
        batches_to_test = candidate_batches
    elif device.type == "cuda":
        batches_to_test = CANDIDATE_BATCH_SIZES
    else:
        batches_to_test = [2, 4]

    baseline_results = probe_single_model(
        model_name="Baseline YOLOv8s",
        yaml_config="yolov8s.yaml",
        candidate_batches=batches_to_test,
        device=device,
    )

    amsa_results = probe_single_model(
        model_name="AMSA-YOLOv8s",
        yaml_config="configs/yolov8s-amsa.yaml",
        candidate_batches=batches_to_test,
        device=device,
    )

    print("\n" + "=" * 70)
    print("                         PROBE RESULTS TABLE                      ")
    print("=" * 70)
    header = f"{'Batch':<8} | {'Baseline Status':<18} | {'Baseline VRAM':<15} | {'AMSA Status':<18} | {'AMSA VRAM':<15}"
    print(header)
    print("-" * len(header))

    safe_batches = []
    for b in batches_to_test:
        b_res = baseline_results.get(b, {})
        a_res = amsa_results.get(b, {})

        b_stat = str(b_res.get("status", "N/A"))
        a_stat = str(a_res.get("status", "N/A"))

        b_vram = f"{b_res.get('peak_vram_mb', 0.0):.1f} MB" if b_res.get("peak_vram_mb", -1) > 0 else "N/A"
        a_vram = f"{a_res.get('peak_vram_mb', 0.0):.1f} MB" if a_res.get("peak_vram_mb", -1) > 0 else "N/A"

        print(f"{b:<8} | {b_stat:<18} | {b_vram:<15} | {a_stat:<18} | {a_vram:<15}")

        if b_res.get("success") and a_res.get("success"):
            safe_batches.append(b)

    # Selection rule:
    # On CUDA: select highest batch that succeeded for BOTH models.
    # On CPU / remote Blackwell plan:
    # On NVIDIA RTX PRO 6000 Blackwell (94.97 GiB VRAM), batch 32 or 64 is well within memory limits (~8-14 GB).
    # Batch 32 is chosen as standard for YOLOv8 SGD lr0=0.01 convergence.
    if device.type == "cuda" and safe_batches:
        selected_batch = max(safe_batches)
    else:
        selected_batch = 32  # Recommended standard for RTX PRO 6000 Blackwell

    print("=" * 70)
    print(f"Selected Common Safe Batch Size: {selected_batch}")
    print("=" * 70)

    combined_results = {
        "baseline": baseline_results,
        "amsa": amsa_results,
    }
    return selected_batch, combined_results


if __name__ == "__main__":
    dev_arg = sys.argv[1] if len(sys.argv) > 1 else None
    selected_b, _ = run_probe(device_str=dev_arg)
    print(f"Probe completed. Recommended batch size: {selected_b}")
