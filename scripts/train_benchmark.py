"""
Unified Training Benchmark Runner for YOLOv8s Baseline vs YOLOv8s-AMSA.

Enforces:
1. Strict parameter and hyperparameter parity between models.
2. Separate output directories: runs/visdrone/baseline/ vs runs/visdrone/amsa/.
3. 100% offline execution compliance under Kaggle Internet OFF constraints.
4. Pretrained weight transfer for AMSA via transfer_yolov8s_weights().
5. Comprehensive metrics reporting (mAP50, mAP50-95, precision, recall, params, runtime).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

from src.amsa import register_amsa
from src.amsa.pretrained import transfer_yolov8s_weights
from src.training import (
    disable_external_logging_callbacks,
    find_offline_file,
    get_training_args,
    resolve_visdrone_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fair training benchmark for YOLOv8s baseline vs YOLOv8s-AMSA on VisDrone."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["baseline", "amsa"],
        help="Model architecture to train: 'baseline' (YOLOv8s) or 'amsa' (YOLOv8s-AMSA).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to stock yolov8s.pt weights checkpoint. Required for offline pretrained initialization.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/visdrone.yaml",
        help="Path to dataset configuration YAML. Default: configs/visdrone.yaml.",
    )
    parser.add_argument(
        "--custom-data-root",
        type=str,
        default=None,
        help="Optional override for VisDrone dataset root directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total training epochs (default: 300 for full, 1 for smoke).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution. Default: 640.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g. 0, 'cuda:0', 'cpu'). If omitted, auto-selects.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers count. Default: 8.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/visdrone",
        help="Root output directory. Default: runs/visdrone.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (defaults to model type, e.g. 'baseline' or 'amsa').",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1-epoch smoke test with small batch to verify pipeline validity.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of dataset to use (e.g. 0.01 for fast smoke testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build models and verify weight initialization without starting training.",
    )
    return parser.parse_args()


def setup_model(
    model_type: str,
    weights_path: Optional[str] = None,
    allow_untrained_fallback: bool = False,
) -> YOLO:
    """
    Construct model and initialize weights according to fair benchmark protocol.

    Baseline:
        Constructed from yolov8s.yaml and loads stock pretrained weights.
    AMSA:
        Constructed from configs/yolov8s-amsa.yaml and transfers stock pretrained
        weights via transfer_yolov8s_weights(), leaving AMSA lateral modules fresh.
    """
    register_amsa()
    disable_external_logging_callbacks()

    # Attempt to locate offline weights file
    resolved_weights: Optional[Path] = None
    if weights_path and weights_path.lower() != "none":
        resolved_weights = find_offline_file(
            "yolov8s.pt",
            explicit_path=weights_path,
        )
    else:
        try:
            resolved_weights = find_offline_file("yolov8s.pt")
        except FileNotFoundError:
            if not allow_untrained_fallback:
                raise

    if model_type == "baseline":
        print("\n[Setup] Initializing Baseline YOLOv8s Model...")
        model = YOLO("yolov8s.yaml")
        if resolved_weights is not None:
            print(f"[Setup] Loading pretrained weights into Baseline from: {resolved_weights}")
            ckpt = torch.load(resolved_weights, map_location="cpu")
            weights_sd = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.model.load_state_dict(weights_sd, strict=False)
            print("[Setup] Baseline model loaded with stock pretrained weights.")
        else:
            print("[Setup] WARNING: No pretrained weights provided. Baseline initialized with random weights.")

    elif model_type == "amsa":
        print("\n[Setup] Initializing AMSA-YOLOv8s Model...")
        model = YOLO("configs/yolov8s-amsa.yaml")
        if resolved_weights is not None:
            print(f"[Setup] Transferring pretrained weights into AMSA from: {resolved_weights}")
            ckpt = torch.load(resolved_weights, map_location="cpu")
            report = transfer_yolov8s_weights(model, ckpt, strict_dtype=False)
            print(report.summary())
            assert report.total_transferred > 0, "Weight transfer failed: zero keys transferred."
            print(f"[Setup] Transferred {report.total_transferred} stock keys into AMSA-YOLOv8s.")
            print(f"[Setup] Protected {len(report.amsa_keys_left_fresh)} fresh AMSA parameters.")
        else:
            # When testing offline without yolov8s.pt binary, transfer from an offline stock YOLO model
            print("[Setup] WARNING: No external yolov8s.pt provided.")
            print("[Setup] Generating stock state_dict from offline yolov8s.yaml to verify transfer pipeline...")
            stock_ref = YOLO("yolov8s.yaml")
            report = transfer_yolov8s_weights(model, stock_ref.model.state_dict(), strict_dtype=False)
            print(report.summary())
            print(f"[Setup] Transferred {report.total_transferred} keys from reference stock architecture.")

    disable_external_logging_callbacks(model)
    return model


def run_benchmark_training(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute benchmark training run."""
    print("=" * 70)
    print(f"       STARTING AMSA-YOLO BENCHMARK RUN: {args.model.upper()}       ")
    print("=" * 70)
    start_time = time.time()

    # Determine device
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    # Resolve dataset configuration
    data_yaml = resolve_visdrone_dataset(
        config_path=args.data,
        custom_root=args.custom_data_root,
    )
    print(f"[Config] Resolved Dataset YAML: {data_yaml}")

    # Build model and setup initialization
    allow_fallback = args.smoke or args.dry_run
    model = setup_model(
        model_type=args.model,
        weights_path=args.weights,
        allow_untrained_fallback=allow_fallback,
    )

    # Calculate model complexity
    param_count = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"[Architecture] Total Parameters:     {param_count:,} ({param_count / 1e6:.2f} M)")
    print(f"[Architecture] Trainable Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")

    # Prepare training arguments
    epochs = 1 if args.smoke else args.epochs
    batch = 2 if (args.smoke and device == "cpu") else (4 if args.smoke else args.batch)
    run_name = args.name if args.name else (f"smoke_{args.model}" if args.smoke else args.model)

    extra_overrides: Dict[str, Any] = {}
    if args.fraction is not None:
        extra_overrides["fraction"] = args.fraction
    elif args.smoke and "smoke" not in str(data_yaml):
        extra_overrides["fraction"] = 0.01  # Use 1% of full data for fast smoke validation

    train_args = get_training_args(
        model_type=args.model,
        data_path=data_yaml,
        project=args.project,
        name=run_name,
        epochs=epochs,
        batch=batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        extra_overrides=extra_overrides,
    )

    output_dir = Path(args.project) / run_name
    print(f"[Output] Dedicated Output Directory: {output_dir}")
    print("[Config] Final Training Hyperparameters:")
    for k in sorted(train_args.keys()):
        print(f"  - {k}: {train_args[k]}")

    if args.dry_run:
        print("\n[Dry-Run] Completed dry-run validation. Model and configs verified cleanly.")
        return {
            "status": "DRY_RUN_PASS",
            "model": args.model,
            "params": param_count,
            "train_args": train_args,
        }

    # Sanitize callbacks before training execution
    disable_external_logging_callbacks(model)

    # Execute training
    print(f"\n[Training] Launching {args.model.upper()} training ({train_args['epochs']} epochs)...")
    train_results = model.train(**train_args)
    elapsed = time.time() - start_time

    # Collect metrics
    metrics_summary: Dict[str, Any] = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "total_params": param_count,
        "elapsed_seconds": round(elapsed, 2),
        "output_dir": str(output_dir.resolve()),
    }

    if hasattr(train_results, "results_dict") and isinstance(train_results.results_dict, dict):
        rd = train_results.results_dict
        metrics_summary.update({
            "mAP50": rd.get("metrics/mAP50(B)", 0.0),
            "mAP50_95": rd.get("metrics/mAP50-95(B)", 0.0),
            "precision": rd.get("metrics/precision(B)", 0.0),
            "recall": rd.get("metrics/recall(B)", 0.0),
        })

    metrics_file = output_dir / "benchmark_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"       BENCHMARK RUN COMPLETED: {args.model.upper()}       ")
    print("=" * 70)
    print(f"Total Time:      {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print(f"Metrics File:    {metrics_file}")
    for k, v in metrics_summary.items():
        if k not in ("output_dir", "timestamp"):
            print(f"  - {k}: {v}")
    print("=" * 70)

    return metrics_summary


if __name__ == "__main__":
    args = parse_args()
    run_benchmark_training(args)
