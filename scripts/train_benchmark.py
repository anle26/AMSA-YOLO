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
from typing import Any, Dict, Optional, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

from src.amsa import register_amsa
from src.amsa.pretrained import transfer_yolov8s_weights
from src.training import (
    AMSAReproductionTrainer,
    disable_external_logging_callbacks,
    find_offline_file,
    get_training_args,
    resolve_resume_checkpoint,
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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from previous run checkpoint (runs/visdrone/<model>/weights/last.pt).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Explicit path to checkpoint file (e.g. /path/to/last.pt) to resume from.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="benchmark",
        choices=["benchmark", "paper_repro"],
        help="Training profile: 'benchmark' (default benchmark parity) or 'paper_repro' (paper-faithful reproduction).",
    )
    parser.add_argument(
        "--scale-aware-loss",
        action="store_true",
        default=None,
        help="Explicitly enable paper scale-aware detection loss.",
    )
    parser.add_argument(
        "--no-scale-aware-loss",
        action="store_true",
        default=False,
        help="Explicitly disable paper scale-aware detection loss and use standard YOLO loss.",
    )
    parser.add_argument(
        "--initialization",
        type=str,
        default=None,
        choices=["pretrained", "scratch"],
        help="Weight initialization policy: 'pretrained' (default) or 'scratch'.",
    )
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default=None,
        help="Path to trained baseline weights for progressive Stage 2/3 AMSA training.",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=None,
        help="Stage 1 baseline training epochs for progressive schedule (default: 300).",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=None,
        help="Stage 2/3 AMSA fine-tuning epochs for progressive schedule (default: 300).",
    )
    return parser.parse_args()


def setup_model(
    model_type: str,
    weights_path: Optional[str] = None,
    allow_untrained_fallback: bool = False,
    resume_checkpoint: Optional[Union[str, Path]] = None,
    baseline_weights: Optional[Union[str, Path]] = None,
    initialization: str = "pretrained",
) -> YOLO:
    """
    Construct model and initialize weights according to benchmark or paper reproduction protocol.

    Fresh Baseline:
        Constructed from yolov8s.yaml and loads stock pretrained weights (if initialization='pretrained').
    Fresh AMSA:
        Constructed from configs/yolov8s-amsa.yaml and transfers weights via transfer_yolov8s_weights():
        - From trained baseline checkpoint if baseline_weights is provided (progressive training)
        - From stock yolov8s.pt if baseline_weights is None
        AMSA lateral modules remain fresh in both cases.
    Resume Run (Baseline or AMSA):
        Loaded directly from verified checkpoint (e.g. weights/last.pt) preserving
        all weights, optimizer state, scheduler state, and training epoch counter.
    """
    register_amsa()
    disable_external_logging_callbacks()

    if resume_checkpoint is not None:
        resolved_ckpt = Path(resume_checkpoint).resolve()
        if not resolved_ckpt.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resolved_ckpt}")
        print(f"\n[Setup] Resuming {model_type.upper()} model directly from checkpoint: {resolved_ckpt}")
        model = YOLO(str(resolved_ckpt))
        model.ckpt_path = str(resolved_ckpt)
        disable_external_logging_callbacks(model)
        return model

    if initialization == "scratch":
        print(f"\n[Setup] Initializing {model_type.upper()} from scratch (random initialization)...")
        model = YOLO("yolov8s.yaml" if model_type == "baseline" else "configs/yolov8s-amsa.yaml")
        model.ckpt = {"model": model.model}
        disable_external_logging_callbacks(model)
        return model

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
            if not allow_untrained_fallback and not baseline_weights:
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
        model.ckpt = {"model": model.model}

    elif model_type == "amsa":
        print("\n[Setup] Initializing AMSA-YOLOv8s Model...")
        model = YOLO("configs/yolov8s-amsa.yaml")
        if baseline_weights is not None:
            b_path = Path(baseline_weights).resolve()
            if not b_path.is_file():
                raise FileNotFoundError(f"Specified baseline weights do not exist: {baseline_weights}")
            print(f"[Setup] Progressive Training: Transferring trained baseline weights into AMSA from: {b_path}")
            ckpt = torch.load(b_path, map_location="cpu")
            report = transfer_yolov8s_weights(model, ckpt, strict_dtype=False)
            print(report.summary())
            assert report.total_transferred > 0, "Progressive weight transfer failed: zero keys transferred."
            print(f"[Setup] Transferred {report.total_transferred} trained baseline keys into AMSA-YOLOv8s.")
            print(f"[Setup] Protected {len(report.amsa_keys_left_fresh)} fresh AMSA parameters for fine-tuning.")
        elif resolved_weights is not None:
            print(f"[Setup] Transferring pretrained weights into AMSA from: {resolved_weights}")
            ckpt = torch.load(resolved_weights, map_location="cpu")
            report = transfer_yolov8s_weights(model, ckpt, strict_dtype=False)
            print(report.summary())
            assert report.total_transferred > 0, "Weight transfer failed: zero keys transferred."
            print(f"[Setup] Transferred {report.total_transferred} stock keys into AMSA-YOLOv8s.")
            print(f"[Setup] Protected {len(report.amsa_keys_left_fresh)} fresh AMSA parameters.")
        else:
            # When testing offline without yolov8s.pt binary, transfer from an offline stock YOLO model
            print("[Setup] WARNING: No external yolov8s.pt or baseline checkpoint provided.")
            print("[Setup] Generating stock state_dict from offline yolov8s.yaml to verify transfer pipeline...")
            stock_ref = YOLO("yolov8s.yaml")
            report = transfer_yolov8s_weights(model, stock_ref.model.state_dict(), strict_dtype=False)
            print(report.summary())
            print(f"[Setup] Transferred {report.total_transferred} keys from reference stock architecture.")
        model.ckpt = {"model": model.model}

    disable_external_logging_callbacks(model)
    return model


def run_benchmark_training(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute benchmark or paper reproduction training run."""
    profile_label = "PAPER REPRODUCTION" if getattr(args, "profile", "benchmark") == "paper_repro" else "BENCHMARK"
    print("=" * 70)
    print(f"       STARTING AMSA-YOLO {profile_label} RUN: {args.model.upper()}       ")
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

    # Determine run name and check for resume checkpoint
    profile = getattr(args, "profile", "benchmark")
    if args.name:
        run_name = args.name
    elif args.smoke:
        run_name = f"smoke_paper_{args.model}" if profile == "paper_repro" else f"smoke_{args.model}"
    elif profile == "paper_repro":
        run_name = f"paper_{args.model}"
    else:
        run_name = args.model

    resume_checkpoint = resolve_resume_checkpoint(
        model_type=args.model,
        resume=args.resume,
        resume_from=args.resume_from,
        project=args.project,
        name=run_name,
        profile=profile,
    )
    is_resuming = resume_checkpoint is not None
    if is_resuming:
        print(f"[Resume] Resuming training from checkpoint: {resume_checkpoint}")

    # Resolve progressive baseline weights if in paper reproduction mode for AMSA
    baseline_weights = getattr(args, "baseline_weights", None)
    if profile == "paper_repro" and args.model == "amsa" and not is_resuming and not baseline_weights:
        candidate = (Path(args.project) / "paper_baseline" / "weights" / "best.pt").resolve()
        if candidate.is_file():
            baseline_weights = str(candidate)
            print(f"[Progressive] Auto-discovered Stage 1 baseline weights at: {baseline_weights}")

    # Resolve scale aware loss flag
    scale_aware = None
    if getattr(args, "scale_aware_loss", None):
        scale_aware = True
    elif getattr(args, "no_scale_aware_loss", False):
        scale_aware = False

    initialization = getattr(args, "initialization", None) or "pretrained"

    # Build model and setup initialization
    allow_fallback = args.smoke or args.dry_run
    model = setup_model(
        model_type=args.model,
        weights_path=args.weights,
        allow_untrained_fallback=allow_fallback,
        resume_checkpoint=resume_checkpoint,
        baseline_weights=baseline_weights,
        initialization=initialization,
    )

    # Calculate model complexity
    param_count = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"[Architecture] Total Parameters:     {param_count:,} ({param_count / 1e6:.2f} M)")
    print(f"[Architecture] Trainable Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")

    # Progressive stage schedules (IMPLEMENTATION_ASSUMPTION: paper does not disclose stage boundary)
    stage1_ep = getattr(args, "stage1_epochs", None) or 300
    stage2_ep = getattr(args, "stage2_epochs", None) or 300

    # Determine epochs for this execution
    if args.epochs is not None:
        epochs = int(args.epochs)
    elif args.smoke:
        epochs = 1
    elif profile == "paper_repro":
        epochs = stage1_ep if args.model == "baseline" else stage2_ep
    else:
        epochs = 300
    batch = 2 if (args.smoke and device == "cpu") else (4 if args.smoke else args.batch)

    extra_overrides: Dict[str, Any] = {}
    if args.fraction is not None:
        extra_overrides["fraction"] = args.fraction
    elif args.smoke and "smoke" not in str(data_yaml):
        extra_overrides["fraction"] = 0.01  # Use 1% of full data for fast smoke validation

    train_args = get_training_args(
        model_type=args.model,
        data_path=data_yaml,
        profile=profile,
        project=args.project,
        name=run_name,
        epochs=epochs,
        batch=batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        scale_aware_loss=scale_aware,
        initialization=initialization,
        stage1_epochs=stage1_ep,
        stage2_epochs=stage2_ep,
        extra_overrides=extra_overrides,
        resume=is_resuming,
    )

    # Runtime progressive schedule and optimization audit log
    init_source = (
        str(baseline_weights)
        if baseline_weights
        else (str(args.weights) if args.weights else ("scratch" if initialization == "scratch" else "stock yolov8s.pt"))
    )
    base_lr = train_args.get("lr0", 0.01)
    amsa_lr = train_args.get("amsa_lr0", 0.001) if (args.model == "amsa" and profile == "paper_repro") else "N/A"
    paper_reported_epochs = 300
    cumulative_budget = stage1_ep + stage2_ep

    print("\n" + "=" * 70)
    print("       PROGRESSIVE SCHEDULE & OPTIMIZATION AUDIT       ")
    print("=" * 70)
    print(f"  - Current Run Target Epochs:                 {epochs}")
    print(f"  - Paper-reported epochs:                     {paper_reported_epochs}")
    print(f"  - Baseline stage epochs:                     {stage1_ep}")
    print(f"  - AMSA fine-tuning epochs:                   {stage2_ep}")
    print(f"  - Cumulative implementation training budget: {cumulative_budget}")
    print(f"  - Progressive schedule status:               IMPLEMENTATION_ASSUMPTION")
    print(f"  - Initialization Checkpoint:                 {init_source}")
    print(f"  - Base LR:                                   {base_lr}")
    print(f"  - AMSA LR:                                   {amsa_lr}")
    print("=" * 70 + "\n")

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
            "profile": profile,
            "params": param_count,
            "train_args": train_args,
            "resumed": is_resuming,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "stage1_epochs": stage1_ep,
            "stage2_epochs": stage2_ep,
            "paper_reported_epochs": paper_reported_epochs,
            "cumulative_budget": cumulative_budget,
            "progressive_schedule_status": "IMPLEMENTATION_ASSUMPTION",
        }

    # Sanitize callbacks before training execution
    disable_external_logging_callbacks(model)

    # Determine trainer class
    if profile == "paper_repro" or train_args.get("scale_aware_loss", False) or (args.model == "amsa" and "amsa_lr0" in train_args):
        trainer_cls = AMSAReproductionTrainer
        print(f"[Trainer] Using AMSAReproductionTrainer (differential LR & scale-aware loss enabled).")
    else:
        trainer_cls = None
        print(f"[Trainer] Using standard Ultralytics DetectionTrainer.")

    # Execute training
    print(f"\n[Training] Launching {args.model.upper()} training ({train_args['epochs']} epochs)...")
    if trainer_cls is not None:
        train_results = model.train(trainer=trainer_cls, **train_args)
    else:
        train_results = model.train(**train_args)
    elapsed = time.time() - start_time

    # Collect metrics
    metrics_summary: Dict[str, Any] = {
        "model": args.model,
        "profile": profile,
        "timestamp": datetime.now().isoformat(),
        "total_params": param_count,
        "elapsed_seconds": round(elapsed, 2),
        "output_dir": str(output_dir.resolve()),
    }
    if torch.cuda.is_available():
        metrics_summary["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    if hasattr(train_results, "fitness"):
        metrics_summary["best_fitness"] = float(train_results.fitness)

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
