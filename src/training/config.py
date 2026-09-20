"""
Training Configuration and Offline Path Resolution for AMSA-YOLO Benchmark.

Guarantees:
1. Strict parameter parity between baseline YOLOv8s and YOLOv8s-AMSA.
2. 100% offline execution compliance under Kaggle Internet OFF constraints.
3. Separate output directories for baseline and AMSA runs.
"""

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.training.callbacks import disable_external_logging_callbacks


# Standard benchmark training configuration (retained for backward compatibility)
DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    # Architecture & input
    "imgsz": 640,
    "epochs": 300,
    "batch": 32,  # Probed safe size for RTX PRO 6000 Blackwell
    # Optimization
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,  # Final lr = lr0 * lrf = 0.0001
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,  # Standard linear decay
    # Precision & reproducibility
    "amp": True,
    "seed": 0,
    "deterministic": True,
    # Hardware & performance
    "workers": 8,
    "device": 0,
    # Augmentations (identical across baseline and AMSA)
    "mosaic": 1.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    # Validation & logging
    "val": True,
    "plots": True,
    "save": True,
    "save_period": 10,
    "exist_ok": True,
    "verbose": True,
}
BENCHMARK_TRAINING_CONFIG = DEFAULT_TRAINING_CONFIG

# Paper-faithful reproduction training configuration
# Reference: Neural Networks, Volume 197 (2026), Article 108545.
PAPER_REPRO_TRAINING_CONFIG: Dict[str, Any] = {
    # Architecture & input
    "imgsz": 640,                  # EXACT_FROM_PAPER
    "epochs": 300,                 # EXACT_FROM_PAPER
    "batch": 16,                   # EXACT_FROM_PAPER
    # Optimization
    "optimizer": "AdamW",          # EXACT_FROM_PAPER
    "lr0": 0.01,                   # EXACT_FROM_PAPER (Base YOLO initial learning rate)
    "amsa_lr0": 0.001,             # EXACT_FROM_PAPER (AMSA lateral modules fine-tuning learning rate)
    "lrf": 0.01,                   # IMPLEMENTATION_ASSUMPTION (Final lr = 0.01 * 0.01 = 0.0001)
    "momentum": 0.937,             # IMPLEMENTATION_ASSUMPTION (AdamW beta1; default Ultralytics)
    "weight_decay": 0.0005,        # EXACT_FROM_PAPER
    "warmup_epochs": 3.0,          # EXACT_FROM_PAPER
    "warmup_momentum": 0.8,        # IMPLEMENTATION_ASSUMPTION
    "warmup_bias_lr": 0.0,         # IMPLEMENTATION_ASSUMPTION (Standard for AdamW)
    "cos_lr": True,                # EXACT_FROM_PAPER (Cosine annealing learning rate schedule)
    # Augmentations
    "mosaic": 0.5,                 # EXACT_FROM_PAPER
    "mixup": 0.1,                  # EXACT_FROM_PAPER
    "fliplr": 0.5,                 # EXACT_FROM_PAPER
    "scale": 0.5,                  # ULTRALYTICS_APPROXIMATION (Random scaling range [1 - 0.5, 1 + 0.5] = [0.5, 1.5])
    "hsv_h": 0.015,                # IMPLEMENTATION_ASSUMPTION (Standard YOLO HSV augmentation enabled)
    "hsv_s": 0.7,                  # IMPLEMENTATION_ASSUMPTION
    "hsv_v": 0.4,                  # IMPLEMENTATION_ASSUMPTION
    "degrees": 0.0,                # IMPLEMENTATION_ASSUMPTION
    "translate": 0.1,              # IMPLEMENTATION_ASSUMPTION
    "shear": 0.0,                  # IMPLEMENTATION_ASSUMPTION
    "perspective": 0.0,            # IMPLEMENTATION_ASSUMPTION
    "flipud": 0.0,                 # IMPLEMENTATION_ASSUMPTION
    # Loss Configuration
    "scale_aware_loss": True,      # EXACT_FROM_PAPER / ULTRALYTICS_APPROXIMATION
    "box": 7.5,                    # IMPLEMENTATION_ASSUMPTION
    "cls": 0.5,                    # IMPLEMENTATION_ASSUMPTION
    "dfl": 1.5,                    # IMPLEMENTATION_ASSUMPTION
    # Strategy & Initialization (IMPLEMENTATION_ASSUMPTION: stage boundary is not disclosed in paper)
    "stage1_epochs": 300,          # IMPLEMENTATION_ASSUMPTION (Stage 1 baseline training epochs)
    "stage2_epochs": 300,          # IMPLEMENTATION_ASSUMPTION (Stage 2/3 AMSA fine-tuning epochs)
    "initialization": "pretrained",# IMPLEMENTATION_ASSUMPTION (Configurable: "pretrained" or "scratch")
    # Precision & Execution
    "amp": True,                   # IMPLEMENTATION_ASSUMPTION
    "seed": 0,                     # IMPLEMENTATION_ASSUMPTION
    "deterministic": True,         # IMPLEMENTATION_ASSUMPTION
    "workers": 8,                  # IMPLEMENTATION_ASSUMPTION
    "device": 0,                   # IMPLEMENTATION_ASSUMPTION
    "val": True,                   # IMPLEMENTATION_ASSUMPTION
    "plots": True,                 # IMPLEMENTATION_ASSUMPTION
    "save": True,                  # IMPLEMENTATION_ASSUMPTION
    "save_period": 10,             # IMPLEMENTATION_ASSUMPTION
    "exist_ok": True,              # IMPLEMENTATION_ASSUMPTION
    "verbose": True,               # IMPLEMENTATION_ASSUMPTION
}


def get_training_args(
    model_type: str,
    data_path: Union[str, Path],
    profile: str = "benchmark",
    project: str = "runs/visdrone",
    name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
    device: Optional[Union[int, str]] = None,
    workers: Optional[int] = None,
    scale_aware_loss: Optional[bool] = None,
    initialization: Optional[str] = None,
    stage1_epochs: Optional[int] = None,
    stage2_epochs: Optional[int] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Generate the training argument dictionary for a benchmark or paper reproduction run.

    Args:
        model_type: 'baseline' or 'amsa'.
        data_path: Path to dataset YAML configuration.
        profile: 'benchmark' (default) or 'paper_repro'.
        project: Root output directory. Default: 'runs/visdrone'.
        name: Subdirectory name. Defaults to model_type (or 'paper_<model>' for paper_repro).
        epochs: Optional epoch override.
        batch: Optional batch size override.
        imgsz: Optional image size override.
        device: Optional device override (e.g. 0 or 'cpu').
        workers: Optional workers count override.
        scale_aware_loss: Optional override for scale-aware loss.
        initialization: Optional override ('pretrained' or 'scratch').
        extra_overrides: Optional additional hyperparameter overrides.
        resume: Whether to resume training from an existing checkpoint.

    Returns:
        Dict[str, Any]: Complete, verified training argument dictionary.
    """
    if model_type not in ("baseline", "amsa"):
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'baseline' or 'amsa'.")

    if profile == "benchmark":
        base_cfg = DEFAULT_TRAINING_CONFIG
        default_name = model_type
    elif profile == "paper_repro":
        base_cfg = PAPER_REPRO_TRAINING_CONFIG
        default_name = f"paper_{model_type}"
    else:
        raise ValueError(f"Invalid profile '{profile}'. Must be 'benchmark' or 'paper_repro'.")

    args = deepcopy(base_cfg)
    args["data"] = str(data_path)
    args["project"] = str(project)
    args["name"] = name if name is not None else default_name
    args["profile"] = profile

    if epochs is not None:
        args["epochs"] = int(epochs)
    if batch is not None:
        args["batch"] = int(batch)
    if imgsz is not None:
        args["imgsz"] = int(imgsz)
    if device is not None:
        args["device"] = device
    if workers is not None:
        args["workers"] = int(workers)
    if scale_aware_loss is not None:
        args["scale_aware_loss"] = bool(scale_aware_loss)
    if initialization is not None:
        args["initialization"] = str(initialization)
    if stage1_epochs is not None:
        args["stage1_epochs"] = int(stage1_epochs)
    if stage2_epochs is not None:
        args["stage2_epochs"] = int(stage2_epochs)

    if extra_overrides:
        args.update(extra_overrides)

    if resume:
        args["resume"] = True

    return args


def resolve_resume_checkpoint(
    model_type: str,
    resume: bool = False,
    resume_from: Optional[Union[str, Path]] = None,
    project: Union[str, Path] = "runs/visdrone",
    name: Optional[str] = None,
    profile: str = "benchmark",
) -> Optional[Path]:
    """
    Resolve and validate checkpoint path for training resumption.

    Args:
        model_type: 'baseline' or 'amsa'.
        resume: If True, automatically looks for <project>/<name>/weights/last.pt.
        resume_from: Explicit path to checkpoint file.
        project: Root output directory (default: 'runs/visdrone').
        name: Experiment run name (defaults to model_type or 'paper_<model>').
        profile: Training profile ('benchmark' or 'paper_repro').

    Returns:
        Optional[Path]: Resolved Path to checkpoint file, or None if fresh run.

    Raises:
        FileNotFoundError: If resume is requested but checkpoint file does not exist.
        ValueError: If model_type is invalid.
    """
    if model_type not in ("baseline", "amsa"):
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'baseline' or 'amsa'.")

    if not resume and not resume_from:
        return None

    if resume_from:
        explicit_path = Path(resume_from).resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(
                f"Explicit resume checkpoint does not exist: {resume_from}"
            )
        return explicit_path

    # Automatic resume path: <project>/<name>/weights/last.pt
    default_name = f"paper_{model_type}" if profile == "paper_repro" else model_type
    run_name = name if name is not None else default_name
    auto_path = (Path(project) / run_name / "weights" / "last.pt").resolve()
    if not auto_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume {model_type} training: checkpoint not found at '{auto_path}'.\n"
            f"Ensure a previous training run generated 'weights/last.pt' or specify an explicit path with --resume-from."
        )
    return auto_path


def find_offline_file(
    filename: str,
    explicit_path: Optional[Union[str, Path]] = None,
    search_dirs: Optional[List[Union[str, Path]]] = None,
) -> Path:
    """
    Locate a file locally or within mounted Kaggle inputs without network calls.

    Raises:
        FileNotFoundError: If the file cannot be located on the local filesystem.
    """
    if explicit_path:
        p = Path(explicit_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"Explicitly specified file does not exist: {explicit_path}"
        )

    candidates: List[Path] = [
        Path.cwd() / filename,
        Path.cwd() / "weights" / filename,
        Path.cwd() / "configs" / filename,
    ]

    if search_dirs:
        for d in search_dirs:
            d_path = Path(d)
            if d_path.exists():
                candidates.extend(list(d_path.glob(f"**/{filename}")))

    # Scan standard Kaggle input directory
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(list(kaggle_input.glob(f"**/{filename}")))

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not locate required offline file '{filename}'.\n"
        f"To operate under Internet OFF constraints, stage '{filename}' locally or pass its path via CLI.\n"
        f"Searched candidates:\n" + "\n".join(f"  - {c}" for c in candidates[:10])
    )


def resolve_visdrone_dataset(
    config_path: Union[str, Path] = "configs/visdrone.yaml",
    custom_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve and validate the VisDrone dataset path.

    If the path specified in config_path does not exist on disk, attempts to
    auto-discover mounted Kaggle VisDrone directories or environment overrides.

    Returns:
        Path: Path to a valid, resolved dataset YAML.
    """
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Dataset configuration YAML does not exist: {config_path}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    configured_root = Path(data.get("path", ""))

    # 1. Custom root takes precedence
    candidates: List[Path] = []
    if custom_root:
        candidates.append(Path(custom_root).resolve())

    # 2. Environment variable VISDRONE_ROOT
    if "VISDRONE_ROOT" in os.environ:
        candidates.append(Path(os.environ["VISDRONE_ROOT"]).resolve())

    # 3. Configured root in YAML
    if configured_root and configured_root.exists():
        return cfg_file

    # 4. Standard Kaggle mounted paths
    candidates.extend([
        Path("/kaggle/input/visdrone-2019-yolo/VisDrone_2019/VisDrone"),
        Path("/kaggle/input/visdrone-2019-yolo"),
        Path.cwd() / "datasets" / "VisDrone",
    ])

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sub in kaggle_input.iterdir():
            if "visdrone" in sub.name.lower():
                candidates.append(sub)
                candidates.append(sub / "VisDrone_2019" / "VisDrone")

    for cand in candidates:
        if (cand / "images" / "val").exists() or (cand / "VisDrone" / "images" / "val").exists():
            resolved_root = cand if (cand / "images" / "val").exists() else (cand / "VisDrone")
            # If resolved path differs from YAML, create a temporary resolved YAML
            if resolved_root != configured_root:
                resolved_data = deepcopy(data)
                resolved_data["path"] = str(resolved_root.resolve())
                resolved_yaml_path = Path.cwd() / ".resolved_visdrone.yaml"
                with open(resolved_yaml_path, "w", encoding="utf-8") as rf:
                    yaml.dump(resolved_data, rf, default_flow_style=False)
                return resolved_yaml_path
            return cfg_file

    return cfg_file
