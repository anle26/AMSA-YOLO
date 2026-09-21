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
from typing import Any, Dict, List, Optional, Tuple, Union
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

# Paper-faithful reproduction training configuration (STRICTLY valid Ultralytics arguments)
# Reference: Neural Networks, Volume 197 (2026), Article 108545.
PAPER_REPRO_TRAINING_CONFIG: Dict[str, Any] = {
    # Architecture & input
    "imgsz": 640,                  # EXACT_FROM_PAPER
    "epochs": 300,                 # EXACT_FROM_PAPER
    "batch": 16,                   # EXACT_FROM_PAPER
    # Optimization
    "optimizer": "AdamW",          # EXACT_FROM_PAPER
    "lr0": 0.01,                   # EXACT_FROM_PAPER (Base YOLO initial learning rate)
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
    "box": 7.5,                    # IMPLEMENTATION_ASSUMPTION
    "cls": 0.5,                    # IMPLEMENTATION_ASSUMPTION
    "dfl": 1.5,                    # IMPLEMENTATION_ASSUMPTION
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

# Model construction parameters (valid for Model/DetectionModel instantiation, but invalid in Ultralytics training cfg/overrides)
MODEL_CONSTRUCTION_ONLY_KEYS = {
    "nc",
    "ch",
}

# Project-specific reproduction runtime keys (MUST NEVER leak into Ultralytics self.args or get_cfg)
REPRODUCTION_CUSTOM_KEYS = {
    "profile",
    "amsa_lr0",
    "scale_aware_loss",
    "initialization",
    "baseline_weights",
    "stage1_epochs",
    "stage2_epochs",
    "cumulative_budget",
    "paper_reported_epochs",
    "progressive_schedule_status",
}

# Combined set of non-training configuration keys that must be strictly sanitized from trainer overrides
NON_TRAINING_CFG_KEYS = REPRODUCTION_CUSTOM_KEYS | MODEL_CONSTRUCTION_ONLY_KEYS

# Project-specific reproduction runtime defaults
REPRODUCTION_RUNTIME_CONFIG: Dict[str, Any] = {
    "profile": "paper_repro",
    "amsa_lr0": 0.001,
    "scale_aware_loss": False,     # False for baseline, True for AMSA
    "initialization": "pretrained",
    "baseline_weights": None,
    "stage1_epochs": 300,
    "stage2_epochs": 300,
    "paper_reported_epochs": 300,
    "progressive_schedule_status": "IMPLEMENTATION_ASSUMPTION",
}


def split_training_and_reproduction_args(
    args_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Strictly split an arguments dictionary into:
    1. ultralytics_train_args: strictly containing ONLY valid Ultralytics YOLO arguments.
       (Excludes project-specific reproduction keys and model-construction parameters like 'nc').
    2. repro_config: containing project-specific reproduction runtime settings.

    Guarantees that custom project keys and model-construction-only arguments NEVER leak
    into Ultralytics DetectionTrainer / DetectionValidator.
    """
    ultralytics_train_args: Dict[str, Any] = {}
    repro_config: Dict[str, Any] = {}
    for k, v in args_dict.items():
        if k in REPRODUCTION_CUSTOM_KEYS:
            repro_config[k] = v
        elif k in MODEL_CONSTRUCTION_ONLY_KEYS:
            # Model construction parameter (e.g. 'nc'); stripped from training cfg overrides
            continue
        else:
            ultralytics_train_args[k] = v
    return ultralytics_train_args, repro_config


def get_training_and_reproduction_args(
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
    baseline_weights: Optional[Union[str, Path]] = None,
    stage1_epochs: Optional[int] = None,
    stage2_epochs: Optional[int] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate separated (ultralytics_train_args, repro_config) dictionaries.

    Guarantees ultralytics_train_args contains ZERO custom reproduction keys.
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

    u_args = deepcopy(base_cfg)
    u_args["data"] = str(data_path)
    u_args["project"] = str(project)
    u_args["name"] = name if name is not None else default_name

    if epochs is not None:
        u_args["epochs"] = int(epochs)
    if batch is not None:
        u_args["batch"] = int(batch)
    if imgsz is not None:
        u_args["imgsz"] = int(imgsz)
    if device is not None:
        u_args["device"] = device
    if workers is not None:
        u_args["workers"] = int(workers)
    if resume:
        u_args["resume"] = True

    # Setup reproduction runtime config
    repro: Dict[str, Any] = deepcopy(REPRODUCTION_RUNTIME_CONFIG)
    repro["profile"] = profile
    repro["model"] = model_type

    # Default: baseline has scale_aware_loss=False; AMSA has scale_aware_loss=True (under paper_repro)
    if scale_aware_loss is not None:
        repro["scale_aware_loss"] = bool(scale_aware_loss)
    elif profile == "paper_repro":
        repro["scale_aware_loss"] = (model_type == "amsa")
    else:
        repro["scale_aware_loss"] = False

    if initialization is not None:
        repro["initialization"] = str(initialization)
    if baseline_weights is not None:
        repro["baseline_weights"] = str(baseline_weights)
    if stage1_epochs is not None:
        repro["stage1_epochs"] = int(stage1_epochs)
    if stage2_epochs is not None:
        repro["stage2_epochs"] = int(stage2_epochs)

    # Process extra_overrides
    if extra_overrides:
        clean_extra, repro_extra = split_training_and_reproduction_args(extra_overrides)
        u_args.update(clean_extra)
        repro.update(repro_extra)

    # Final split safeguard: ensure NO custom key ever exists in ultralytics args
    clean_ultralytics_args, extracted = split_training_and_reproduction_args(u_args)
    repro.update(extracted)

    return clean_ultralytics_args, repro


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
    baseline_weights: Optional[Union[str, Path]] = None,
    stage1_epochs: Optional[int] = None,
    stage2_epochs: Optional[int] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Generate strictly valid Ultralytics YOLO training arguments.
    Guarantees no custom reproduction keys exist in the returned dictionary.
    """
    u_args, _ = get_training_and_reproduction_args(
        model_type=model_type,
        data_path=data_path,
        profile=profile,
        project=project,
        name=name,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        scale_aware_loss=scale_aware_loss,
        initialization=initialization,
        baseline_weights=baseline_weights,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        extra_overrides=extra_overrides,
        resume=resume,
    )
    return u_args


def get_reproduction_config(
    model_type: str,
    data_path: Union[str, Path] = "configs/visdrone.yaml",
    profile: str = "paper_repro",
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate reproduction runtime configuration dictionary.
    """
    _, repro = get_training_and_reproduction_args(
        model_type=model_type,
        data_path=data_path,
        profile=profile,
        **kwargs,
    )
    return repro


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
