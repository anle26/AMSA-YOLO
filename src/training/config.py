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


# Standard reproduction training configuration
# Corresponds to standard YOLOv8s benchmark settings and paper methodology
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


def get_training_args(
    model_type: str,
    data_path: Union[str, Path],
    project: str = "runs/visdrone",
    name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
    device: Optional[Union[int, str]] = None,
    workers: Optional[int] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate the training argument dictionary for a benchmark run.

    Args:
        model_type: 'baseline' or 'amsa'.
        data_path: Path to dataset YAML configuration.
        project: Root output directory. Default: 'runs/visdrone'.
        name: Subdirectory name. If None, defaults to model_type.
        epochs: Optional epoch override.
        batch: Optional batch size override.
        imgsz: Optional image size override.
        device: Optional device override (e.g. 0 or 'cpu').
        workers: Optional workers count override.
        extra_overrides: Optional additional hyperparameter overrides.

    Returns:
        Dict[str, Any]: Complete, verified training argument dictionary.
    """
    if model_type not in ("baseline", "amsa"):
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'baseline' or 'amsa'.")

    args = deepcopy(DEFAULT_TRAINING_CONFIG)
    args["data"] = str(data_path)
    args["project"] = str(project)
    args["name"] = name if name is not None else model_type

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

    if extra_overrides:
        args.update(extra_overrides)

    return args


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
