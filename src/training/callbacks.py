"""
Callback management and external logging sanitization for AMSA-YOLO benchmark.

Ensures:
1. Ray Tune callbacks are disabled/removed to prevent compatibility crashes with Ray versions.
2. Weights & Biases (W&B) callbacks are disabled to ensure 100% offline Kaggle execution.
3. Core Ultralytics callbacks (checkpoint saving, validation, CSV logging, plots,
   and TensorBoard if available) remain completely intact.
4. Installed site-packages are never modified.
"""

from collections.abc import Mapping
import importlib
import sys
from typing import Any, Dict, List, Optional, Set


def is_raytune_or_wandb_callback(cb: Any) -> bool:
    """
    Determine whether a callback function belongs to Ray Tune or Weights & Biases.

    Args:
        cb: A callback callable.

    Returns:
        bool: True if the callback belongs to Ray Tune or W&B, False otherwise.
    """
    mod = getattr(cb, "__module__", "") or ""
    qualname = getattr(cb, "__qualname__", "") or getattr(cb, "__name__", "") or ""

    # Specifically identify Ray Tune and W&B callback modules / functions
    external_module_patterns = (
        "ultralytics.utils.callbacks.raytune",
        "ultralytics.utils.callbacks.wb",
        "ray.tune",
        "ray.air",
        "wandb",
    )
    for pattern in external_module_patterns:
        if pattern in mod:
            return True

    # Check for functions within raytune or wb module files
    if "raytune" in mod or ".wb" in mod:
        return True

    return False


def purge_external_callbacks(callbacks_dict: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    """
    Purge Ray Tune and W&B callbacks from a callbacks dictionary in-place.

    Preserves:
    - Base Ultralytics callbacks (checkpoints, metrics, validation)
    - Hub callbacks (if any)
    - TensorBoard callbacks (if installed)
    - Custom user callbacks

    Args:
        callbacks_dict: Mapping from event names (e.g. 'on_fit_epoch_end') to lists of callables.

    Returns:
        Dict[str, List[str]]: Summary of removed callbacks mapped by event name.
    """
    removed: Dict[str, List[str]] = {}
    for event, cb_list in list(callbacks_dict.items()):
        if not isinstance(cb_list, list):
            continue
        kept = []
        for cb in cb_list:
            if is_raytune_or_wandb_callback(cb):
                cb_desc = f"{getattr(cb, '__module__', 'unknown')}.{getattr(cb, '__name__', str(cb))}"
                removed.setdefault(event, []).append(cb_desc)
            else:
                kept.append(cb)
        callbacks_dict[event] = kept
    return removed


def _clean_trainer_on_pretrain_start(trainer: Any) -> None:
    """Trainer hook to ensure no external callbacks survive into the training loop."""
    if hasattr(trainer, "callbacks") and isinstance(trainer.callbacks, dict):
        purge_external_callbacks(trainer.callbacks)


def disable_external_logging_callbacks(target: Optional[Any] = None) -> None:
    """
    Disable Ray Tune and W&B callbacks cleanly for the offline benchmark pipeline.

    Multi-layer defense:
    1. Sets Ultralytics in-memory SETTINGS['raytune'] = False and SETTINGS['wandb'] = False.
    2. Clears the module-level `callbacks` dictionary in `ultralytics.utils.callbacks.raytune`
       and `ultralytics.utils.callbacks.wb` if they are loaded.
    3. If a target (YOLO model, Trainer instance, or callbacks dictionary) is provided:
       - Purges any existing Ray Tune or W&B callbacks in target.callbacks.
       - If target is a YOLO model, registers an `on_pretrain_routine_start` hook
         so any newly instantiated Trainer is sanitized before training begins.

    Does NOT modify installed site-packages files on disk.
    Does NOT touch core checkpointing, validation, CSV logging, plots, or TensorBoard callbacks.

    Args:
        target: Optional YOLO model, Trainer instance, or callbacks dict to sanitize.
    """
    # 1. Update in-memory Ultralytics SETTINGS
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS["raytune"] = False
        SETTINGS["wandb"] = False
    except Exception:
        pass

    # 2. Clear module-level callbacks dictionaries in Ultralytics integration modules
    for mod_name in ("ultralytics.utils.callbacks.raytune", "ultralytics.utils.callbacks.wb"):
        try:
            mod = sys.modules.get(mod_name)
            if mod is None:
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:
                    pass
            if mod is not None and hasattr(mod, "callbacks") and isinstance(mod.callbacks, dict):
                mod.callbacks.clear()
        except Exception:
            pass

    # 3. Sanitize target if provided
    if target is not None:
        # Case A: Dictionary of callbacks
        if isinstance(target, dict):
            purge_external_callbacks(target)

        # Case B: YOLO model or Trainer instance with .callbacks
        elif hasattr(target, "callbacks") and isinstance(target.callbacks, dict):
            purge_external_callbacks(target.callbacks)

            # If target has add_callback (e.g. YOLO instance), attach a pretrain routine hook
            if hasattr(target, "add_callback") and callable(target.add_callback):
                target.add_callback("on_pretrain_routine_start", _clean_trainer_on_pretrain_start)
