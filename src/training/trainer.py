"""
Paper Reproduction Detection Trainer for AMSA-YOLO.

Enforces:
1. Differential Learning Rates:
   - Base YOLOv8 parameters: lr0 = 0.01 (AdamW)
   - Newly introduced AMSA parameters: amsa_lr0 = 0.001 (AdamW)
   - Weight decay: 0.0005 on weights, 0.0 on biases and normalization layers
2. Paper-faithful Scale-Aware Loss injection (ScaleAwareDetectionLoss)
3. Offline execution compliance with zero site-packages patching.
"""

from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Set
import torch
import torch.nn as nn
import torch.optim as optim

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER, colorstr

from src.amsa.amsa import AMSAModule
from src.training.config import (
    NON_TRAINING_CFG_KEYS,
    REPRODUCTION_CUSTOM_KEYS,
    REPRODUCTION_RUNTIME_CONFIG,
    split_training_and_reproduction_args,
)
from src.training.loss import ScaleAwareDetectionLoss


def identify_amsa_parameter_ids(model: nn.Module) -> Set[int]:
    """
    Collect the set of memory IDs of all parameters belonging to AMSAModule instances.

    Args:
        model: PyTorch model containing potential AMSAModule submodules.

    Returns:
        Set[int]: Set of id(param) for all AMSA parameters.
    """
    amsa_ids: Set[int] = set()
    for mod in model.modules():
        if isinstance(mod, AMSAModule):
            for param in mod.parameters():
                amsa_ids.add(id(param))
    return amsa_ids


class AMSAReproductionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer for paper-faithful AMSA-YOLO reproduction on VisDrone.

    Guarantees:
    1. Strict separation: custom reproduction parameters are NEVER inserted into self.args
       or passed to Ultralytics get_cfg/validator, preventing SyntaxError in DetectionValidator.
    2. Differential Learning Rates:
       - Base YOLOv8 parameters: lr0 = 0.01 (AdamW)
       - Newly introduced AMSA parameters: amsa_lr0 = 0.001 (AdamW)
       - Weight decay: 0.0005 on weights, 0.0 on biases and normalization layers
    3. Scale-Aware Loss injection (ScaleAwareDetectionLoss) strictly for AMSA when enabled.
    """

    default_repro_config: Dict[str, Any] = deepcopy(REPRODUCTION_RUNTIME_CONFIG)

    @classmethod
    def with_config(cls, repro_config: Optional[Dict[str, Any]] = None):
        """
        Factory creating an AMSAReproductionTrainer class with preconfigured repro_config.
        """
        class ConfiguredTrainer(cls):
            pass
        ConfiguredTrainer.default_repro_config = deepcopy(cls.default_repro_config)
        if repro_config:
            ConfiguredTrainer.default_repro_config.update(repro_config)
        return ConfiguredTrainer

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, repro_config=None):
        raw_overrides = dict(overrides) if overrides else {}
        clean_overrides, extracted_repro = split_training_and_reproduction_args(raw_overrides)

        cfg_dict = deepcopy(self.default_repro_config)
        if repro_config:
            cfg_dict.update(repro_config)
        cfg_dict.update(extracted_repro)

        # Store all custom runtime state strictly OUTSIDE self.args
        self.repro_config = cfg_dict
        self.amsa_lr0 = float(self.repro_config.get("amsa_lr0", 0.001))
        self.scale_aware_loss_enabled = bool(self.repro_config.get("scale_aware_loss", True))
        self.initialization = str(self.repro_config.get("initialization", "pretrained"))
        self.baseline_weights = self.repro_config.get("baseline_weights", None)
        self.stage1_epochs = int(self.repro_config.get("stage1_epochs", 300))
        self.stage2_epochs = int(self.repro_config.get("stage2_epochs", 300))

        if cfg is None:
            cfg = DEFAULT_CFG

        super().__init__(cfg=cfg, overrides=clean_overrides, _callbacks=_callbacks)

        # CRITICAL: Clean any custom reproduction keys or model-construction parameters from self.args
        for k in NON_TRAINING_CFG_KEYS:
            if hasattr(self.args, k):
                delattr(self.args, k)

    def get_validator(self):
        """
        Returns a DetectionValidator initialized strictly with sanitized Ultralytics args.
        Guarantees that DetectionValidator will never crash on custom reproduction keys or construction args.
        """
        clean_args = copy(self.args)
        for k in NON_TRAINING_CFG_KEYS:
            if hasattr(clean_args, k):
                delattr(clean_args, k)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=clean_args, _callbacks=self.callbacks
        )

    def build_optimizer(
        self,
        model: nn.Module,
        name: str = "auto",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ) -> optim.Optimizer:
        """
        Build optimizer with differential parameter groups for base and AMSA parameters.

        Base parameters: lr = self.args.lr0 (0.01)
        AMSA parameters: lr = self.args.amsa_lr0 (0.001)
        """
        opt_name = self.args.optimizer if self.args.optimizer else name
        base_lr = float(self.args.lr0)
        amsa_lr = float(getattr(self, "amsa_lr0", 0.001))
        weight_decay = float(self.args.weight_decay)
        momentum = float(self.args.momentum)

        # Collect AMSA parameter IDs
        amsa_ids = identify_amsa_parameter_ids(model)
        has_amsa = len(amsa_ids) > 0

        # Categorize parameters: decay, no-decay (norm), bias
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

        base_decay: List[nn.Parameter] = []
        base_norm: List[nn.Parameter] = []
        base_bias: List[nn.Parameter] = []

        amsa_decay: List[nn.Parameter] = []
        amsa_norm: List[nn.Parameter] = []
        amsa_bias: List[nn.Parameter] = []

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                is_amsa = id(param) in amsa_ids

                if "bias" in fullname:
                    if is_amsa:
                        amsa_bias.append(param)
                    else:
                        base_bias.append(param)
                elif isinstance(module, bn):
                    if is_amsa:
                        amsa_norm.append(param)
                    else:
                        base_norm.append(param)
                else:
                    if is_amsa:
                        amsa_decay.append(param)
                    else:
                        base_decay.append(param)

        param_groups: List[Dict] = []

        # 1. Base groups (lr0 = 0.01)
        if base_decay:
            param_groups.append({"params": base_decay, "lr": base_lr, "weight_decay": weight_decay, "group_name": "base_decay"})
        if base_norm:
            param_groups.append({"params": base_norm, "lr": base_lr, "weight_decay": 0.0, "group_name": "base_norm"})
        if base_bias:
            param_groups.append({"params": base_bias, "lr": base_lr, "weight_decay": 0.0, "group_name": "base_bias"})

        # 2. AMSA groups (amsa_lr0 = 0.001)
        if has_amsa:
            if amsa_decay:
                param_groups.append({"params": amsa_decay, "lr": amsa_lr, "weight_decay": weight_decay, "group_name": "amsa_decay"})
            if amsa_norm:
                param_groups.append({"params": amsa_norm, "lr": amsa_lr, "weight_decay": 0.0, "group_name": "amsa_norm"})
            if amsa_bias:
                param_groups.append({"params": amsa_bias, "lr": amsa_lr, "weight_decay": 0.0, "group_name": "amsa_bias"})

        # Instantiate optimizer
        if opt_name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, opt_name, optim.AdamW)(
                param_groups,
                lr=base_lr,
                betas=(momentum, 0.999),
            )
        elif opt_name == "SGD":
            optimizer = optim.SGD(param_groups, lr=base_lr, momentum=momentum, nesterov=True)
        elif opt_name == "RMSProp":
            optimizer = optim.RMSprop(param_groups, lr=base_lr, momentum=momentum)
        else:
            raise NotImplementedError(f"Optimizer '{opt_name}' not implemented in AMSAReproductionTrainer.")

        # Log configuration
        total_base = len(base_decay) + len(base_norm) + len(base_bias)
        total_amsa = len(amsa_decay) + len(amsa_norm) + len(amsa_bias)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__} initialized with {len(param_groups)} parameter groups:\n"
            f"  - Base YOLO: {total_base} tensors (lr={base_lr}, weight_decay={weight_decay})\n"
            f"  - AMSA:      {total_amsa} tensors (lr={amsa_lr if has_amsa else 'N/A'}, weight_decay={weight_decay})"
        )
        return optimizer

    def set_model_attributes(self):
        """Set model attributes and inject custom ScaleAwareDetectionLoss if configured."""
        super().set_model_attributes()
        use_scale_aware = getattr(self, "scale_aware_loss_enabled", False)
        if use_scale_aware:
            LOGGER.info(f"{colorstr('loss:')} Injecting paper-faithful ScaleAwareDetectionLoss (<32: 2.0, 32..95: 1.5, >=96: 1.0)")
            self.model.criterion = ScaleAwareDetectionLoss(self.model, enabled=True)
        else:
            LOGGER.info(f"{colorstr('loss:')} Using standard Ultralytics detection loss (scale_aware_loss=False)")
            self.model.criterion = ScaleAwareDetectionLoss(self.model, enabled=False)
