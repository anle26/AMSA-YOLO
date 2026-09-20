"""
Unit and Integration Tests for Baseline vs AMSA Training Benchmark Pipeline.

Validates:
1. Strict hyperparameter equality between baseline and AMSA models.
2. Pretrained initialization parity and clean transfer report.
3. Dedicated output directory separation (runs/visdrone/baseline vs runs/visdrone/amsa).
4. Offline path enforcement and error raising (no network downloads).
5. Dataset YAML specification compliance (10 VisDrone classes, correct paths).
6. Batch size probe script functionality.
7. Model parameter counts matching architectural specifications.
"""

from pathlib import Path
import pytest
import torch
from ultralytics import YOLO
import yaml

from scripts.probe_batch_size import run_probe
from scripts.train_benchmark import setup_model
from src.amsa import register_amsa
from src.amsa.pretrained import transfer_yolov8s_weights
from src.training import (
    DEFAULT_TRAINING_CONFIG,
    disable_external_logging_callbacks,
    find_offline_file,
    get_training_args,
    is_raytune_or_wandb_callback,
    purge_external_callbacks,
    resolve_resume_checkpoint,
    resolve_visdrone_dataset,
)
from ultralytics.utils import SETTINGS
import ultralytics.utils.callbacks.base as base_cb
import ultralytics.utils.callbacks.raytune as raytune_cb
import ultralytics.utils.callbacks.wb as wb_cb


class TestBenchmarkPipeline:
    """Test suite for training benchmark parity and offline validation."""

    def test_config_hyperparameter_equality(self) -> None:
        """
        Verify that baseline and AMSA training configurations have 100%
        identical hyperparameters with only model-specific run names differing.
        """
        data_path = "configs/visdrone.yaml"
        baseline_args = get_training_args("baseline", data_path=data_path)
        amsa_args = get_training_args("amsa", data_path=data_path)

        assert baseline_args["name"] == "baseline"
        assert amsa_args["name"] == "amsa"
        assert baseline_args["project"] == amsa_args["project"]
        assert baseline_args["data"] == amsa_args["data"]

        # Assert all hyperparameter keys are identical
        shared_keys = set(baseline_args.keys()) - {"name"}
        for k in shared_keys:
            assert baseline_args[k] == amsa_args[k], f"Hyperparameter mismatch for '{k}': {baseline_args[k]} != {amsa_args[k]}"

        # Explicit checks for key hyperparameters
        assert baseline_args["imgsz"] == 640
        assert baseline_args["epochs"] == 300
        assert baseline_args["optimizer"] == "SGD"
        assert baseline_args["lr0"] == 0.01
        assert baseline_args["lrf"] == 0.01
        assert baseline_args["momentum"] == 0.937
        assert baseline_args["weight_decay"] == 0.0005
        assert baseline_args["warmup_epochs"] == 3.0
        assert baseline_args["cos_lr"] is False
        assert baseline_args["amp"] is True
        assert baseline_args["seed"] == 0
        assert baseline_args["deterministic"] is True

    def test_output_directory_separation(self) -> None:
        """Verify output directories for baseline and AMSA are disjoint."""
        b_args = get_training_args("baseline", "configs/visdrone.yaml")
        a_args = get_training_args("amsa", "configs/visdrone.yaml")

        b_dir = Path(b_args["project"]) / b_args["name"]
        a_dir = Path(a_args["project"]) / a_args["name"]

        assert b_dir != a_dir
        assert b_dir.name == "baseline"
        assert a_dir.name == "amsa"

    def test_visdrone_dataset_yaml_schema(self) -> None:
        """Verify configs/visdrone.yaml follows official VisDrone 10-class specification."""
        yaml_path = Path("configs/visdrone.yaml")
        assert yaml_path.exists(), "configs/visdrone.yaml must exist."

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "path" in data
        assert "train" in data
        assert "val" in data
        assert "names" in data

        names = data["names"]
        assert len(names) == 10, f"Expected 10 VisDrone classes, got {len(names)}"
        expected_classes = {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor",
        }
        for idx, name in expected_classes.items():
            assert names[idx] == name, f"Class {idx} mismatch: expected {name}, got {names.get(idx)}"

    def test_offline_safety_missing_weights(self) -> None:
        """Verify that attempting to find a missing weights file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Could not locate required offline file"):
            find_offline_file("nonexistent_checkpoint_12345.pt")

        with pytest.raises(FileNotFoundError, match="Explicitly specified file does not exist"):
            find_offline_file("yolov8s.pt", explicit_path="/invalid/path/yolov8s.pt")

    def test_pretrained_initialization_parity(self) -> None:
        """
        Verify that stock weights transferred to AMSA model achieve 100% key parity
        for all stock layers, leaving only AMSA lateral layers freshly initialized.
        """
        register_amsa()
        stock = YOLO("yolov8s.yaml")
        amsa = YOLO("configs/yolov8s-amsa.yaml")

        report = transfer_yolov8s_weights(amsa, stock.model.state_dict())

        assert report.total_transferred == 355
        assert report.backbone_transferred == 162
        assert report.neck_transferred == 108
        assert report.head_transferred == 85
        assert len(report.amsa_keys_left_fresh) == 84
        assert report.is_clean is True

        # Check parameter equality on backbone Layer 0 Conv weight
        b_w = stock.model.model[0].conv.weight
        a_w = amsa.model.model[0].conv.weight
        assert torch.equal(b_w, a_w)

        # Check parameter equality on Neck shifted layer: stock Layer 12 C2f -> AMSA Layer 15 C2f
        b_neck_w = stock.model.model[12].cv1.conv.weight
        a_neck_w = amsa.model.model[15].cv1.conv.weight
        assert torch.equal(b_neck_w, a_neck_w)

    def test_model_parameter_specifications(self) -> None:
        """Verify analytical and measured parameter counts for baseline and AMSA models."""
        register_amsa()
        baseline = YOLO("yolov8s.yaml")
        amsa = YOLO("configs/yolov8s-amsa.yaml")

        b_params = sum(p.numel() for p in baseline.model.parameters())
        a_params = sum(p.numel() for p in amsa.model.parameters())

        # Baseline YOLOv8s parameters (nc=80) = 11,166,560
        assert b_params == 11_166_560
        # AMSA-YOLOv8s parameters (nc=80) = 12,797,246
        assert a_params == 12_797_246
        # Delta strictly equals the sum of three AMSA modules: 1,630,686
        assert a_params - b_params == 1_630_686

    def test_batch_size_probe_dry_run(self) -> None:
        """Verify batch size probe function executes cleanly in dry-run mode."""
        safe_batch, results = run_probe(candidate_batches=[2], device_str="cpu")
        assert isinstance(safe_batch, int)
        assert safe_batch > 0
        assert "baseline" in results
        assert "amsa" in results
        assert 2 in results["baseline"]
        assert results["baseline"][2]["success"] is True
        assert results["amsa"][2]["success"] is True

    def test_disable_external_logging_callbacks_purges_registry(self) -> None:
        """
        Verify that disable_external_logging_callbacks() empties raytune/wb callback
        dictionaries, disables them in SETTINGS, and purges them from callback registries.
        """
        # Inject callbacks into registry to simulate active external integration
        raytune_cb.callbacks["on_fit_epoch_end"] = raytune_cb.on_fit_epoch_end
        wb_cb.callbacks["on_fit_epoch_end"] = wb_cb.on_fit_epoch_end
        wb_cb.callbacks["on_pretrain_routine_start"] = wb_cb.on_pretrain_routine_start

        test_registry = {
            "on_fit_epoch_end": [
                base_cb.on_fit_epoch_end,
                raytune_cb.on_fit_epoch_end,
                wb_cb.on_fit_epoch_end,
            ],
            "on_model_save": [base_cb.on_model_save],
            "on_pretrain_routine_start": [
                base_cb.on_pretrain_routine_start,
                wb_cb.on_pretrain_routine_start,
            ],
        }

        disable_external_logging_callbacks(test_registry)

        # 1. SETTINGS flags disabled
        assert SETTINGS["raytune"] is False
        assert SETTINGS["wandb"] is False

        # 2. Module-level callback dictionaries cleared
        assert len(raytune_cb.callbacks) == 0
        assert len(wb_cb.callbacks) == 0

        # 3. Target registry purged of Ray Tune and W&B callbacks
        remaining_fit_cbs = test_registry["on_fit_epoch_end"]
        assert len(remaining_fit_cbs) == 1
        assert remaining_fit_cbs[0] is base_cb.on_fit_epoch_end

        remaining_pretrain_cbs = test_registry["on_pretrain_routine_start"]
        assert len(remaining_pretrain_cbs) == 1
        assert remaining_pretrain_cbs[0] is base_cb.on_pretrain_routine_start

        # 4. Non-external callbacks untouched
        assert test_registry["on_model_save"] == [base_cb.on_model_save]

    def test_raytune_and_wandb_callbacks_absent_in_trainer(self) -> None:
        """
        Verify that newly initialized DetectionTrainer has zero Ray Tune or W&B callbacks
        registered, while retaining all core Ultralytics callbacks.
        """
        from ultralytics.models.yolo.detect import DetectionTrainer

        # Simulate Ray Tune and W&B enabled in SETTINGS before sanitization
        SETTINGS["raytune"] = True
        SETTINGS["wandb"] = True
        raytune_cb.callbacks["on_fit_epoch_end"] = raytune_cb.on_fit_epoch_end
        wb_cb.callbacks["on_fit_epoch_end"] = wb_cb.on_fit_epoch_end

        # Disable external loggers
        disable_external_logging_callbacks()

        # Build trainer instance with minimal overrides
        trainer = DetectionTrainer(
            overrides=dict(
                model="yolov8s.yaml",
                data="configs/visdrone-smoke.yaml",
                epochs=1,
                batch=2,
                device="cpu",
            )
        )

        # Verify no raytune or wb callback in any trainer event
        for event, cbs in trainer.callbacks.items():
            for cb in cbs:
                assert not is_raytune_or_wandb_callback(cb), (
                    f"Found forbidden external callback in trainer.callbacks['{event}']: "
                    f"{getattr(cb, '__module__', '')}.{getattr(cb, '__name__', '')}"
                )

        # Verify core callbacks are present
        assert any(
            cb is base_cb.on_fit_epoch_end
            for cb in trainer.callbacks.get("on_fit_epoch_end", [])
        )
        assert any(
            cb is base_cb.on_model_save
            for cb in trainer.callbacks.get("on_model_save", [])
        )

    def test_model_pretrain_routine_cleaner_hook(self) -> None:
        """
        Verify that setup_model attaches an on_pretrain_routine_start hook that
        cleans any stray external callbacks if injected into trainer.callbacks.
        """
        model = setup_model("baseline", allow_untrained_fallback=True)

        # Create a mock trainer with an injected raytune callback
        class MockTrainer:
            def __init__(self):
                self.callbacks = {
                    "on_fit_epoch_end": [base_cb.on_fit_epoch_end, raytune_cb.on_fit_epoch_end],
                    "on_model_save": [base_cb.on_model_save],
                }

        mock_trainer = MockTrainer()
        assert len(mock_trainer.callbacks["on_fit_epoch_end"]) == 2

        # Trigger on_pretrain_routine_start callbacks registered on model
        for hook in model.callbacks.get("on_pretrain_routine_start", []):
            hook(mock_trainer)

        # Verify the hook purged raytune callback
        assert len(mock_trainer.callbacks["on_fit_epoch_end"]) == 1
        assert mock_trainer.callbacks["on_fit_epoch_end"][0] is base_cb.on_fit_epoch_end
        assert mock_trainer.callbacks["on_model_save"] == [base_cb.on_model_save]

    def test_benchmark_training_config_unaffected_by_callback_disable(self) -> None:
        """
        Verify that disabling external callbacks has strictly zero impact on
        benchmark training hyperparameters, architectures, or parity.
        """
        args_before = get_training_args("baseline", "configs/visdrone.yaml")
        disable_external_logging_callbacks()
        args_after = get_training_args("baseline", "configs/visdrone.yaml")

        assert args_before == args_after
        for key in (
            "imgsz", "epochs", "batch", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "warmup_epochs", "cos_lr",
            "amp", "seed", "deterministic", "workers", "mosaic",
            "val", "plots", "save", "save_period",
        ):
            assert args_after[key] == DEFAULT_TRAINING_CONFIG[key]

    def test_fresh_baseline_and_amsa_unchanged(self) -> None:
        """Verify fresh runs without --resume return None for resume checkpoint."""
        assert resolve_resume_checkpoint("baseline", resume=False, resume_from=None) is None
        assert resolve_resume_checkpoint("amsa", resume=False, resume_from=None) is None

        # Verify fresh baseline and fresh amsa initialize without error
        register_amsa()
        fresh_b = setup_model("baseline", allow_untrained_fallback=True)
        fresh_a = setup_model("amsa", allow_untrained_fallback=True)
        assert fresh_b is not None
        assert fresh_a is not None

    def test_resume_resolves_baseline_and_amsa_last_pt(self, tmp_path: Path) -> None:
        """Verify --resume automatically resolves runs/visdrone/<model>/weights/last.pt."""
        b_ckpt_dir = tmp_path / "baseline" / "weights"
        b_ckpt_dir.mkdir(parents=True)
        b_last = b_ckpt_dir / "last.pt"
        b_last.touch()

        a_ckpt_dir = tmp_path / "amsa" / "weights"
        a_ckpt_dir.mkdir(parents=True)
        a_last = a_ckpt_dir / "last.pt"
        a_last.touch()

        resolved_b = resolve_resume_checkpoint("baseline", resume=True, project=tmp_path)
        resolved_a = resolve_resume_checkpoint("amsa", resume=True, project=tmp_path)

        assert resolved_b == b_last.resolve()
        assert resolved_a == a_last.resolve()

    def test_resume_from_explicit_path(self, tmp_path: Path) -> None:
        """Verify --resume-from resolves the exact specified checkpoint path."""
        custom_dir = tmp_path / "checkpoints"
        custom_dir.mkdir(parents=True)
        custom_ckpt = custom_dir / "epoch_150.pt"
        custom_ckpt.touch()

        resolved = resolve_resume_checkpoint("baseline", resume_from=custom_ckpt)
        assert resolved == custom_ckpt.resolve()

        # Both flags provided: explicit path takes precedence
        resolved_both = resolve_resume_checkpoint("amsa", resume=True, resume_from=custom_ckpt)
        assert resolved_both == custom_ckpt.resolve()

    def test_missing_checkpoint_raises_file_not_found_error(self, tmp_path: Path) -> None:
        """Verify clear FileNotFoundError when checkpoint does not exist."""
        # Auto-resume missing checkpoint
        with pytest.raises(FileNotFoundError, match="Cannot resume baseline training: checkpoint not found at"):
            resolve_resume_checkpoint("baseline", resume=True, project=tmp_path)

        with pytest.raises(FileNotFoundError, match="Cannot resume amsa training: checkpoint not found at"):
            resolve_resume_checkpoint("amsa", resume=True, project=tmp_path)

        # Explicit resume missing checkpoint
        missing_file = tmp_path / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError, match="Explicit resume checkpoint does not exist"):
            resolve_resume_checkpoint("baseline", resume_from=missing_file)

    def test_resume_path_does_not_invoke_fresh_pretrained_remapping(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Verify that resuming AMSA does NOT rebuild from stock yolov8s.pt and does NOT
        invoke transfer_yolov8s_weights(), loading the saved checkpoint directly.
        """
        register_amsa()
        amsa_model = YOLO("configs/yolov8s-amsa.yaml")

        # Save a valid dummy checkpoint mimicking Ultralytics save format
        dummy_ckpt = {
            "epoch": 25,
            "best_fitness": 0.55,
            "model": amsa_model.model,
            "ema": None,
            "updates": None,
            "optimizer": {"state": {}, "param_groups": [{"lr": 0.01}]},
            "train_args": {
                "model": "configs/yolov8s-amsa.yaml",
                "data": "configs/visdrone-smoke.yaml",
                "epochs": 300,
                "batch": 32,
                "imgsz": 640,
            },
        }
        ckpt_path = tmp_path / "amsa_last.pt"
        torch.save(dummy_ckpt, ckpt_path)

        # Spy on transfer_yolov8s_weights to guarantee it is NOT called
        transfer_called = []
        import scripts.train_benchmark as tb_module
        monkeypatch.setattr(
            tb_module,
            "transfer_yolov8s_weights",
            lambda *args, **kwargs: transfer_called.append(True),
        )

        resumed_model = setup_model("amsa", resume_checkpoint=ckpt_path)

        # Verification: transfer was never invoked
        assert len(transfer_called) == 0, "transfer_yolov8s_weights must NOT be called on resume runs!"
        assert resumed_model.ckpt_path == str(ckpt_path.resolve())
        assert resumed_model.ckpt.get("epoch") == 25

    def test_training_config_parity_with_and_without_resume(self) -> None:
        """
        Verify that training configuration parity is strictly maintained across baseline
        and AMSA models whether running fresh or resuming.
        """
        # 1. Fresh runs
        b_fresh = get_training_args("baseline", "configs/visdrone.yaml", resume=False)
        a_fresh = get_training_args("amsa", "configs/visdrone.yaml", resume=False)
        assert "resume" not in b_fresh
        assert "resume" not in a_fresh
        for k in set(b_fresh.keys()) - {"name"}:
            assert b_fresh[k] == a_fresh[k]

        # 2. Resumed runs
        b_resumed = get_training_args("baseline", "configs/visdrone.yaml", resume=True)
        a_resumed = get_training_args("amsa", "configs/visdrone.yaml", resume=True)
        assert b_resumed["resume"] is True
        assert a_resumed["resume"] is True
        for k in set(b_resumed.keys()) - {"name"}:
            assert b_resumed[k] == a_resumed[k]

        # 3. All non-resume hyperparameters remain identical to DEFAULT_TRAINING_CONFIG
        for key in (
            "imgsz", "epochs", "batch", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "warmup_epochs", "cos_lr",
            "amp", "seed", "deterministic", "workers", "mosaic",
            "val", "plots", "save", "save_period",
        ):
            assert b_resumed[key] == DEFAULT_TRAINING_CONFIG[key]
            assert a_resumed[key] == DEFAULT_TRAINING_CONFIG[key]

    def test_dry_run_with_resume_cli(self, tmp_path: Path) -> None:
        """Verify dry-run execution with --resume CLI flag."""
        from scripts.train_benchmark import run_benchmark_training
        import argparse

        # Create dummy checkpoint
        ckpt_dir = tmp_path / "baseline" / "weights"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "last.pt"

        # Build baseline model and save dummy checkpoint
        base_model = YOLO("yolov8s.yaml")
        torch.save({
            "epoch": 5,
            "best_fitness": 0.2,
            "model": base_model.model,
            "ema": None,
            "updates": None,
            "optimizer": {"state": {}, "param_groups": [{"lr": 0.01}]},
            "train_args": {"model": "yolov8s.yaml", "data": "configs/visdrone-smoke.yaml", "epochs": 300},
        }, ckpt_file)

        args = argparse.Namespace(
            model="baseline",
            weights=None,
            data="configs/visdrone-smoke.yaml",
            custom_data_root=None,
            epochs=300,
            batch=32,
            imgsz=640,
            device="cpu",
            workers=4,
            project=str(tmp_path),
            name="baseline",
            smoke=False,
            fraction=None,
            dry_run=True,
            resume=True,
            resume_from=None,
        )

        res = run_benchmark_training(args)
        assert res["status"] == "DRY_RUN_PASS"
        assert res["resumed"] is True
        assert res["resume_checkpoint"] == str(ckpt_file.resolve())
        assert res["train_args"]["resume"] is True
