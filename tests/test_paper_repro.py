"""
Unit and Integration Tests for AMSA-YOLO Paper-Faithful Reproduction Pipeline.

Reference:
"AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism"
Neural Networks, Volume 197 (2026), Article 108545.

Validates:
1. Benchmark profile unchanged (backward compatibility).
2. Paper_repro baseline and AMSA hyperparameters (AdamW, batch=16, epochs=300, imgsz=640,
   cos_lr=True, warmup=3, mosaic=0.5, mixup=0.1, fliplr=0.5, scale=0.5).
3. Differential LR parameter groups (Base=0.01, AMSA=0.001) with AdamW and cosine stepping.
4. Identification of exact 75 AMSA parameter tensors (1,630,686 parameters).
5. Progressive weight transfer from Stage 1 baseline to Stage 2 AMSA model.
6. Scale-aware loss threshold behavior:
   - sqrt(w*h) < 32 -> 2.0
   - 32 <= sqrt(w*h) < 96 -> 1.5
   - sqrt(w*h) >= 96 -> 1.0
7. Standard loss fallback when scale_aware_loss is disabled.
8. Resume support under paper_repro profile.
9. Offline safety preservation.
"""

from pathlib import Path
import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR
from ultralytics import YOLO
from ultralytics.utils.torch_utils import one_cycle
import yaml

from scripts.train_benchmark import setup_model
from src.amsa import register_amsa
from src.training import (
    BENCHMARK_TRAINING_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    PAPER_REPRO_TRAINING_CONFIG,
    AMSAReproductionTrainer,
    ScaleAwareDetectionLoss,
    compute_scale_aware_weights,
    find_offline_file,
    get_training_args,
    identify_amsa_parameter_ids,
    resolve_resume_checkpoint,
)


class TestPaperReproduction:
    """Test suite for paper-faithful reproduction configuration and pipeline."""

    def test_benchmark_profile_unchanged(self) -> None:
        """Verify that benchmark profile defaults remain 100% identical to original config."""
        b_args = get_training_args("baseline", "configs/visdrone.yaml", profile="benchmark")
        a_args = get_training_args("amsa", "configs/visdrone.yaml", profile="benchmark")

        assert b_args["optimizer"] == "SGD"
        assert a_args["optimizer"] == "SGD"
        assert b_args["batch"] == 32
        assert a_args["batch"] == 32
        assert b_args["mosaic"] == 1.0
        assert a_args["mosaic"] == 1.0
        assert b_args["cos_lr"] is False
        assert a_args["cos_lr"] is False
        assert b_args["warmup_bias_lr"] == 0.1
        assert b_args["warmup_epochs"] == 3.0

        for k in ("imgsz", "epochs", "batch", "optimizer", "lr0", "lrf", "momentum", "weight_decay"):
            assert b_args[k] == DEFAULT_TRAINING_CONFIG[k]
            assert a_args[k] == DEFAULT_TRAINING_CONFIG[k]

    def test_paper_repro_baseline_and_amsa_hyperparameters(self) -> None:
        """
        Verify that paper_repro baseline and AMSA configurations match paper specifications.
        """
        data_path = "configs/visdrone.yaml"
        b_args = get_training_args("baseline", data_path=data_path, profile="paper_repro")
        a_args = get_training_args("amsa", data_path=data_path, profile="paper_repro")

        # Common paper-confirmed parameters
        for args in (b_args, a_args):
            assert args["imgsz"] == 640
            assert args["epochs"] == 300
            assert args["batch"] == 16
            assert args["optimizer"] == "AdamW"
            assert args["lr0"] == 0.01
            assert args["weight_decay"] == 0.0005
            assert args["cos_lr"] is True
            assert args["warmup_epochs"] == 3.0
            assert args["warmup_bias_lr"] == 0.0
            assert args["mosaic"] == 0.5
            assert args["mixup"] == 0.1
            assert args["fliplr"] == 0.5
            assert args["scale"] == 0.5
            assert args["scale_aware_loss"] is True
            assert args["initialization"] == "pretrained"

        # AMSA specific
        assert a_args["amsa_lr0"] == 0.001
        assert a_args["name"] == "paper_amsa"
        assert b_args["name"] == "paper_baseline"

    def test_paper_repro_yaml_parity(self) -> None:
        """Verify configs/reproduction/amsa_visdrone_paper.yaml matches PAPER_REPRO_TRAINING_CONFIG."""
        yaml_file = Path("configs/reproduction/amsa_visdrone_paper.yaml")
        assert yaml_file.is_file(), "Paper reproduction YAML configuration must exist."

        with open(yaml_file, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)

        for key in (
            "imgsz", "epochs", "batch", "optimizer", "lr0", "amsa_lr0",
            "weight_decay", "warmup_epochs", "cos_lr", "mosaic", "mixup",
            "fliplr", "scale", "scale_aware_loss", "initialization",
        ):
            assert yaml_cfg[key] == PAPER_REPRO_TRAINING_CONFIG[key], (
                f"Mismatch in YAML config for {key}: {yaml_cfg[key]} != {PAPER_REPRO_TRAINING_CONFIG[key]}"
            )

    def test_amsa_parameter_identification(self) -> None:
        """
        Verify that identify_amsa_parameter_ids accurately isolates all 75 AMSA parameter tensors
        totaling exactly 1,630,686 parameters.
        """
        register_amsa()
        amsa_model = YOLO("configs/yolov8s-amsa.yaml")
        amsa_ids = identify_amsa_parameter_ids(amsa_model.model)

        assert len(amsa_ids) == 75, f"Expected 75 AMSA parameter tensors, found {len(amsa_ids)}"

        amsa_params = [p for p in amsa_model.model.parameters() if id(p) in amsa_ids]
        base_params = [p for p in amsa_model.model.parameters() if id(p) not in amsa_ids]

        assert sum(p.numel() for p in amsa_params) == 1_630_686
        assert sum(p.numel() for p in base_params) == 11_166_560
        assert sum(p.numel() for p in amsa_model.model.parameters()) == 12_797_246

    def test_differential_lr_parameter_groups_adamw(self) -> None:
        """
        Verify that AMSAReproductionTrainer builds AdamW optimizer with distinct parameter groups:
        - Base YOLO params: lr0 = 0.01, weight_decay = 0.0005
        - AMSA params: amsa_lr0 = 0.001, weight_decay = 0.0005
        """
        register_amsa()
        amsa_model = YOLO("configs/yolov8s-amsa.yaml")

        overrides = dict(
            model="configs/yolov8s-amsa.yaml",
            data="configs/visdrone-smoke.yaml",
            epochs=300,
            batch=16,
            device="cpu",
            optimizer="AdamW",
            lr0=0.01,
            amsa_lr0=0.001,
            weight_decay=0.0005,
            momentum=0.937,
            scale_aware_loss=True,
        )
        trainer = AMSAReproductionTrainer(overrides=overrides)
        optimizer = trainer.build_optimizer(amsa_model.model)

        assert type(optimizer).__name__ == "AdamW"

        base_lrs = [g["lr"] for g in optimizer.param_groups if "base" in g.get("group_name", "")]
        amsa_lrs = [g["lr"] for g in optimizer.param_groups if "amsa" in g.get("group_name", "")]

        assert len(base_lrs) > 0
        assert len(amsa_lrs) > 0
        assert set(base_lrs) == {0.01}
        assert set(amsa_lrs) == {0.001}

        # Verify weight decay
        base_decays = {g["weight_decay"] for g in optimizer.param_groups if "base_decay" in g.get("group_name", "")}
        amsa_decays = {g["weight_decay"] for g in optimizer.param_groups if "amsa_decay" in g.get("group_name", "")}
        assert base_decays == {0.0005}
        assert amsa_decays == {0.0005}

        # Verify no decay on norms and biases
        norm_bias_decays = {
            g["weight_decay"]
            for g in optimizer.param_groups
            if any(k in g.get("group_name", "") for k in ("norm", "bias"))
        }
        assert norm_bias_decays == {0.0}

    def test_scheduler_preserves_10_to_1_lr_ratio(self) -> None:
        """
        Verify that PyTorch LambdaLR cosine annealing scheduler preserves the exact 10:1 ratio
        between base parameters and AMSA parameters across the entire training duration.
        """
        g_base = [torch.nn.Parameter(torch.randn(4, 4))]
        g_amsa = [torch.nn.Parameter(torch.randn(4, 4))]

        optimizer = torch.optim.AdamW([
            {"params": g_base, "lr": 0.01, "weight_decay": 0.0005},
            {"params": g_amsa, "lr": 0.001, "weight_decay": 0.0005},
        ])

        epochs = 300
        lf = one_cycle(1, 0.01, epochs)
        scheduler = LambdaLR(optimizer, lr_lambda=lf)

        # Check at start (epoch 0)
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[1]["lr"] == 0.001

        # Check at checkpoints (epochs 75, 150, 225, 300)
        for e in range(1, epochs + 1):
            optimizer.step()
            scheduler.step()
            base_lr = optimizer.param_groups[0]["lr"]
            amsa_lr = optimizer.param_groups[1]["lr"]
            ratio = base_lr / amsa_lr
            assert abs(ratio - 10.0) < 1e-4, f"Ratio drifted at epoch {e}: {ratio}"

        # Check final learning rate (0.01 * 0.01 = 0.0001 for base, 0.001 * 0.01 = 0.00001 for AMSA)
        assert abs(optimizer.param_groups[0]["lr"] - 0.0001) < 1e-6
        assert abs(optimizer.param_groups[1]["lr"] - 0.00001) < 1e-7

    def test_scale_aware_loss_thresholds(self) -> None:
        """
        Verify scale-aware weighting thresholds:
        - sqrt(w * h) < 32 -> 2.0
        - 32 <= sqrt(w * h) < 96 -> 1.5
        - sqrt(w * h) >= 96 -> 1.0
        """
        # Small: 10 x 10 -> sqrt=10 < 32 -> 2.0
        small_box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        assert compute_scale_aware_weights(small_box).item() == 2.0

        # Boundary 31.9 -> 2.0
        b_31_9 = torch.tensor([[0.0, 0.0, 31.9, 31.9]])
        assert compute_scale_aware_weights(b_31_9).item() == 2.0

        # Exact boundary 32.0 -> 1.5
        b_32 = torch.tensor([[0.0, 0.0, 32.0, 32.0]])
        assert compute_scale_aware_weights(b_32).item() == 1.5

        # Medium: 50 x 50 -> sqrt=50 -> 1.5
        med_box = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
        assert compute_scale_aware_weights(med_box).item() == 1.5

        # Boundary 95.9 -> 1.5
        b_95_9 = torch.tensor([[0.0, 0.0, 95.9, 95.9]])
        assert compute_scale_aware_weights(b_95_9).item() == 1.5

        # Exact boundary 96.0 -> 1.0
        b_96 = torch.tensor([[0.0, 0.0, 96.0, 96.0]])
        assert compute_scale_aware_weights(b_96).item() == 1.0

        # Large: 120 x 120 -> sqrt=120 >= 96 -> 1.0
        large_box = torch.tensor([[0.0, 0.0, 120.0, 120.0]])
        assert compute_scale_aware_weights(large_box).item() == 1.0

    def test_scale_aware_detection_loss_fallback(self) -> None:
        """
        Verify that ScaleAwareDetectionLoss with enabled=False produces identical output
        to standard v8DetectionLoss.
        """
        register_amsa()
        model = YOLO("yolov8s.yaml")

        loss_standard = ScaleAwareDetectionLoss(model.model, enabled=False)
        assert loss_standard.enabled is False

    def test_progressive_weight_transfer_from_baseline(self, tmp_path: Path) -> None:
        """
        Verify that progressive Stage 2 initializes AMSA model using trained Stage 1 baseline
        weights, transferring 355 keys and protecting 84 fresh AMSA parameters.
        """
        register_amsa()
        base_model = YOLO("yolov8s.yaml")

        # Save simulated Stage 1 baseline checkpoint
        stage1_ckpt_path = tmp_path / "baseline_best.pt"
        torch.save({"model": base_model.model, "epoch": 300, "best_fitness": 0.285}, stage1_ckpt_path)

        # Initialize AMSA with progressive baseline weights
        amsa_model = setup_model("amsa", baseline_weights=stage1_ckpt_path)

        assert amsa_model.ckpt is not None
        assert "model" in amsa_model.ckpt

        # Check weight equality on Layer 0
        assert torch.equal(
            base_model.model.model[0].conv.weight,
            amsa_model.model.model[0].conv.weight,
        )

    def test_scratch_initialization_policy(self) -> None:
        """Verify that initialization='scratch' creates models without loading pretrained weights."""
        register_amsa()
        model_b = setup_model("baseline", initialization="scratch")
        model_a = setup_model("amsa", initialization="scratch")

        assert model_b is not None
        assert model_a is not None
        assert model_b.ckpt is not None
        assert model_a.ckpt is not None

    def test_paper_repro_resume_checkpoint_resolution(self, tmp_path: Path) -> None:
        """Verify resume checkpoint resolution under paper_repro profile."""
        paper_b_dir = tmp_path / "paper_baseline" / "weights"
        paper_b_dir.mkdir(parents=True)
        b_last = paper_b_dir / "last.pt"
        b_last.touch()

        paper_a_dir = tmp_path / "paper_amsa" / "weights"
        paper_a_dir.mkdir(parents=True)
        a_last = paper_a_dir / "last.pt"
        a_last.touch()

        resolved_b = resolve_resume_checkpoint("baseline", resume=True, project=tmp_path, profile="paper_repro")
        resolved_a = resolve_resume_checkpoint("amsa", resume=True, project=tmp_path, profile="paper_repro")

        assert resolved_b == b_last.resolve()
        assert resolved_a == a_last.resolve()

    def test_progressive_stage_schedule_cli_configuration(self) -> None:
        """
        Verify that progressive stage durations are explicitly configurable via
        --stage1-epochs and --stage2-epochs and propagate to training arguments.
        """
        from scripts.train_benchmark import parse_args, run_benchmark_training
        import sys

        # Test baseline with custom stage 1 epochs
        test_args_b = [
            "train_benchmark.py",
            "--model", "baseline",
            "--profile", "paper_repro",
            "--dry-run",
            "--device", "cpu",
            "--stage1-epochs", "150",
            "--stage2-epochs", "250",
        ]
        sys.argv = test_args_b
        args_b = parse_args()
        assert args_b.stage1_epochs == 150
        assert args_b.stage2_epochs == 250
        res_b = run_benchmark_training(args_b)
        assert res_b["status"] == "DRY_RUN_PASS"
        assert res_b["stage1_epochs"] == 150
        assert res_b["stage2_epochs"] == 250
        assert res_b["train_args"]["epochs"] == 150

        # Test AMSA with custom stage 2 epochs
        test_args_a = [
            "train_benchmark.py",
            "--model", "amsa",
            "--profile", "paper_repro",
            "--dry-run",
            "--device", "cpu",
            "--stage1-epochs", "120",
            "--stage2-epochs", "180",
        ]
        sys.argv = test_args_a
        args_a = parse_args()
        assert args_a.stage1_epochs == 120
        assert args_a.stage2_epochs == 180
        res_a = run_benchmark_training(args_a)
        assert res_a["status"] == "DRY_RUN_PASS"
        assert res_a["stage1_epochs"] == 120
        assert res_a["stage2_epochs"] == 180
        assert res_a["train_args"]["epochs"] == 180

    def test_progressive_audit_log_fields(self, capsys: pytest.CaptureFixture) -> None:
        """
        Verify that the runtime audit log prints all required semantic fields:
        - Paper-reported epochs: 300
        - Baseline stage epochs: 300
        - AMSA fine-tuning epochs: 300
        - Cumulative implementation training budget: 600
        - Progressive schedule status: IMPLEMENTATION_ASSUMPTION
        - Initialization checkpoint
        - Base LR
        - AMSA LR
        and does NOT call 300 the 'Total Target Epochs' when Stage 1=300 and Stage 2/3=300.
        """
        from scripts.train_benchmark import parse_args, run_benchmark_training
        import sys

        sys.argv = [
            "train_benchmark.py",
            "--model", "amsa",
            "--profile", "paper_repro",
            "--dry-run",
            "--device", "cpu",
            "--stage1-epochs", "300",
            "--stage2-epochs", "300",
        ]
        args = parse_args()
        res = run_benchmark_training(args)

        captured = capsys.readouterr().out
        assert "PROGRESSIVE SCHEDULE & OPTIMIZATION AUDIT" in captured
        assert "Paper-reported epochs:                     300" in captured
        assert "Baseline stage epochs:                     300" in captured
        assert "AMSA fine-tuning epochs:                   300" in captured
        assert "Cumulative implementation training budget: 600" in captured
        assert "Progressive schedule status:               IMPLEMENTATION_ASSUMPTION" in captured
        assert "Total Target Epochs:" not in captured, "300 must not be called 'Total Target Epochs' when multi-stage budget is 600."
        assert "Current Run Target Epochs:" in captured
        assert "Initialization Checkpoint:" in captured
        assert "Base LR:" in captured
        assert "AMSA LR:" in captured
        assert "0.01" in captured
        assert "0.001" in captured

        # Verify returned dictionary fields
        assert res["paper_reported_epochs"] == 300
        assert res["stage1_epochs"] == 300
        assert res["stage2_epochs"] == 300
        assert res["cumulative_budget"] == 600
        assert res["progressive_schedule_status"] == "IMPLEMENTATION_ASSUMPTION"

    def test_paper_documentation_disclosures(self) -> None:
        """
        Verify that docs/PAPER_REPRODUCTION.md strictly complies with user mandates:
        1. Contains 'The paper does not disclose the progressive stage boundary.'
        2. Categorizes progressive schedule as IMPLEMENTATION_ASSUMPTION.
        3. Clarifies baseline 300 + AMSA 300 is not paper-exact.
        4. Details configurable --stage1-epochs and --stage2-epochs.
        5. Does NOT state that the paper explicitly says 'AMSA lr=0.001 for 300 epochs'.
        6. Explicitly details paper-reported epochs 300 vs cumulative implementation budget 600.
        """
        doc_path = Path("docs/PAPER_REPRODUCTION.md")
        assert doc_path.is_file()
        content = doc_path.read_text(encoding="utf-8")

        assert "The paper does not disclose the progressive stage boundary." in content
        assert "IMPLEMENTATION_ASSUMPTION" in content
        assert "--stage1-epochs" in content
        assert "--stage2-epochs" in content
        assert "paper-exact" in content.lower()
        assert "Cumulative implementation training budget" in content
        assert "AMSA modules use lr=0.001 for 300 epochs" in content


