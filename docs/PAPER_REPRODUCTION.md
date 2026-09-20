# Paper-Faithful Reproduction Specification & Methodology

**Target Publication:**  
*AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism*  
Canjin Wang, Peng Sun, Chunhui Yang, Xianglong Teng, Rijun Wang  
*Neural Networks*, Volume 197, Article 108545 (January 2026).

---

## 1. Provenance Classification & Parameter Taxonomy

Every training setting, loss formulation, and optimization hyperparameter is explicitly categorized under one of three strict scientific classifications:

1. **`EXACT_FROM_PAPER`**: Explicitly documented in the paper text, tables, or equations.
2. **`ULTRALYTICS_APPROXIMATION`**: The closest defensible mapping to Ultralytics 8.2.103 anchor-free mechanics where modern YOLOv8 internals diverge from legacy anchor-based formulations.
3. **`IMPLEMENTATION_ASSUMPTION`**: Unspecified by the authors; principled engineering defaults are adopted and made configurable without claiming paper authority.

---

## 2. Comprehensive Hyperparameter Provenance Table

| Setting | Paper Specification | Repository Implementation | Status | Notes / Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | Baseline: YOLOv8s<br>Proposed: YOLOv8s + AMSA | `yolov8s.yaml`<br>`configs/yolov8s-amsa.yaml` | `EXACT_FROM_PAPER` | AMSA inserted laterally at backbone P3, P4, P5 prior to neck. |
| **Dataset** | VisDrone2019<br>Train: 6,471 images<br>Val: 548 images | `configs/visdrone.yaml` (10 classes) | `EXACT_FROM_PAPER` | Exact class and split parity. |
| **Input Resolution** | $640 \times 640$ | `imgsz: 640` | `EXACT_FROM_PAPER` | Standard evaluation resolution. |
| **Batch Size** | 16 | `batch: 16` | `EXACT_FROM_PAPER` | Explicitly stated in paper Section 4. |
| **Total Epochs** | 300 | `epochs: 300` | `EXACT_FROM_PAPER` | Explicitly stated in paper Section 4. |
| **Optimizer** | AdamW | `optimizer: AdamW` | `EXACT_FROM_PAPER` | Explicitly stated in paper Section 4. |
| **Base Learning Rate ($lr_0$)** | 0.01 | `lr0: 0.01` | `EXACT_FROM_PAPER` | Base YOLO parameter group initial learning rate. |
| **AMSA Fine-Tuning LR** | 0.001 | `amsa_lr0: 0.001` | `EXACT_FROM_PAPER` | AMSA lateral modules parameter group initial learning rate (Section 4). Note: Paper states this separately from total epochs; does not state AMSA lr=0.001 for 300 epochs. |
| **Weight Decay** | 0.0005 | `weight_decay: 0.0005` | `EXACT_FROM_PAPER` | Explicitly stated in paper Section 4. |
| **LR Schedule** | Cosine Annealing | `cos_lr: True` | `EXACT_FROM_PAPER` | PyTorch `LambdaLR` with cosine curve down to `lrf`. |
| **Warmup Duration** | 3 epochs | `warmup_epochs: 3.0` | `EXACT_FROM_PAPER` | Warmup applied across all parameter groups. |
| **Warmup Bias LR** | Unspecified | `warmup_bias_lr: 0.0` | `IMPLEMENTATION_ASSUMPTION` | Standard best practice for AdamW to prevent bias divergence. |
| **Final LR Factor (`lrf`)** | Unspecified | `lrf: 0.01` | `IMPLEMENTATION_ASSUMPTION` | Final LR = $0.01 \times lr_0$ ($10^{-4}$ for base, $10^{-5}$ for AMSA). |
| **AdamW Betas / Momentum** | Unspecified | `(0.937, 0.999)` | `IMPLEMENTATION_ASSUMPTION` | Standard Ultralytics AdamW beta configuration. |
| **Mosaic Augmentation** | Probability = 0.5 | `mosaic: 0.5` | `EXACT_FROM_PAPER` | Explicitly stated in paper Section 4. |
| **MixUp Augmentation** | Probability = 0.1 | `mixup: 0.1` | `EXACT_FROM_PAPER` | Explicitly stated in paper Section 4. |
| **Horizontal Flip** | Probability = 0.5 | `fliplr: 0.5` | `EXACT_FROM_PAPER` | Explicitly stated in paper Section 4. |
| **Random Scaling** | Range 0.5 to 1.5 | `scale: 0.5` | `ULTRALYTICS_APPROXIMATION` | Ultralytics maps `scale: 0.5` to range $[1 - 0.5, 1 + 0.5] = [0.5, 1.5]$. |
| **HSV Augmentation** | Enabled (values unspecified) | `hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4` | `IMPLEMENTATION_ASSUMPTION` | Standard YOLO HSV color-space augmentation values. |
| **Scale-Aware Loss Thresholds** | $<32 \to 2.0$<br>$32..95 \to 1.5$<br>$\ge 96 \to 1.0$ | `compute_scale_aware_weights()` in `src/training/loss.py` | `EXACT_FROM_PAPER` | Exact piece-wise thresholds based on $\sqrt{w \cdot h}$ in pixels. |
| **Loss Formulation** | $L_{total} = \lambda_{cls} w_s L_{cls} + \lambda_{box} w_s L_{box} + \lambda_{obj} w_s L_{obj}$ | `ScaleAwareDetectionLoss` applying $w_s$ to CIoU, DFL, and cls | `ULTRALYTICS_APPROXIMATION` | YOLOv8 replaces $L_{obj}$ with Task-Aligned Assigner + DFL. |
| **Pretrained vs Scratch** | Unspecified in paper text | Default `pretrained` (configurable: `--initialization scratch`) | `IMPLEMENTATION_ASSUMPTION` | Configurable policy; documented as engineering assumption. |
| **Progressive Training Concept** | Two-stage: Train baseline, then fine-tune AMSA | Stage 1: baseline run<br>Stage 2/3: `--baseline-weights` transfer | `EXACT_FROM_PAPER` | Conceptual strategy: baseline weights transferred to backbone & neck; AMSA lateral fresh. |
| **Progressive Stage Boundary & Schedule** | Unspecified in paper | Configurable: `--stage1-epochs 300 --stage2-epochs 300` | `IMPLEMENTATION_ASSUMPTION` | **The paper does not disclose the progressive stage boundary.** Whether 300 epochs means total epochs or per-stage is ambiguous. |
| **Attention Regularization** | "Appropriate regularization applied" | Weight decay 0.0005 applied to attention parameters | `IMPLEMENTATION_ASSUMPTION` | Unified with global AdamW weight decay. |

---

## 3. Progressive Training Schedule Disclosures & Assumptions

> [!IMPORTANT]
> **The paper does not disclose the progressive stage boundary.**  
> The AMSA-YOLO paper separately states:
> 1. Training epochs = 300 (Section 4, experimental setup).
> 2. First train the basic YOLOv8 network, then gradually add AMSA modules.
> 3. AMSA modules use learning rate $lr = 0.001$.
> 
> Critical Semantic Disclosures:
> - **No Bound Duration**: The paper does **NOT** state that "AMSA modules use lr=0.001 for 300 epochs". The 300 epochs setting and the AMSA lr=0.001 setting are presented separately and are **not** explicitly bound into a 300-epoch AMSA fine-tuning stage.
> - **Unspecified Stage Boundary**: The paper does **NOT** specify:
>   - How many epochs belong to Stage 1 (baseline training).
>   - How many epochs belong to Stage 2/3 (AMSA fine-tuning).
>   - Whether the reported 300 epochs denotes total cumulative progressive epochs across both stages (e.g. $150 + 150 = 300$), or 300 epochs for AMSA fine-tuning starting after a separately trained baseline checkpoint ($300 + 300 = 600$ total implementation budget).
> 
> Therefore:
> - Running baseline 300 epochs followed by AMSA for another 300 epochs is **NOT** paper-exact. It is an **`IMPLEMENTATION_ASSUMPTION`**.
> - The stage durations are made explicitly configurable in the pipeline via `--stage1-epochs` and `--stage2-epochs` (or `stage1_epochs` / `stage2_epochs` in config).
> - The standard baseline reproduction is preserved at 300 epochs.
> - When running Stage 1 = 300 and Stage 2/3 = 300, the runtime audit cleanly separates per-stage targets from the cumulative budget:
>   - **Paper-reported epochs**: 300
>   - **Baseline stage epochs**: 300
>   - **AMSA fine-tuning epochs**: 300
>   - **Cumulative implementation training budget**: 600
>   - **Progressive schedule status**: `IMPLEMENTATION_ASSUMPTION`
> - We do **not** ambiguously call 300 the "Total Target Epochs" when running a multi-stage schedule.

---

## 4. Scale-Aware Loss Reconciliation (Section 3.6)

### Mathematical Formulation in Paper:
The paper defines instance-level scale-dependent weighting:
$$w_{scale} = \begin{cases} 2.0 & \text{if } \sqrt{w \cdot h} < 32 \\ 1.5 & \text{if } 32 \le \sqrt{w \cdot h} < 96 \\ 1.0 & \text{if } \sqrt{w \cdot h} \ge 96 \end{cases}$$
$$L_{total} = \lambda_{cls} w_{scale} L_{cls} + \lambda_{box} w_{scale} L_{box} + \lambda_{obj} w_{scale} L_{obj}$$

### Architectural Mismatch in Ultralytics YOLOv8:
1. **Absence of Objectness Loss ($L_{obj}$):**  
   YOLOv8 is an anchor-free architecture that abandons the classical separate objectness branch ($L_{obj}$) present in YOLOv3–YOLOv5. Instead, classification and objectness are unified through Task-Aligned Assigner targets:
   $$L_{cls} = \text{BCEWithLogitsLoss}(pred\_scores, target\_scores)$$
2. **Distribution Focal Loss ($L_{dfl}$):**  
   Boundary regression uses Generalized Focal Loss (DFL) to model continuous coordinate distributions, alongside complete intersection-over-union (CIoU) loss ($L_{box}$).

### Our Defensible Approximation (`ULTRALYTICS_APPROXIMATION`):
In `src/training/loss.py` (`ScaleAwareDetectionLoss`):
1. Bounding box pixel dimensions $\sqrt{w \cdot h}$ are evaluated for all assigned positive instances ($fg\_mask$).
2. The exact piece-wise multiplier $w_{scale} \in \{2.0, 1.5, 1.0\}$ is applied to:
   - CIoU regression loss ($L_{box}$)
   - Distribution Focal Loss ($L_{dfl}$)
   - Foreground anchor classification loss ($L_{cls}$)
3. Background anchors retain neutral unit weighting ($1.0$).
4. Feature flag `--scale-aware-loss` / `--no-scale-aware-loss` allows toggling between scale-aware loss and standard Ultralytics detection loss.

---

## 5. Differential Learning Rate Architecture

In `src/training/trainer.py` (`AMSAReproductionTrainer`):
1. **Parameter Segregation:**
   - All 75 parameter tensors of the AMSA lateral modules (`AMSAModule`, totaling 1,630,686 parameters) are isolated via `identify_amsa_parameter_ids()`.
   - Remaining 184 parameter tensors (11,166,560 parameters) belong to the base YOLOv8s backbone, neck, and Detect head.
2. **Optimizer Parameter Groups (AdamW):**
   - Base weights (decay 0.0005): initial learning rate = `0.01`
   - Base norm / bias (decay 0.0): initial learning rate = `0.01`
   - AMSA weights (decay 0.0005): initial learning rate = `0.001`
   - AMSA norm / bias (decay 0.0): initial learning rate = `0.001`
3. **Scheduler Mechanics:**
   PyTorch `optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=one_cycle(1, 0.01, epochs))` computes:
   $$lr_{group}(epoch) = initial\_lr_{group} \times \lambda(epoch)$$
   Both base parameters and AMSA parameters follow the exact same cosine annealing curve, maintaining the exact $10 : 1$ ratio ($0.01 : 0.001 \to 0.0001 : 0.00001$) throughout all 300 epochs.

---

## 6. Execution Guide: Running Paper Reproduction

### A. Baseline YOLOv8s Paper Reproduction
Trains standard YOLOv8s under paper-confirmed hyperparameters (AdamW, Batch 16, Cosine LR, Mosaic 0.5, MixUp 0.1, Flip 0.5):
```bash
python scripts/train_benchmark.py \
  --model baseline \
  --profile paper_repro \
  --data configs/visdrone.yaml \
  --device 0
```
Output checkpoint: `runs/visdrone/paper_baseline/weights/best.pt`.

### B. AMSA-YOLO Progressive Paper Reproduction
Initializes from the Stage 1 trained baseline, transfers backbone and neck/head weights, and fine-tunes with AMSA modules at differential LR = 0.001 and scale-aware loss:
```bash
python scripts/train_benchmark.py \
  --model amsa \
  --profile paper_repro \
  --baseline-weights runs/visdrone/paper_baseline/weights/best.pt \
  --data configs/visdrone.yaml \
  --device 0
```
*Note: If `--baseline-weights` is omitted, the pipeline automatically looks for `runs/visdrone/paper_baseline/weights/best.pt`.*

### C. Resuming an Interrupted Paper Reproduction Run
Both baseline and AMSA reproduction runs fully support seamless resumption preserving optimizer state, scheduler step, epoch counter, and EMA:
```bash
# Resume baseline reproduction
python scripts/train_benchmark.py \
  --model baseline \
  --profile paper_repro \
  --resume \
  --device 0

# Resume AMSA reproduction
python scripts/train_benchmark.py \
  --model amsa \
  --profile paper_repro \
  --resume \
  --device 0
```

---

## 7. Paper Target Metrics (Table 2 on VisDrone)

| Architecture | Paper Reported Metric | Target Interpretation | Repository Output Metric Keys |
| :--- | :--- | :--- | :--- |
| **YOLOv8s Baseline** | $\approx 28.5$ | Overall mAP@0.5:0.95 | `metrics/mAP50-95(B)` & `metrics/mAP50(B)` |
| **AMSA-YOLOv8s** | $\approx 31.2$ | Overall mAP@0.5:0.95 | `metrics/mAP50-95(B)` & `metrics/mAP50(B)` |
| **Reported Gain** | $+2.7$ mAP | Statistical improvement on small objects | Tracked in `benchmark_metrics.json` |
