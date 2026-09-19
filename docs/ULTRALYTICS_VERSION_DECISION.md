# Ultralytics Baseline Version Decision for AMSA-YOLO Reproduction

**Document Status:** FINAL & LOCKED  
**Chosen Reproduction Baseline:** `ultralytics==8.2.103` (Release Date: September 28, 2024)  

---

## 1. Exact Paper Facts

The paper governing this reproduction project is:
- **Title:** *AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism*
- **Authors:** Canjin Wang, Peng Sun, Chunhui Yang, Xianglong Teng, Rijun Wang
- **Publication Venue:** *Neural Networks*, Volume 197, Article 108545 (Published January 2026)
- **Baseline Architecture:** The paper explicitly establishes **YOLOv8s** (YOLOv8 Small) as its comparative and architectural baseline.
- **Benchmark Claims:** AMSA-YOLO achieves a +2.3 percentage point increase in mAP@0.5:0.95 and a +3.6 percentage point increase in AP_s (small objects) over baseline YOLOv8s on the COCO dataset, with evaluations on VisDrone and CrowdHuman.
- **Absence of Version Metadata in Published Paper:**
  - The published paper does **not** disclose an exact Ultralytics pip package version tag (e.g. `v8.x.x`) or a Git commit SHA.
  - Academic indexing databases (ScienceDirect, ResearchGate, DBLP) and public registries link **no official public code repository** attached to this paper.

---

## 2. Controlled Reproduction Baseline: `ultralytics==8.2.103`

Because the original paper does not disclose the specific software commit or package release, **`ultralytics==8.2.103` was selected as a controlled reproduction baseline chosen by this project**:

- **It is NOT verified as the authors' exact version.**
- **It is an engineering and scientific baseline chosen for strict reproducibility.**

### Technical Rationale for this Baseline Choice:
1. **Definitive YOLOv8 Milestone Prior to YOLO11 (v8.3.0):**
   - On September 30, 2024, Ultralytics released version `8.3.0`, introducing the YOLO11 model family and significantly restructuring internal module registries, YAML parser semantics, and export backends.
   - Version `8.2.103` is the **final release of the 8.2 series** right before the YOLO11 architectural overhaul. It embodies the definitive, stable form of the YOLOv8 architecture.
2. **Architecture Invariance:**
   - The core YOLOv8 layer definitions (`yolov8.yaml`, `yolov8s.yaml`), convolutional structures (`Conv`, `C2f`, `SPPF`), and detection head (`Detect`) in `8.2.103` maintain the exact tensor dimensions, channel counts, and anchor-free loss functions used by YOLOv8.
3. **Compatibility with Kaggle PyTorch 2.10.0+cu128:**
   - Modern bleeding-edge releases (e.g. `8.4.x`) introduce unpinned PyPI dependencies that conflict with Kaggle's verified container. Version `8.2.103` resolves cleanly on Python 3.12 (`cp312`) without requiring external package changes.

---

## 3. Explicit Acknowledgement of Uncertainty & Reporting Mandate

1. **Uncertainty:** The original authors may have developed their attention modules on an earlier minor release (e.g. `8.0.x`, `8.1.x`) or a private local fork. There is no public record documenting their exact environment.
2. **Reporting Requirement:** All future architecture reproduction results, benchmark metrics, and ablation studies produced by this project **must explicitly report `ultralytics==8.2.103` as the chosen baseline**, ensuring scientific transparency rather than claiming identity with an unstated upstream environment.
