"""
Scale-Aware Detection Loss for AMSA-YOLO.

Paper Reference:
"AMSA-YOLO: Real-time object detection with adaptive multi-scale attention mechanism"
Neural Networks, Volume 197 (2026), Article 108545.

Equation:
    w_scale = 2.0 if sqrt(w * h) < 32
              1.5 if 32 <= sqrt(w * h) < 96
              1.0 if sqrt(w * h) >= 96

    L_total = lambda_cls * w_scale * L_cls +
              lambda_box * w_scale * L_box +
              lambda_dfl * w_scale * L_dfl

Ultralytics 8.2.103 Reconciliation Note:
    The paper specifies a classical formulation with L_obj (objectness). YOLOv8 is an
    anchor-free architecture with TaskAlignedAssigner that replaces separate objectness
    with Task-Aligned BCE classification and Distribution Focal Loss (DFL).
    This implementation applies the paper's exact sqrt(w * h) thresholds to positive
    anchors across box regression (CIoU), DFL, and classification BCE, labeled as
    ULTRALYTICS_APPROXIMATION.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def compute_scale_aware_weights(
    bboxes: torch.Tensor,
    small_thresh: float = 32.0,
    med_thresh: float = 96.0,
    small_weight: float = 2.0,
    med_weight: float = 1.5,
    large_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute scale-aware weights based on bounding box dimensions in pixel space.

    Args:
        bboxes: Bounding box coordinates [..., 4] in xyxy or xywh format.
        small_thresh: Threshold below which an object is considered small (< 32 px).
        med_thresh: Threshold below which an object is considered medium (< 96 px).
        small_weight: Loss multiplier for small objects (default: 2.0).
        med_weight: Loss multiplier for medium objects (default: 1.5).
        large_weight: Loss multiplier for large objects (default: 1.0).

    Returns:
        torch.Tensor: Weight tensor of shape bboxes.shape[:-1].
    """
    # Determine width and height
    # If xyxy: w = x2 - x1, h = y2 - y1
    # Check if x2 >= x1 across elements
    w = (bboxes[..., 2] - bboxes[..., 0]).clamp(min=0.0)
    h = (bboxes[..., 3] - bboxes[..., 1]).clamp(min=0.0)
    scale = torch.sqrt(w * h)

    weights = torch.full_like(scale, large_weight)
    weights = torch.where(scale < small_thresh, torch.tensor(small_weight, device=scale.device, dtype=scale.dtype), weights)
    weights = torch.where(
        (scale >= small_thresh) & (scale < med_thresh),
        torch.tensor(med_weight, device=scale.device, dtype=scale.dtype),
        weights,
    )
    return weights


class ScaleAwareBboxLoss(BboxLoss):
    """
    Bounding Box loss (CIoU + DFL) scaled by instance-level scale-aware weights.
    """

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        scale_weights=None,
    ):
        """IoU and DFL loss with scale-aware instance weighting."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if scale_weights is not None:
            weight = weight * scale_weights.unsqueeze(-1)

        from ultralytics.utils.metrics import bbox_iou
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            from ultralytics.utils.loss import bbox2dist
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class ScaleAwareDetectionLoss(v8DetectionLoss):
    """
    Paper-faithful scale-aware detection loss for AMSA-YOLO.

    Inherits from Ultralytics v8DetectionLoss and injects instance-level
    scale-aware weighting based on paper Section 3.6 equations.
    """

    def __init__(self, model, tal_topk: int = 10, enabled: bool = True):
        super().__init__(model, tal_topk=tal_topk)
        self.enabled = enabled
        self.scale_bbox_loss = ScaleAwareBboxLoss(self.reg_max).to(self.device)

    def __call__(self, preds, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate loss with scale-aware weighting for box, cls, and dfl.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - total_loss * batch_size (for backward propagation)
                - loss items detached [box, cls, dfl]
        """
        if not self.enabled:
            return super().__call__(preds, batch)

        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Target assignment
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Compute scale-aware weights on target_bboxes in pixel space
        # target_bboxes are in pixel coordinates [0, 640] at this point
        scale_weights = compute_scale_aware_weights(target_bboxes)

        # Classification loss (weighted for positive instances)
        cls_weights = torch.ones_like(pred_scores)
        if fg_mask.sum():
            cls_weights = torch.where(
                fg_mask.unsqueeze(-1),
                scale_weights.unsqueeze(-1),
                cls_weights,
            )
        loss[1] = (self.bce(pred_scores, target_scores.to(dtype)) * cls_weights).sum() / target_scores_sum

        # Bbox and DFL loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            fg_scale_weights = scale_weights[fg_mask]
            loss[0], loss[2] = self.scale_bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                scale_weights=fg_scale_weights,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()
