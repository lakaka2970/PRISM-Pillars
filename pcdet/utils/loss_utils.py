import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


# ---------------------------------------------------------------------------
# PRISM-Pillars-RF additional losses (Paper Section 5, 6, 12)
# ---------------------------------------------------------------------------


class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross-Entropy with ignore index support.

    Used for reliability training (Paper Section 5.3).
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_value=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_value = ignore_value

    def forward(self, pred, target):
        """
        Args:
            pred: (N,) or (N, 1) float tensor, predictions (after sigmoid).
            target: (N,) float tensor, targets with ignore_value for ambiguous.
        Returns:
            loss: scalar.
        """
        pred = pred.view(-1)
        target = target.view(-1)
        mask = target != self.ignore_value
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        p = pred[mask]
        t = target[mask]

        bce = F.binary_cross_entropy(p, t, reduction='none')
        pt = torch.where(t > 0.5, p, 1.0 - p)
        alpha_t = torch.where(t > 0.5, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma

        return (focal_weight * bce).mean()


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss for reliability estimator (Paper Section 5.3).

    L_rank = mean(max(0, margin - q^+ + q^-))
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, q, targets):
        """
        Args:
            q: (N,) float tensor, predicted reliabilities.
            targets: (N,) float tensor, 1 (positive) / 0 (negative) / -1 (ignore).
        Returns:
            loss: scalar.
        """
        pos_mask = targets == 1.0
        neg_mask = targets == 0.0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=q.device, requires_grad=True)

        q_pos = q[pos_mask]
        q_neg = q[neg_mask]

        # Sample pairs
        n_pairs = min(q_pos.shape[0], q_neg.shape[0])
        pos_idx = torch.randperm(q_pos.shape[0], device=q.device)[:n_pairs]
        neg_idx = torch.randperm(q_neg.shape[0], device=q.device)[:n_pairs]

        margin_loss = F.relu(self.margin - q_pos[pos_idx] + q_neg[neg_idx])
        return margin_loss.mean()


class UncertaintyRegularizationLoss(nn.Module):
    """
    Uncertainty regularization (Paper Section 12).

    L_sigma = mean[max(0, s_r - s_r_max)
                  + max(0, s_t - s_t_max)
                  + max(0, s_r - s_t)]
    """
    def __init__(self, s_r_max=0.60, s_t_max=2.00):
        super().__init__()
        self.s_r_max = s_r_max
        self.s_t_max = s_t_max

    def forward(self, s_r, s_t):
        loss = (
            F.relu(s_r - self.s_r_max).mean()
            + F.relu(s_t - self.s_t_max).mean()
            + F.relu(s_r - s_t).mean()
        )
        return loss


class CrossAugmentationConsistencyLoss(nn.Module):
    """
    Cross-augmentation invariance loss (Paper Section 12).

    L_inv = 1/|Omega| * sum_j ||norm(F_j^a) - stopgrad(norm(F_j^b))||^2

    Applied only on GT or high-confidence foreground regions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, feat_a, feat_b, mask=None):
        """
        Args:
            feat_a: (B, C, H, W) features from augmentation A.
            feat_b: (B, C, H, W) features from augmentation B.
            mask: (B, H, W) bool, foreground mask (optional).
        Returns:
            loss: scalar.
        """
        # Normalize features
        norm_a = F.normalize(feat_a, p=2, dim=1)
        norm_b = F.normalize(feat_b.detach(), p=2, dim=1)  # stopgrad on B

        diff = (norm_a - norm_b) ** 2

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            diff = diff * mask.float()
            loss = diff.sum() / mask.float().sum().clamp(min=1)
        else:
            loss = diff.mean()

        return loss


class DIoULoss(nn.Module):
    """
    Distance-IoU Loss for 3D rotated box regression (simplified).

    Used in CenterHead to improve box regression quality for rotated objects.
    Reference: RadarNeXt (Jia et al., 2025) — dIoU improves mAP ~0.5-1.0.

    Full implementation would require 3D rotated IoU (iou3d CUDA ops).
    This simplified version uses corner distance + L1 regularization as proxy.
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_boxes, gt_boxes, weights=None):
        """
        Args:
            pred_boxes: (N, 7) decoded boxes [x, y, z, l, w, h, yaw].
            gt_boxes: (N, 7) ground truth boxes.
            weights: (N,) optional per-box weights.

        Returns:
            loss: scalar, 1 - mean(exp(-beta * center_dist)) + L1_reg.
        """
        # Center distance penalty
        center_dist = torch.norm(pred_boxes[:, :3] - gt_boxes[:, :3], dim=-1)  # (N,)
        diou_penalty = 1.0 - torch.exp(-self.beta * center_dist)

        # L1 regularization on all box parameters
        l1_reg = F.l1_loss(pred_boxes, gt_boxes, reduction='none').mean(dim=-1)

        loss = diou_penalty + l1_reg

        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()

        return loss


# ---------------------------------------------------------------------------
# Ghost augmentation utility for reliability training robustness
# (Paper Section 5.3: BCE + ranking + ghost augmentation)
# ---------------------------------------------------------------------------

def apply_ghost_augmentation(history_points, ghost_ratio=0.05):
    """
    Inject ghost points into historical point cloud as data augmentation
    for reliability training robustness.

    Ghost points simulate false radar returns (multipath, noise, clutter).
    By exposing the reliability estimator to fake points during training,
    it learns to assign low q_i to unsupported returns.

    Args:
        history_points: (N, D) float tensor, historical radar points.
        ghost_ratio: float, fraction of total points to add as ghosts (0.05 = 5%).

    Returns:
        augmented_points: (N + M, D) float tensor.
        ghost_mask: (N + M,) bool tensor, True for ghost points.
    """
    N = history_points.shape[0]
    M = int(N * ghost_ratio)
    if M == 0:
        return history_points, torch.zeros(N, dtype=torch.bool, device=history_points.device)

    # Generate ghost points from random noise
    device = history_points.device
    dtype = history_points.dtype
    D = history_points.shape[1]

    # Sample ghost positions uniformly in point cloud range
    ghost = torch.randn(M, D, device=device, dtype=dtype)
    # Scale ghosts to match the distribution of real points
    if N > 0:
        real_mean = history_points[:, :3].mean(dim=0)
        real_std = history_points[:, :3].std(dim=0).clamp(min=0.1)
        ghost[:, :3] = ghost[:, :3] * real_std.unsqueeze(0) + real_mean.unsqueeze(0)
    # Random small values for other features
    ghost[:, 3:] *= 0.1

    augmented = torch.cat([history_points, ghost], dim=0)
    ghost_mask = torch.cat([
        torch.zeros(N, dtype=torch.bool, device=device),
        torch.ones(M, dtype=torch.bool, device=device),
    ], dim=0)

    return augmented, ghost_mask


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)
