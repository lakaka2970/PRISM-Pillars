"""
STER: Self-Supervised Temporal Evidence Reliability (Paper Section 5 / Module B).

Estimates q_i in [0, 1] for each historical point — NOT a foreground probability,
but a measure of whether the compensated historical point can serve as valid
temporal evidence for the current frame.

The estimator is trained with self-supervised pseudo-labels derived from
current-frame spatial support, without requiring additional point-level annotation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalReliabilityEstimator(nn.Module):
    """
    MLP-based reliability estimator.

    Architecture (from paper Section 5.1):
        Linear(in_dim, hidden) -> LayerNorm -> SiLU
        -> Linear(hidden, hidden) -> SiLU
        -> Linear(hidden, 1) -> Sigmoid

    Input features should include:
        - Historical point embedded features
        - Local density / RCS statistics
        - Compensated velocity magnitude
        - Time delta
        - Range
    """

    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        """
        Args:
            features: (N, in_channels) float tensor, per-point features.

        Returns:
            q: (N, 1) float tensor, reliability scores in [0, 1].
        """
        logits = self.net(features)
        q = torch.sigmoid(logits)
        return q


class ReliabilityLoss(nn.Module):
    """
    Combined reliability loss from paper Section 5.3:

        L_rel = L_FocalBCE + 0.2 * L_rank

    Where:
        - FocalBCE with ignore index for ambiguous samples
        - Ranking loss: max(0, m - q_i^+ + q_i^-) with m = 0.2

    Pseudo-labels:
        y_i = 1 if s_i > 0.6
        y_i = 0 if s_i < 0.2
        y_i = ignore otherwise
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        pos_threshold=0.6,
        neg_threshold=0.2,
        rank_margin=0.2,
        rank_weight=0.2,
        ignore_value=-1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.rank_margin = rank_margin
        self.rank_weight = rank_weight
        self.ignore_value = ignore_value

    def make_pseudo_labels(self, support_scores):
        """
        Convert temporal support scores into pseudo-labels.

        Args:
            support_scores: (N,) float tensor, s_i in [0, 1].

        Returns:
            labels: (N,) float tensor, 1/0/-1 (ignore).
        """
        labels = torch.full_like(support_scores, self.ignore_value)
        labels[support_scores > self.pos_threshold] = 1.0
        labels[support_scores < self.neg_threshold] = 0.0
        return labels

    def focal_bce_loss(self, q, targets):
        """
        Focal Binary Cross-Entropy with ignore support.

        Args:
            q: (N, 1) float tensor, predicted reliabilities.
            targets: (N,) float tensor, pseudo-labels (with ignore_value).

        Returns:
            loss: scalar float tensor.
        """
        mask = targets != self.ignore_value
        if mask.sum() == 0:
            return torch.tensor(0.0, device=q.device, requires_grad=True)

        q_valid = q[mask].view(-1)
        t_valid = targets[mask].view(-1)

        # Focal BCE — must run outside autocast (BCE is numerically unsafe in FP16)
        with torch.cuda.amp.autocast(enabled=False):
            bce = F.binary_cross_entropy(q_valid.float(), t_valid.float(), reduction='none')

        pt = torch.where(t_valid > 0.5, q_valid, 1.0 - q_valid)
        alpha_t = torch.where(t_valid > 0.5, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma

        loss = (focal_weight * bce).mean()
        return loss

    def ranking_loss(self, q, targets):
        """
        Pairwise ranking loss: encourages q^+ > q^-.

        L_rank = mean(max(0, margin - q^+ + q^-))

        Args:
            q: (N, 1) float tensor, predicted reliabilities.
            targets: (N,) float tensor, pseudo-labels.

        Returns:
            loss: scalar float tensor.
        """
        pos_mask = targets == 1.0
        neg_mask = targets == 0.0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=q.device, requires_grad=True)

        q_pos = q[pos_mask].view(-1)
        q_neg = q[neg_mask].view(-1)

        # Randomly sample equal numbers for pairwise comparison
        n_pairs = min(q_pos.shape[0], q_neg.shape[0])
        if n_pairs > 1:
            pos_idx = torch.randperm(q_pos.shape[0], device=q.device)[:n_pairs]
            neg_idx = torch.randperm(q_neg.shape[0], device=q.device)[:n_pairs]
            q_pos_sampled = q_pos[pos_idx]
            q_neg_sampled = q_neg[neg_idx]
        else:
            q_pos_sampled = q_pos
            q_neg_sampled = q_neg

        margin_loss = F.relu(self.rank_margin - q_pos_sampled + q_neg_sampled)
        return margin_loss.mean()

    def forward(self, q, support_scores):
        """
        Compute the combined reliability loss.

        Args:
            q: (N, 1) float tensor, predicted reliabilities [0, 1].
            support_scores: (N,) float tensor, temporal support s_i.

        Returns:
            total_loss: scalar float tensor.
            loss_dict: dict with components for logging.
        """
        targets = self.make_pseudo_labels(support_scores)

        focal_loss = self.focal_bce_loss(q, targets)
        rank_loss = self.ranking_loss(q, targets)
        total_loss = focal_loss + self.rank_weight * rank_loss

        loss_dict = {
            'reliability_focal_loss': focal_loss.item(),
            'reliability_rank_loss': rank_loss.item(),
            'reliability_total_loss': total_loss.item(),
            'reliability_pos_ratio': (targets == 1.0).float().mean().item(),
            'reliability_neg_ratio': (targets == 0.0).float().mean().item(),
        }

        return total_loss, loss_dict


def compute_analytical_reliability(support_distance, doppler_error, eta1=1.0, eta2=1.0):
    """
    Analytical (non-learned) fallback reliability from paper Section 26.

    q_i = exp(-eta1 * d_i^support - eta2 * |e_i^doppler|)

    Used when learned reliability is unstable.

    Args:
        support_distance: (N,) float tensor, nearest neighbor distance to current points.
        doppler_error: (N,) float tensor, Doppler consistency error.
        eta1, eta2: float, scaling factors.

    Returns:
        q: (N,) float tensor, analytical reliability scores.
    """
    return torch.exp(-eta1 * support_distance - eta2 * torch.abs(doppler_error))
