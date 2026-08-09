"""
Mahalanobis Geometric Bias for CRLF (Paper Section 8.1).

b_{ij}^{geo} = -0.5 * (c_i - c_j)^T (Sigma_j + sigma_c^2 * I)^{-1} (c_i - c_j)

This bias is added to the attention score to account for spatial uncertainty
of historical pillars — pillars with larger or more anisotropic uncertainty
receive lower similarity scores for distant current pillars.
"""

import torch
import torch.nn as nn

from ...utils.covariance_2d import pairwise_mahalanobis


class MahalanobisBias(nn.Module):
    """
    Computes Mahalanobis-based attention bias between current and history pillars.
    """

    def __init__(self, sigma_c=0.1, enabled=True):
        """
        Args:
            sigma_c: float, additional isotropic uncertainty added to diagonal.
            enabled: bool, whether to compute bias (can disable for ablation).
        """
        super().__init__()
        self.sigma_c = sigma_c
        self.enabled = enabled

    def forward(self, current_centers, history_centers, history_covariances, candidate_mask):
        """
        Args:
            current_centers: (Q, 2) float tensor, current pillar BEV centers.
            history_centers: (K, 2) float tensor, history pillar BEV centers.
            history_covariances: (K, 2, 2) float tensor, history pillar covariances.
            candidate_mask: (Q, K, bool) tensor, True for valid candidate pairs.

        Returns:
            bias: (Q, K) float tensor, geometric attention bias.
                  Invalid pairs are set to very negative values.
        """
        if not self.enabled:
            return torch.zeros(
                current_centers.shape[0],
                history_centers.shape[0],
                device=current_centers.device,
                dtype=current_centers.dtype,
            )

        Q = current_centers.shape[0]
        K = history_centers.shape[0]

        if Q == 0 or K == 0:
            return torch.zeros(Q, K, device=current_centers.device, dtype=current_centers.dtype)

        # Compute pairwise Mahalanobis distances
        d2 = pairwise_mahalanobis(
            current_centers,
            history_centers,
            history_covariances,
            sigma_c=self.sigma_c,
        )  # (Q, K)

        bias = -0.5 * d2

        # Mask invalid pairs
        bias = torch.where(candidate_mask, bias, torch.tensor(-1e9, device=bias.device, dtype=bias.dtype))

        return bias
