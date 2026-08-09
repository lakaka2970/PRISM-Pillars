"""
Temporal Support Builder for STER (Paper Section 5.2).

Computes per-point temporal support scores s_i by finding the best current-frame
match for each historical point's predicted position, using Mahalanobis distance
with detached covariance to prevent co-adaptation with the reliability estimator.
"""

import torch
import torch.nn as nn

from ...utils.covariance_2d import pairwise_mahalanobis


class TemporalSupportBuilder(nn.Module):
    """
    Computes self-supervised support scores for reliability training.

    For each historical point i, with predicted position mu_i and covariance Sigma_i,
    finds the best matching current-frame point j:

        s_i = max_{j in P_t} exp(-0.5 * (p_j - mu_i)^T Sigma_bar^{-1} (p_j - mu_i))

    where Sigma_bar uses detached (fixed) covariance to prevent the reliability
    estimator and uncertainty predictor from jointly inflating uncertainty.
    """

    def __init__(
        self,
        use_learned_covariance=False,
        fixed_sigma_r=0.10,
        fixed_sigma_t=0.50,
        sigma_0=0.03,
        max_current_points=1024,
    ):
        """
        Args:
            use_learned_covariance: If True, uses detach()'ed learned covariance.
                                     If False, uses fixed sigma values.
            fixed_sigma_r: float, fixed radial std (m).
            fixed_sigma_t: float, fixed tangential std (m).
            sigma_0: float, minimum isotropic uncertainty.
            max_current_points: int, max number of current points to use (subsample).
        """
        super().__init__()
        self.use_learned_covariance = use_learned_covariance
        self.fixed_sigma_r = fixed_sigma_r
        self.fixed_sigma_t = fixed_sigma_t
        self.sigma_0 = sigma_0
        self.max_current_points = max_current_points

    def compute_support(
        self,
        mu,
        Sigma,
        current_points_xy,
        u_vectors=None,
    ):
        """
        Compute temporal support score for each historical point.

        Args:
            mu: (N_h, 2) float tensor, predicted BEV positions of historical points.
            Sigma: (N_h, 2, 2) float tensor, covariance matrices (detached).
            current_points_xy: (N_c, 2) float tensor, current frame BEV points.
            u_vectors: (N_h, 2) float tensor, optional LOS directions for fixed cov.

        Returns:
            s: (N_h,) float tensor, temporal support scores in [0, 1].
        """
        N_h = mu.shape[0]
        N_c = current_points_xy.shape[0]
        device = mu.device

        if N_h == 0 or N_c == 0:
            return torch.zeros(N_h, device=device)

        # Subsample current points for efficiency
        if N_c > self.max_current_points:
            idx = torch.randperm(N_c, device=device)[: self.max_current_points]
            current_points_xy = current_points_xy[idx]

        # Use fixed or detached covariance
        if self.use_learned_covariance and Sigma is not None:
            Sigma_used = Sigma.detach()
        else:
            # Build fixed anisotropic covariance
            Sigma_used = self._build_fixed_covariance(N_h, u_vectors, device)

        # Compute pairwise Mahalanobis distance
        # pairwise_mahalanobis expects Sigma_j per KEY point.
        # We have Sigma per QUERY (mu) point, so swap arguments and transpose.
        d2 = pairwise_mahalanobis(current_points_xy, mu, Sigma_used, sigma_c=self.sigma_0)
        d2 = d2.T  # (N_c, N_h) → (N_h, N_c)

        # Support score: exp(-0.5 * min_j d^2(i, j))
        min_d2, _ = d2.min(dim=-1)  # (N_h,)
        s = torch.exp(-0.5 * min_d2)

        return s

    def _build_fixed_covariance(self, N, u_vectors, device):
        """
        Build fixed anisotropic covariance matrices.

        Args:
            N: int, number of points.
            u_vectors: (N, 2) or None, LOS directions.
            device: torch.device.

        Returns:
            Sigma: (N, 2, 2) float tensor.
        """
        from ...utils.covariance_2d import build_anisotropic_covariance
        from ...utils.radar_geometry import compute_tangential_direction

        s_r = torch.full((N,), self.fixed_sigma_r, device=device)
        s_t = torch.full((N,), self.fixed_sigma_t, device=device)

        if u_vectors is None:
            # Default: x-direction as radial (forward-looking radar)
            u = torch.zeros(N, 2, device=device)
            u[:, 0] = 1.0
        else:
            u = u_vectors

        Sigma = build_anisotropic_covariance(s_r, s_t, u, sigma_0=self.sigma_0)
        return Sigma

    def forward(self, mu, Sigma, current_points_xy, u_vectors=None):
        return self.compute_support(mu, Sigma, current_points_xy, u_vectors)
