"""
DAUT: Doppler-Aware Uncertainty Tube (Paper Section 6 / Module C).

Models the motion uncertainty of historical radar returns as anisotropic
Gaussians aligned with the Doppler-observable radial direction and the
poorly-observable tangential direction.

Key equations:
    mu_i = p_i + delta_t_i * v^{comp}_{r,i} * u_i
    s_{r,i} = sigma_{p,r} + |delta_t_i| * sigma_{v,r,i}
    s_{t,i} = sigma_{p,t} + |delta_t_i| * sigma_{v,t,i}
    Sigma_i = s_r^2 * u * u^T + s_t^2 * n * n^T + sigma_0^2 * I

Physical constraint: s_{t,i} >= s_{r,i} (tangential is worse-constrained).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.radar_geometry import (
    compute_line_of_sight_unit_vector,
    compute_tangential_direction,
)
from ...utils.covariance_2d import build_anisotropic_covariance, validate_covariance


class DopplerUncertaintyTube(nn.Module):
    """
    Predicts anisotropic motion uncertainty for historical radar points.

    Supports two modes:
        - Fixed: uses constant sigma values (paper Section 6, initial config)
        - Learnable: MLP predicts a_r, a_t for bounded parameterization
    """

    def __init__(self, model_cfg):
        """
        Args (from model_cfg.DOPPLER_TUBE):
            ENABLED: bool
            LEARNABLE: bool
            SIGMA_POSITION_BASE: float (default 0.03)
            SIGMA_R_MIN: float (default 0.03)
            SIGMA_R_MAX: float (default 0.60)
            SIGMA_T_MAX: float (default 2.00)
            FIXED_SIGMA_R_POSITION: float (default 0.10)
            FIXED_SIGMA_T_POSITION: float (default 0.50)
            HIDDEN_DIM: int (default 32)
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.enabled = model_cfg.get('ENABLED', True)
        self.learnable = model_cfg.get('LEARNABLE', True)

        # Base positional uncertainty (same for all points)
        self.sigma_p_base = model_cfg.get('SIGMA_POSITION_BASE', 0.03)
        self.sigma_r_min = model_cfg.get('SIGMA_R_MIN', 0.03)
        self.sigma_r_max = model_cfg.get('SIGMA_R_MAX', 0.60)
        self.sigma_t_max = model_cfg.get('SIGMA_T_MAX', 2.00)

        # Fixed mode defaults
        self.fixed_sigma_r_pos = model_cfg.get('FIXED_SIGMA_R_POSITION', 0.10)
        self.fixed_sigma_t_pos = model_cfg.get('FIXED_SIGMA_T_POSITION', 0.50)
        self.sigma_0 = model_cfg.get('SIGMA_POSITION_BASE', 0.03)

        # MLP for learnable mode
        # Input: [range, log_rcs, |v_comp|, |delta_t|, density]
        self.input_dim = 5
        self.hidden_dim = model_cfg.get('HIDDEN_DIM', 32)

        if self.learnable:
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2),  # [a_r, a_t]
            )

            # Initialize for near-zero output -> sigma near min values
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def _bounded_sigma(self, raw_a_r, raw_a_t):
        """
        Apply bounded parameterization from paper Section 6.2.

        sigma_{v,r} = sigma_{r,min} + (sigma_{r,max} - sigma_{r,min}) * sigmoid(a_r)
        sigma_{v,t} = sigma_{v,r} + (sigma_{t,max} - sigma_{v,r}) * sigmoid(a_t)

        This ensures:
            1. All sigmas are in valid ranges
            2. sigma_t >= sigma_r (physical constraint)

        Args:
            raw_a_r: (N,) float tensor, raw radial uncertainty logit.
            raw_a_t: (N,) float tensor, raw tangential uncertainty logit.

        Returns:
            sigma_v_r: (N,) float tensor, velocity radial std.
            sigma_v_t: (N,) float tensor, velocity tangential std.
        """
        sigma_v_r = self.sigma_r_min + (self.sigma_r_max - self.sigma_r_min) * torch.sigmoid(raw_a_r)
        sigma_v_t = sigma_v_r + (self.sigma_t_max - sigma_v_r) * torch.sigmoid(raw_a_t)
        # sigma_v_t is guaranteed >= sigma_v_r because sigmoid(a_t) >= 0
        return sigma_v_r, sigma_v_t

    def _build_input_features(self, history_points, history_point_features):
        """
        Build MLP input features from historical point data.

        Args:
            history_points: (N, D) float tensor, raw points.
            history_point_features: (N, F) float tensor, embedded features.

        Returns:
            mlp_input: (N, 5) float tensor.
        """
        device = history_points.device
        N = history_points.shape[0]

        x = history_points[:, 0]
        y = history_points[:, 1]
        r = torch.sqrt(x * x + y * y + 1e-8)

        rcs = history_points[:, 3] if history_points.shape[1] > 3 else torch.zeros(N, device=device)
        log_rcs = torch.log(rcs.clamp(min=1e-8))

        v_comp = history_points[:, 5] if history_points.shape[1] > 5 else torch.zeros(N, device=device)
        v_comp_abs = torch.abs(v_comp)

        # delta_t from feature or from input
        delta_t = history_points[:, 6] if history_points.shape[1] > 6 else torch.zeros(N, device=device)
        delta_t_abs = torch.abs(delta_t)

        # Density proxy: use distance to origin as rough measure
        density_proxy = 1.0 / (r + 1.0)

        mlp_input = torch.stack([
            r,
            log_rcs,
            v_comp_abs,
            delta_t_abs,
            density_proxy,
        ], dim=-1)

        return mlp_input

    def forward(self, history_points, delta_t=None, history_point_features=None):
        """
        Predict anisotropic uncertainty for historical points.

        Args:
            history_points: (N, D) float tensor, historical radar points.
                             Columns: [x, y, z, rcs, v_rel, v_comp, ...]
            delta_t: (N,) float tensor (optional), per-point time deltas.
            history_point_features: (N, F) float tensor (optional), embedded features.

        Returns:
            mu: (N, 2) float tensor, predicted BEV positions.
            Sigma: (N, 2, 2) float tensor, covariance matrices.
            sigma_r: (N,) float tensor, radial standard deviations.
            sigma_t: (N,) float tensor, tangential standard deviations.
        """
        N = history_points.shape[0]
        device = history_points.device
        dtype = history_points.dtype

        if N == 0:
            return (
                torch.zeros(0, 2, device=device, dtype=dtype),
                torch.zeros(0, 2, 2, device=device, dtype=dtype),
                torch.zeros(0, device=device, dtype=dtype),
                torch.zeros(0, device=device, dtype=dtype),
            )

        # Extract key fields
        x = history_points[:, 0]
        y = history_points[:, 1]
        v_comp = history_points[:, 5] if history_points.shape[1] > 5 else torch.zeros(N, device=device, dtype=dtype)

        # Time delta
        if delta_t is None:
            if history_points.shape[1] > 6:
                delta_t = history_points[:, 6]
            else:
                delta_t = torch.zeros(N, device=device, dtype=dtype)

        # Line-of-sight unit vector
        u = compute_line_of_sight_unit_vector(x, y)  # (N, 2)

        # Current BEV position after ego-motion alignment
        p_xy = torch.stack([x, y], dim=-1)  # (N, 2)

        # Deterministic radial prediction: mu = p + dt * v_comp * u
        dt = delta_t.view(N, 1)
        v_comp_exp = v_comp.view(N, 1)
        mu = p_xy + dt * v_comp_exp * u  # (N, 2)

        # --- Predict velocity uncertainty ---
        if self.learnable:
            mlp_input = self._build_input_features(history_points, history_point_features)
            raw = self.mlp(mlp_input)  # (N, 2)
            raw_a_r = raw[:, 0]
            raw_a_t = raw[:, 1]
            sigma_v_r, sigma_v_t = self._bounded_sigma(raw_a_r, raw_a_t)
        else:
            # Fixed mode
            sigma_v_r = torch.full((N,), 0.0, device=device, dtype=dtype)
            sigma_v_t = torch.full((N,), 0.0, device=device, dtype=dtype)

        # Time-dependent total standard deviations
        dt_abs = torch.abs(dt).view(N)
        if self.learnable:
            s_r = self.sigma_p_base + dt_abs * sigma_v_r
            s_t = self.sigma_p_base + dt_abs * sigma_v_t
        else:
            s_r = torch.full((N,), self.fixed_sigma_r_pos, device=device, dtype=dtype)
            s_t = torch.full((N,), self.fixed_sigma_t_pos, device=device, dtype=dtype)

        # Build anisotropic covariance
        Sigma = build_anisotropic_covariance(s_r, s_t, u, sigma_0=self.sigma_0)

        return mu, Sigma, s_r, s_t

    def get_fixed_sigmas(self, N, device, dtype=None):
        """Return fixed sigma values for deterministic baseline."""
        s_r = torch.full((N,), self.fixed_sigma_r_pos, device=device, dtype=dtype)
        s_t = torch.full((N,), self.fixed_sigma_t_pos, device=device, dtype=dtype)
        return s_r, s_t


class UncertaintyRegularizer(nn.Module):
    """
    Regularization loss for uncertainty parameters (Paper Section 12).

    L_sigma = mean[
        max(0, s_r - s_{r,max}) +     # prevent radial blow-up
        max(0, s_t - s_{t,max}) +     # prevent tangential blow-up
        max(0, s_r - s_t)             # enforce physical constraint
    ]
    """

    def __init__(self, s_r_max=0.60, s_t_max=2.00):
        super().__init__()
        self.s_r_max = s_r_max
        self.s_t_max = s_t_max

    def forward(self, s_r, s_t):
        """
        Args:
            s_r: (N,) float tensor, radial standard deviations.
            s_t: (N,) float tensor, tangential standard deviations.

        Returns:
            loss: scalar float tensor.
        """
        loss_r = F.relu(s_r - self.s_r_max).mean()
        loss_t = F.relu(s_t - self.s_t_max).mean()
        loss_constraint = F.relu(s_r - s_t).mean()
        return loss_r + loss_t + loss_constraint
