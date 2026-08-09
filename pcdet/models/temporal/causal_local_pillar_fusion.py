"""
CRLF: Causal Reliability-Aware Local Fusion (Paper Section 8 / Module E).

Current pillars act as queries, selectively attending to local historical
probability evidence using multi-head attention enriched with:
    - Mahalanobis geometric bias (spatial uncertainty awareness)
    - Reliability log-prior (down-weight unreliable history)
    - Evidence mass bonus (more evidence -> higher confidence)
    - Temporal decay (older history -> less relevant)

Outputs are gated with the original current features for stable training:
    F_i^{out} = F_i + g_i * H_hat_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..radar_evidence.temporal_support_builder import TemporalSupportBuilder


class CausalLocalPillarFusion(nn.Module):
    """
    Local multi-head attention fusion between current and historical pillars.

    Paper Eq. (Section 8.1):
        e_ij = (W_Q F_i)(W_K H_j)^T / sqrt(d)
             + b_ij^geo
             + alpha * log(q_bar_j + eps)
             + gamma * log(1 + m_j)
             - beta * |delta_t_j|

        a_ij = softmax_j(e_ij)
        H_hat_i = sum_j a_ij * W_V * H_j

    Paper Eq. (Section 8.2):
        g_i = sigmoid(MLP([F_i, H_hat_i, max_j a_ij, E_i, q_bar_i, m_i]))
        F_i^{out} = F_i + g_i * H_hat_i
    """

    def __init__(self, model_cfg):
        """
        Args (from model_cfg.TEMPORAL_FUSION):
            HIDDEN_DIM: int (default 64)
            NUM_HEADS: int (default 4)
            LOCAL_RADIUS: int (default 3)
            TOPK: int (default 16)
            RELIABILITY_ALPHA: float (default 1.0)
            EVIDENCE_MASS_GAMMA: float (default 0.5)
            TIME_DECAY_BETA: float (default 1.0)
            USE_MAHALANOBIS_BIAS: bool (default True)
            USE_GATE: bool (default True)
            ENABLED: bool (default True)
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.enabled = model_cfg.get('ENABLED', True)
        self.hidden_dim = model_cfg.get('HIDDEN_DIM', 64)
        self.num_heads = model_cfg.get('NUM_HEADS', 4)
        self.reliability_alpha = model_cfg.get('RELIABILITY_ALPHA', 1.0)
        self.evidence_mass_gamma = model_cfg.get('EVIDENCE_MASS_GAMMA', 0.5)
        self.time_decay_beta = model_cfg.get('TIME_DECAY_BETA', 1.0)
        self.use_mahalanobis_bias = model_cfg.get('USE_MAHALANOBIS_BIAS', True)
        self.use_gate = model_cfg.get('USE_GATE', True)

        assert self.hidden_dim % self.num_heads == 0, \
            f"HIDDEN_DIM ({self.hidden_dim}) must be divisible by NUM_HEADS ({self.num_heads})"
        self.head_dim = self.hidden_dim // self.num_heads

        # Input channels: will be set on first forward
        self.current_channels = None
        self.history_channels = None
        self._initialized = False

    def _initialize_weights(self, current_channels, history_channels, device, dtype):
        """Lazy initialization based on input channel dimensions."""
        self.current_channels = current_channels
        self.history_channels = history_channels

        # Query projection (from current features)
        self.W_Q = nn.Linear(current_channels, self.hidden_dim, device=device, dtype=dtype)

        # Key projection (from history features)
        self.W_K = nn.Linear(history_channels, self.hidden_dim, device=device, dtype=dtype)

        # Value projection (from history features)
        self.W_V = nn.Linear(history_channels, self.hidden_dim, device=device, dtype=dtype)

        # Gating MLP (if enabled)
        # Input: [F_i, H_hat_i, max_attn, entropy, q_bar_i, m_i]
        if self.use_gate:
            gate_input_dim = current_channels + self.hidden_dim + 4
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_input_dim, 32, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(32, 1, device=device, dtype=dtype),
            )

        self._initialized = True

    def _compute_attention_scores(
        self,
        F_i,
        H_j,
        b_geo,
        q_bar_j,
        m_j,
        delta_t_j,
        candidate_mask,
    ):
        """
        Compute multi-head attention scores with geometric/temporal priors.

        Args:
            F_i: (Q, C_c) float tensor, current pillar features.
            H_j: (K, C_h) float tensor, history pillar features.
            b_geo: (Q, K) float tensor, Mahalanobis bias.
            q_bar_j: (K,) float tensor, pillar average reliability.
            m_j: (K,) float tensor, pillar evidence mass.
            delta_t_j: (K,) float tensor, pillar mean time delta.
            candidate_mask: (Q, K) bool tensor.

        Returns:
            a_ij: (Q, K) float tensor, attention weights (row-wise softmax).
            H_hat_i: (Q, hidden_dim) float tensor, attended history features.
        """
        Q = F_i.shape[0]
        K = H_j.shape[0]
        device = F_i.device
        dtype = F_i.dtype

        if Q == 0 or K == 0:
            return (
                torch.zeros(Q, K, device=device, dtype=dtype),
                torch.zeros(Q, self.hidden_dim, device=device, dtype=dtype),
            )

        # Project to multi-head space
        Q_proj = self.W_Q(F_i).view(Q, self.num_heads, self.head_dim)  # (Q, h, d)
        K_proj = self.W_K(H_j).view(K, self.num_heads, self.head_dim)  # (K, h, d)
        V_proj = self.W_V(H_j).view(K, self.num_heads, self.head_dim)  # (K, h, d)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_scores = torch.einsum('qhd,khd->hqk', Q_proj, K_proj) / scale  # (h, Q, K)

        # Add geometric bias (broadcast across heads)
        if self.use_mahalanobis_bias and b_geo is not None:
            attn_scores = attn_scores + b_geo.unsqueeze(0)  # (h, Q, K)

        # Add temporal priors (broadcast across heads and queries)
        eps = 1e-8

        # Reliability prior: alpha * log(q_bar_j + eps)
        if self.reliability_alpha != 0:
            rel_prior = self.reliability_alpha * torch.log(q_bar_j + eps)  # (K,)
            attn_scores = attn_scores + rel_prior.view(1, 1, K)

        # Evidence mass prior: gamma * log(1 + m_j)
        if self.evidence_mass_gamma != 0:
            mass_prior = self.evidence_mass_gamma * torch.log(1.0 + m_j)  # (K,)
            attn_scores = attn_scores + mass_prior.view(1, 1, K)

        # Time decay: -beta * |delta_t_j|
        if self.time_decay_beta != 0:
            time_penalty = -self.time_decay_beta * torch.abs(delta_t_j)  # (K,)
            attn_scores = attn_scores + time_penalty.view(1, 1, K)

        # Mask invalid candidates
        if candidate_mask is not None:
            attn_scores = torch.where(
                candidate_mask.unsqueeze(0),
                attn_scores,
                torch.tensor(-1e9, device=device, dtype=dtype),
            )

        # Softmax over history dimension
        a_ij = F.softmax(attn_scores, dim=-1)  # (h, Q, K)

        # Weighted sum of values
        H_hat = torch.einsum('hqk,khd->qhd', a_ij, V_proj)  # (Q, h, d)
        H_hat = H_hat.reshape(Q, self.hidden_dim)  # (Q, hidden_dim)

        # Average attention across heads for per-pair analysis
        a_ij_avg = a_ij.mean(dim=0)  # (Q, K)

        return a_ij_avg, H_hat

    def _compute_gate(self, F_i, H_hat_i, a_ij, q_bar_i, m_i):
        """
        Compute per-pillar fusion gate.

        Args:
            F_i: (Q, C_c) float tensor.
            H_hat_i: (Q, hidden_dim) float tensor.
            a_ij: (Q, K) float tensor, attention weights.
            q_bar_i: (Q,) float tensor, current pillar avg reliability.
            m_i: (Q,) float tensor, current pillar evidence mass.

        Returns:
            g_i: (Q, 1) float tensor, fusion gate in [0, 1].
        """
        Q = F_i.shape[0]
        device = F_i.device
        dtype = F_i.dtype
        eps = 1e-8

        if not self.use_gate or Q == 0:
            return torch.ones(Q, 1, device=device, dtype=dtype)

        # Attention statistics
        max_attn = a_ij.max(dim=-1)[0] if a_ij.numel() > 0 else torch.zeros(Q, device=device, dtype=dtype)  # (Q,)
        entropy = -(a_ij * torch.log(a_ij + eps)).sum(dim=-1) if a_ij.numel() > 0 else torch.zeros(Q, device=device, dtype=dtype)  # (Q,)

        gate_input = torch.cat([
            F_i,                                # (Q, C_c)
            H_hat_i,                            # (Q, hidden_dim)
            max_attn.unsqueeze(-1),             # (Q, 1)
            entropy.unsqueeze(-1),              # (Q, 1)
            q_bar_i.unsqueeze(-1),              # (Q, 1)
            m_i.unsqueeze(-1),                  # (Q, 1)
        ], dim=-1)

        g_i = torch.sigmoid(self.gate_mlp(gate_input))  # (Q, 1)
        return g_i

    def forward(
        self,
        current_features,
        current_coords,
        history_features,
        history_coords,
        evidence_mass=None,
        reliability=None,
        covariance=None,
        delta_t=None,
    ):
        """
        Fuse historical probability evidence into current pillar features.

        Args:
            current_features: (Q, C_c) float tensor, current pillar features.
            current_coords: (Q, 3) long tensor, [batch, y, x].
            history_features: (K, C_h) float tensor, history pillar features.
            history_coords: (K, 3) long tensor, [batch, y, x].
            evidence_mass: (K,) float tensor, history pillar evidence mass m_j.
            reliability: (K,) float tensor, history pillar reliability q_bar_j.
            covariance: (K, 2, 2) float tensor, history pillar covariances.
            delta_t: (K,) float tensor, history pillar mean time deltas.

        Returns:
            fused_features: (Q, C_c) float tensor, updated current pillar features.
            fusion_dict: dict with attention weights, gates, etc. for monitoring.
        """
        Q = current_features.shape[0]
        K = history_features.shape[0]
        device = current_features.device
        dtype = current_features.dtype

        if not self.enabled or K == 0:
            fusion_dict = {
                'attention_weights': torch.zeros(Q, 0, device=device, dtype=dtype),
                'fusion_gate': torch.ones(Q, 1, device=device, dtype=dtype),
            }
            return current_features, fusion_dict

        C_c = current_features.shape[1]
        C_h = history_features.shape[1]

        # Lazy initialization
        if not self._initialized:
            self._initialize_weights(C_c, C_h, device, dtype)

        # Default values for optional inputs
        if evidence_mass is None:
            evidence_mass = torch.ones(K, device=device, dtype=dtype)
        if reliability is None:
            reliability = torch.ones(K, device=device, dtype=dtype)
        if delta_t is None:
            delta_t = torch.zeros(K, device=device, dtype=dtype)

        # Compute Mahalanobis bias
        if self.use_mahalanobis_bias and covariance is not None:
            from ...utils.covariance_2d import pairwise_mahalanobis
            # Convert pillar grid coords to approximate world coordinates.
            # Grid coords are integers [y_idx, x_idx]; multiply by voxel size
            # to get meter-scale coordinates matching the covariance units.
            # Default voxel size fallback: 0.16m (standard RadarPillars config).
            grid_to_world = self.model_cfg.get('VOXEL_SIZE', [0.16, 0.16]) if hasattr(self, 'model_cfg') else [0.16, 0.16]
            vy, vx = grid_to_world[0], grid_to_world[1]
            # voxel_coords format: [batch_idx, z, y, x]; take last 2 for BEV (y, x)
            c_i = current_coords[:, -2:].float() * torch.tensor([vy, vx], device=device, dtype=dtype)  # (Q, 2)
            c_j = history_coords[:, -2:].float() * torch.tensor([vy, vx], device=device, dtype=dtype)   # (K, 2)
            d2 = pairwise_mahalanobis(c_i, c_j, covariance, sigma_c=0.1)
            b_geo = -0.5 * d2  # (Q, K)
        else:
            b_geo = torch.zeros(Q, K, device=device, dtype=dtype)

        # Build candidate mask (all pairs for simplicity; local retrieval pre-filters)
        candidate_mask = torch.ones(Q, K, dtype=torch.bool, device=device)

        # === Attention ===
        a_ij, H_hat_i = self._compute_attention_scores(
            current_features,
            history_features,
            b_geo,
            reliability,
            evidence_mass,
            delta_t,
            candidate_mask,
        )

        # === Gate ===
        # Compute per-query aggregated statistics
        if a_ij.numel() > 0:
            attended_reliability = (a_ij * reliability.unsqueeze(0)).sum(dim=-1)  # (Q,)
            attended_mass = (a_ij * evidence_mass.unsqueeze(0)).sum(dim=-1)  # (Q,)
        else:
            attended_reliability = torch.zeros(Q, device=device, dtype=dtype)
            attended_mass = torch.zeros(Q, device=device, dtype=dtype)

        g_i = self._compute_gate(
            current_features,
            H_hat_i,
            a_ij,
            attended_reliability,
            attended_mass,
        )

        # === Residual fusion ===
        # Project H_hat_i to match current feature channels (if needed)
        if H_hat_i.shape[1] != C_c:
            if not hasattr(self, '_out_proj') or self._out_proj.in_features != H_hat_i.shape[1]:
                self._out_proj = nn.Linear(H_hat_i.shape[1], C_c, device=device, dtype=dtype)
            H_hat_proj = self._out_proj(H_hat_i)
        else:
            H_hat_proj = H_hat_i
        fused_features = current_features + g_i * H_hat_proj

        fusion_dict = {
            'attention_weights': a_ij,
            'fusion_gate': g_i,
            'attended_history': H_hat_i,
        }

        return fused_features, fusion_dict
