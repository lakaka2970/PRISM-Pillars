"""
PRISMPillarsRF Detector (Paper Section 3, 15).

Full integration of PRISM-Pillars-RF following the Correct-then-Refine pipeline:

    1. Current-frame deterministic RadarPillars encoding
    2. Historical probabilistic evidence encoding (STER + DAUT + RAPR)
    3. Causal local temporal fusion (CRLF)
    4. Current BEV scatter
    5. RepDWC multi-scale backbone
    6. Lite-MDFEN foreground refinement
    7. Detection head (AnchorHead or CenterHead)

Total loss (Paper Section 12):
    L = L_det + lambda_rel * L_rel + lambda_sigma * L_sigma + lambda_inv * L_inv
"""

import torch
import torch.nn as nn

from .detector3d_template import Detector3DTemplate
from .. import backbones_2d, backbones_3d, dense_heads
from ..backbones_2d import map_to_bev
from ..radar_evidence import (
    RadarPointEmbedding,
    TemporalReliabilityEstimator,
    TemporalSupportBuilder,
    DopplerUncertaintyTube,
    ProbabilisticPillarRouter,
)
from ..temporal import CausalLocalPillarFusion
from ..radar_evidence.doppler_uncertainty_tube import UncertaintyRegularizer
from ..radar_evidence.temporal_reliability import ReliabilityLoss
from ...utils.loss_utils import CrossAugmentationConsistencyLoss


class PRISMPillarsRF(Detector3DTemplate):
    """
    PRISM-Pillars-RF: Physics-Guided Reliable Temporal Evidence Fusion
    with Re-parameterized Foreground Refinement.
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        # Extend module topology with new stages
        self.module_topology = [
            'vfe',
            'backbone_3d',
            'map_to_bev_module',
            'pfe',
            'backbone_2d',
            'neck',
            'dense_head',
            'point_head',
            'roi_head',
        ]

        # PRISM-specific modules (built separately from topology)
        self.prism_modules = nn.ModuleDict()

        self.module_list = self.build_networks()

    def build_networks(self):
        """Override to add PRISM-specific modules."""
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
        }

        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)

        # Build PRISM-specific modules
        self._build_prism_modules(model_info_dict)

        return model_info_dict['module_list']

    def _build_prism_modules(self, model_info_dict):
        """Build temporal evidence modules outside the standard topology."""
        cfg = self.model_cfg

        # Point embedding (shared between current and history)
        if cfg.get('POINT_FEATURES', None) is not None:
            self.prism_modules['point_embedding'] = RadarPointEmbedding(cfg.POINT_FEATURES)
        else:
            # Use VFE output dim as fallback
            self.prism_modules['point_embedding'] = None

        # Reliability estimator
        if cfg.get('RELIABILITY', {}).get('ENABLED', True):
            in_ch = cfg.RELIABILITY.get('HIDDEN_DIM', 32)
            self.prism_modules['reliability'] = TemporalReliabilityEstimator(
                in_channels=in_ch,
                hidden_dim=cfg.RELIABILITY.get('HIDDEN_DIM', 32),
            )
            self.prism_modules['support_builder'] = TemporalSupportBuilder(
                use_learned_covariance=False,
                fixed_sigma_r=0.10,
                fixed_sigma_t=0.50,
            )
        else:
            self.prism_modules['reliability'] = None
            self.prism_modules['support_builder'] = None

        # Doppler uncertainty tube
        if cfg.get('DOPPLER_TUBE', {}).get('ENABLED', True):
            self.prism_modules['doppler_tube'] = DopplerUncertaintyTube(cfg.DOPPLER_TUBE)
        else:
            self.prism_modules['doppler_tube'] = None

        # Probabilistic pillar router
        if cfg.get('PROBABILISTIC_ROUTING', None) is not None:
            self.prism_modules['prob_router'] = ProbabilisticPillarRouter(cfg.PROBABILISTIC_ROUTING)
            self.prism_modules['prob_router'].set_grid_params(
                voxel_size=model_info_dict['voxel_size'],
                pc_range=model_info_dict['point_cloud_range'],
                grid_shape=model_info_dict['grid_size'],
            )
        else:
            self.prism_modules['prob_router'] = None

        # Temporal fusion
        if cfg.get('TEMPORAL_FUSION', {}).get('ENABLED', True):
            self.prism_modules['temporal_fusion'] = CausalLocalPillarFusion(cfg.TEMPORAL_FUSION)
        else:
            self.prism_modules['temporal_fusion'] = None

        # Loss weights
        loss_cfg = cfg.get('LOSS', {})
        self.lambda_rel = loss_cfg.get('LAMBDA_REL', 0.20)
        self.lambda_sigma = loss_cfg.get('LAMBDA_SIGMA', 0.01)
        self.lambda_inv = loss_cfg.get('LAMBDA_INV', 0.05)

        # Pre-construct loss modules (avoid per-iteration imports)
        self._uncertainty_reg = UncertaintyRegularizer()
        self._reliability_loss_fn = ReliabilityLoss(
            pos_threshold=cfg.get('RELIABILITY', {}).get('POS_THRESHOLD', 0.60),
            neg_threshold=cfg.get('RELIABILITY', {}).get('NEG_THRESHOLD', 0.20),
            rank_margin=cfg.get('RELIABILITY', {}).get('RANK_MARGIN', 0.20),
            rank_weight=0.2,
        )
        self._consistency_loss_fn = CrossAugmentationConsistencyLoss()

        # Feature dimension reconciliation: ensure history pillar features
        # from RAPR match current pillar features from VFE+Attention.
        # If they differ, a projection layer will be created lazily on first forward.
        self._history_feat_proj = None  # Created lazily if dims differ

    def build_neck(self, model_info_dict):
        """Build Lite-MDFEN neck (between backbone_2d and dense_head)."""
        if self.model_cfg.get('LITE_MDFEN', {}).get('ENABLED', True):
            neck_module = backbones_2d.__all__['LiteMDFEN'](
                model_cfg=self.model_cfg.LITE_MDFEN,
            )
            model_info_dict['num_bev_features'] = neck_module.num_bev_features
            model_info_dict['module_list'].append(neck_module)
        else:
            neck_module = None
        return neck_module, model_info_dict

    def forward(self, batch_dict):
        """
        Main forward pass implementing paper Section 15.

        Args:
            batch_dict: dict with current_points, history_points, gt_boxes, etc.
        """
        # ============================================================
        # 1. Current-frame deterministic RadarPillars branch
        # ============================================================
        current_points = batch_dict.get('current_points', batch_dict.get('points'))

        # Standard processing through module list (VFE, Attention, etc.)
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)

        current_pillars = batch_dict['pillar_features']
        current_coords = batch_dict['voxel_coords']

        # ============================================================
        # 2. Historical probabilistic evidence branch
        # ============================================================
        history_points = batch_dict.get('history_points', None)
        history_dict = {}

        # Initialize with safe defaults in case history is unavailable
        q = None
        s_r = None
        s_t = None

        if history_points is not None and history_points.shape[0] > 0:
            history_points_tensor = history_points
            if not isinstance(history_points_tensor, torch.Tensor):
                history_points_tensor = torch.from_numpy(history_points).float().to(current_pillars.device)

            # collate_batch now concatenates (with batch_idx prefix) instead of stacking.
            # history_points: (sum N_i, D+1) with batch_idx at column 0.
            # history_points: (B, N_i, D) only if from old collate_batch (legacy path).
            if history_points_tensor.dim() == 3:
                # Legacy: stacked format (B, N_i, D) → take first sample
                history_points_tensor = history_points_tensor[0]
                # Extract original point data (no batch_idx prefix)
                history_points_raw = history_points_tensor
                batch_idx = torch.zeros(history_points_tensor.shape[0], dtype=torch.long, device=history_points_tensor.device)
            elif history_points_tensor.dim() == 2 and history_points_tensor.shape[1] >= 8:
                # New concatenated format: (sum N_i, 1+D_raw) with batch_idx at col 0
                batch_idx = history_points_tensor[:, 0].long()
                history_points_raw = history_points_tensor[:, 1:]  # Strip batch_idx
            else:
                # Unknown format, treat as raw points
                history_points_raw = history_points_tensor
                batch_idx = torch.zeros(history_points_tensor.shape[0], dtype=torch.long, device=history_points_tensor.device)

            delta_t = batch_dict.get('history_delta_t', None)
            if delta_t is not None:
                if not isinstance(delta_t, torch.Tensor):
                    delta_t = torch.from_numpy(delta_t).float().to(current_pillars.device)
                # Concatenated format: (sum N_i,) — no stripping needed.
                # Legacy stacked format: (B, N_i) → strip first dim.
                if delta_t.dim() == 2:
                    delta_t = delta_t[0]

            # 2a. Point embedding (shared with current frame)
            if self.prism_modules['point_embedding'] is not None:
                history_feat = self.prism_modules['point_embedding'](history_points_raw, delta_t)
            else:
                # Use raw points as features (fallback)
                history_feat = history_points_raw[:, :10]  # Take first 10 dims

            # 2b. Doppler uncertainty tube
            if self.prism_modules['doppler_tube'] is not None:
                mu, Sigma, s_r, s_t = self.prism_modules['doppler_tube'](
                    history_points_raw, delta_t=delta_t,
                    history_point_features=history_feat,
                )
            else:
                # Fixed uncertainty fallback
                mu = history_points_raw[:, :2]  # Use raw x,y as mean
                Sigma = None
                s_r = s_t = None

            # 2c. Reliability estimation
            if self.prism_modules['reliability'] is not None:
                q = self.prism_modules['reliability'](history_feat)
                # Compute self-supervised support scores for reliability training
                if self.training and self.prism_modules['support_builder'] is not None:
                    # Current frame points (before voxelization) for support matching
                    current_xy = current_points[:, :2] if isinstance(current_points, torch.Tensor) else \
                        batch_dict.get('points', torch.zeros(0, 2, device=history_feat.device))[:, :2]
                    support_scores = self.prism_modules['support_builder'](
                        mu=mu, Sigma=Sigma, current_points_xy=current_xy,
                        u_vectors=None,
                    )
                    # Store for loss computation
                    self._current_support_scores = support_scores
                    self._current_q = q
                else:
                    self._current_support_scores = None
                    self._current_q = None
            else:
                q = torch.ones(history_feat.shape[0], 1, device=history_feat.device)
                self._current_support_scores = None
                self._current_q = None

            # 2d. Probabilistic pillar routing
            if self.prism_modules['prob_router'] is not None:

                history_evidence = self.prism_modules['prob_router'](
                    point_features=history_feat,
                    mean=mu,
                    covariance=Sigma if Sigma is not None else torch.eye(2, device=history_feat.device).unsqueeze(0).expand(history_feat.shape[0], -1, -1),
                    reliability=q,
                    batch_idx=batch_idx,
                    delta_t=delta_t,
                )

                history_dict = {
                    'features': history_evidence['features'],
                    'coords': history_evidence['coords'],
                    'evidence_mass': history_evidence['evidence_mass'],
                    'reliability': history_evidence['pillar_reliability'],
                    'covariance': history_evidence['pillar_covariance'],
                    'delta_t': history_evidence['mean_delta_t'],
                }
            else:
                history_dict = {
                    'features': history_feat,
                    'coords': torch.zeros(0, 3, dtype=torch.long, device=history_feat.device),
                    'evidence_mass': torch.ones(history_feat.shape[0], device=history_feat.device),
                    'reliability': q.squeeze(-1),
                    'covariance': None,
                    'delta_t': delta_t if delta_t is not None else torch.zeros(history_feat.shape[0], device=history_feat.device),
                }
        else:
            history_dict = None

        # ============================================================
        # 3. Causal local temporal fusion
        # ============================================================
        if history_dict is not None and self.prism_modules['temporal_fusion'] is not None:
            # --- Feature dimension reconciliation ---
            # Ensure history pillar features match current pillar feature dims.
            # VFE+Attention output dim (current_pillars) and RAPR output dim
            # (history_dict['features']) should both be 32 (from config),
            # but we enforce this at runtime for safety.
            hist_feat_dim = history_dict['features'].shape[1]
            cur_feat_dim = current_pillars.shape[1]
            if hist_feat_dim != cur_feat_dim:
                if self._history_feat_proj is None or self._history_feat_proj.in_features != hist_feat_dim:
                    self._history_feat_proj = nn.Linear(
                        hist_feat_dim, cur_feat_dim,
                        device=current_pillars.device, dtype=current_pillars.dtype
                    )
                history_dict['features'] = self._history_feat_proj(history_dict['features'])

            fused_pillars, fusion_info = self.prism_modules['temporal_fusion'](
                current_features=current_pillars,
                current_coords=current_coords,
                history_features=history_dict['features'],
                history_coords=history_dict['coords'],
                evidence_mass=history_dict['evidence_mass'],
                reliability=history_dict['reliability'],
                covariance=history_dict['covariance'],
                delta_t=history_dict['delta_t'],
            )
            batch_dict['pillar_features'] = fused_pillars

        # ============================================================
        # 4. Current BEV scatter
        # ============================================================
        batch_dict = self.map_to_bev_module(batch_dict)

        # ============================================================
        # 5. RepDWC multi-scale backbone
        # ============================================================
        batch_dict = self.backbone_2d(batch_dict)

        # ============================================================
        # 6. Lite-MDFEN foreground refinement
        # ============================================================
        if hasattr(self, 'neck') and self.neck is not None:
            # RepBEVBackbone stores multi-scale features in data_dict
            ms_features = batch_dict.get('multi_scale_features', None)
            if ms_features is not None:
                neck_out = self.neck(ms_features)
            else:
                # Fallback: extract multi-scale features from backbone data_dict
                # RepBEVBackbone stores per-stride features as 'spatial_features_Nx'
                ms_features = {}
                for key_prefix, target_key in [('spatial_features_1x', 'F1'), ('spatial_features_2x', 'F2'), ('spatial_features_4x', 'F3')]:
                    if key_prefix in batch_dict:
                        ms_features[target_key] = batch_dict[key_prefix]
                if len(ms_features) == 3:
                    neck_out = self.neck(ms_features)
                else:
                    # If backbone didn't store features, skip neck
                    neck_out = batch_dict['spatial_features_2d']
            batch_dict['spatial_features_2d'] = neck_out

        # ============================================================
        # 7. Detection head
        # ============================================================
        batch_dict = self.dense_head(batch_dict)

        # ============================================================
        # Loss computation (training)
        # ============================================================
        if self.training:
            loss, tb_dict, disp_dict = self._compute_total_loss(batch_dict, q, s_r, s_t)
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def _compute_total_loss(self, batch_dict, q=None, s_r=None, s_t=None):
        """
        Compute total training loss (Paper Section 12).

        L = L_det + lambda_rel * L_rel + lambda_sigma * L_sigma + lambda_inv * L_inv

        Where:
            L_det = detection head loss (AnchorHead or CenterHead)
            L_rel = FocalBCE(q, pseudo_labels) + 0.2 * RankingLoss(q, pseudo_labels)
            L_sigma = mean[max(0, s_r - s_r_max) + max(0, s_t - s_t_max) + max(0, s_r - s_t)]
            L_inv = cross-augmentation feature consistency loss

        Returns:
            total_loss, tb_dict, disp_dict
        """
        disp_dict = {}
        total_loss = torch.tensor(0.0, device=batch_dict['spatial_features_2d'].device, requires_grad=True)
        tb_dict = {}

        # ================================================================
        # 1. Detection loss
        # ================================================================
        loss_det, tb_dict_det = self.dense_head.get_loss()
        total_loss = total_loss + loss_det
        tb_dict.update({f'det_{k}': v for k, v in tb_dict_det.items()})

        # ================================================================
        # 2. Reliability loss (Paper Section 5.3)
        # ================================================================
        if (self.lambda_rel > 0 and
            hasattr(self, '_current_q') and self._current_q is not None and
            hasattr(self, '_current_support_scores') and self._current_support_scores is not None):

            loss_rel, rel_loss_dict = self._reliability_loss_fn(
                q=self._current_q,
                support_scores=self._current_support_scores,
            )
            total_loss = total_loss + self.lambda_rel * loss_rel
            tb_dict.update({f'rel_{k}': v for k, v in rel_loss_dict.items()})

            # Clear stored tensors to free memory
            self._current_q = None
            self._current_support_scores = None

        # ================================================================
        # 3. Uncertainty regularization (Paper Section 12)
        # ================================================================
        if s_r is not None and s_t is not None and self.lambda_sigma > 0:
            loss_sigma = self._uncertainty_reg(s_r, s_t)
            total_loss = total_loss + self.lambda_sigma * loss_sigma
            tb_dict['loss_sigma'] = loss_sigma.item()

        # ================================================================
        # 4. Cross-augmentation consistency (Paper Section 12)
        # ================================================================
        if self.lambda_inv > 0:
            feat_a = batch_dict.get('spatial_features_2d', None)
            feat_b = batch_dict.get('spatial_features_2d_aug2', None)
            if feat_a is not None and feat_b is not None:
                loss_inv = self._consistency_loss_fn(feat_a, feat_b)
                total_loss = total_loss + self.lambda_inv * loss_inv
                tb_dict['loss_inv'] = loss_inv.item()

        tb_dict['loss_total'] = total_loss.item()

        return total_loss, tb_dict, disp_dict

    def get_training_loss(self):
        """Legacy method for compatibility with training loop."""
        disp_dict = {}
        loss_det, tb_dict_det = self.dense_head.get_loss()
        loss = loss_det
        tb_dict = {'loss_rpn': loss_det.item(), **tb_dict_det}
        return loss, tb_dict, disp_dict
