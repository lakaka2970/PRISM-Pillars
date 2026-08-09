"""
SR-MDFEN: Single-Deformable Raw-Bypass MDFEN (Paper Section 10).

Lite-MDFEN preserves two key design principles from RadarNeXt ablations:
    1. Only ONE DCNv3 layer (at high-resolution raw features)
    2. Untouched raw feature bypass prevents information loss

Fusion topology (paper Section 10.2):
    T2 = RepDWC[Concat(F2, Up(F3))]
    E1 = DCNv3(F1)                     <- single DCNv3 on high-res raw
    T1 = RepDWC[Concat(F1, E1, Up(T2))] <- raw bypass + enhanced + upsampled
    B2 = RepDWC[Concat(T2, Down(T1))]   <- bottom-up feedback

    F_neck = Concat[T1, Up(B2), Up(F3)] -> 1x1 Conv -> output

Where:
    F1: (B, C, H, W)      - high-resolution (backbone stage 1)
    F2: (B, C, H/2, W/2)  - mid-resolution (backbone stage 2)
    F3: (B, C, H/4, W/4)  - low-resolution (backbone stage 3)

Without DCNv3, falls back to multi-path without deformable (paper ablation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rep_dwc import RepDWCBlock
from .dcnv3_wrapper import DCNv3Wrapper, has_dcnv3_support


class LiteMDFEN(nn.Module):
    """
    Single-Deformable Raw-Bypass Multi-scale Feature Enhancement Neck.

    Config (from model_cfg.LITE_MDFEN):
        ENABLED: bool (default True)
        CHANNELS: int (default 32)
        USE_SINGLE_DCNV3: bool (default True)
        DCN_KERNEL_SIZE: int (default 3)
        DCN_GROUPS: int (default 4)
        DCN_PATH: str, 'HIGH_RES_RAW_FEATURE' (default)
        PRESERVE_RAW_BYPASS: bool (default True)
        FUSION_BLOCK: str, 'RepDWC' (default)
        OUTPUT_CHANNELS: int (default 96)
    """

    def __init__(self, model_cfg):
        """
        Args:
            model_cfg: EasyDict with LITE_MDFEN config.
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.enabled = model_cfg.get('ENABLED', True)
        self.channels = model_cfg.get('CHANNELS', 32)
        self.use_single_dcnv3 = model_cfg.get('USE_SINGLE_DCNV3', True)
        self.dcn_kernel_size = model_cfg.get('DCN_KERNEL_SIZE', 3)
        self.dcn_groups = model_cfg.get('DCN_GROUPS', 4)
        self.preserve_raw_bypass = model_cfg.get('PRESERVE_RAW_BYPASS', True)
        self.output_channels = model_cfg.get('OUTPUT_CHANNELS', 96)
        self.deploy_mode = model_cfg.get('DEPLOY_MODE', False)

        # Check DCNv3 availability
        self._has_dcnv3 = has_dcnv3_support() and self.use_single_dcnv3

        # Channel dimensions (all same due to uniform scaling)
        C = self.channels

        # === T2 = RepDWC[Concat(F2, Up(F3))] ===
        self.t2_up = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(),
        )
        self.t2_fusion = RepDWCBlock(C * 2, C, stride=1, deploy_mode=self.deploy_mode)

        # === E1 = DCNv3(F1) ===
        if self._has_dcnv3:
            self.dcnv3 = DCNv3Wrapper(
                C, kernel_size=self.dcn_kernel_size,
                groups=self.dcn_groups,
            )
        else:
            # Fallback: RepDWC instead of DCNv3
            self.dcnv3 = RepDWCBlock(C, C, stride=1, deploy_mode=self.deploy_mode)

        # === T1 = RepDWC[Concat(F1, E1, Up(T2))] ===
        self.t1_up = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(),
        )
        t1_in_channels = C * 3 if self.preserve_raw_bypass else C * 2
        self.t1_fusion = RepDWCBlock(t1_in_channels, C, stride=1, deploy_mode=self.deploy_mode)

        # === B2 = RepDWC[Concat(T2, Down(T1))] ===
        self.b2_down = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(),
        )
        self.b2_fusion = RepDWCBlock(C * 2, C, stride=1, deploy_mode=self.deploy_mode)

        # === Final output ===
        # Concat[T1, Up(B2), Up(Up(F3))] -> 1x1 Conv -> output_channels
        # B2 at stride 2: Up → stride 1 (2×)
        # F3 at stride 4: Up → stride 2 → stride 1 (4×)
        self.out_b2_up = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(),
        )
        self.out_f3_up = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(),
            nn.ConvTranspose2d(C, C, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(),
        )
        self.out_compress = nn.Conv2d(C * 3, self.output_channels, kernel_size=1, bias=False)

        self.num_bev_features = self.output_channels

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: dict with keys 'F1', 'F2', 'F3'.
                F1: (B, C, H, W) high-res
                F2: (B, C, H/2, W/2) mid-res
                F3: (B, C, H/4, W/4) low-res

        Returns:
            neck_features: (B, output_channels, H, W) BEV features.
        """
        if not self.enabled:
            # Pass through: just concatenate upsampled features
            return self._passthrough(multi_scale_features)

        F1 = multi_scale_features['F1']
        F2 = multi_scale_features['F2']
        F3 = multi_scale_features['F3']

        # === T2: mid-resolution fusion ===
        F3_up = self.t2_up(F3)  # (B, C, H/2, W/2)
        T2 = self.t2_fusion(torch.cat([F2, F3_up], dim=1))  # (B, C, H/2, W/2)

        # === E1: single DCNv3 on high-res raw features ===
        E1 = self.dcnv3(F1)  # (B, C, H, W)

        # === T1: high-resolution fusion with raw bypass ===
        T2_up = self.t1_up(T2)  # (B, C, H, W)
        if self.preserve_raw_bypass:
            T1 = self.t1_fusion(torch.cat([F1, E1, T2_up], dim=1))  # (B, C, H, W)
        else:
            T1 = self.t1_fusion(torch.cat([E1, T2_up], dim=1))  # (B, C, H, W)

        # === B2: bottom-up feedback ===
        T1_down = self.b2_down(T1)  # (B, C, H/2, W/2)
        B2 = self.b2_fusion(torch.cat([T2, T1_down], dim=1))  # (B, C, H/2, W/2)

        # === Final output ===
        B2_up = self.out_b2_up(B2)  # (B, C, H, W)
        F3_up_final = self.out_f3_up(F3)  # (B, C, H, W)

        out = torch.cat([T1, B2_up, F3_up_final], dim=1)  # (B, C*3, H, W)
        out = self.out_compress(out)  # (B, output_channels, H, W)

        return out

    def _passthrough(self, multi_scale_features):
        """Simple FPN passthrough when MDFEN is disabled."""
        F1 = multi_scale_features['F1']
        F2 = multi_scale_features['F2']
        F3 = multi_scale_features['F3']

        # Simple upsample-concat
        H, W = F1.shape[2:]
        F2_up = F.interpolate(F2, size=(H, W), mode='bilinear', align_corners=False)
        F3_up = F.interpolate(F3, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([F1, F2_up, F3_up], dim=1)
        # Project to expected output channels if needed
        if out.shape[1] != self.output_channels:
            if not hasattr(self, '_passthrough_proj'):
                self._passthrough_proj = nn.Conv2d(out.shape[1], self.output_channels, 1, bias=False).to(out.device)
            out = self._passthrough_proj(out)
        return out
