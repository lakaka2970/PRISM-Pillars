"""
RBB: Re-parameterizable BEV Backbone (Paper Section 9).

Three-stage backbone using RepDWC blocks with uniform channel strategy
inherited from RadarPillars.

Architecture (inherited from RadarPillars, paper Section 9.3):
    Stage 1: 3 RepDWC blocks, stride 1
    Stage 2: 5 RepDWC blocks, first block stride 2
    Stage 3: 5 RepDWC blocks, first block stride 2

Channels: [32, 32, 32] (fair) or [48, 48, 48] (accuracy)

Deploy mode:
    All RepDWC blocks are converted to single-path for inference.
    Supports toggling via config or post-training conversion.
"""

import torch
import torch.nn as nn

from .rep_dwc import RepDWCBlock, RepDWCDownsample


class RepBEVBackbone(nn.Module):
    """
    Three-stage RepDWC backbone for BEV feature extraction.

    Config:
        NUM_FILTERS: list of int, channels per stage. E.g., [32, 32, 32]
        LAYER_NUMS: list of int, blocks per stage. E.g., [3, 5, 5]
        LAYER_STRIDES: list of int, first-block stride per stage. E.g., [1, 2, 2]
        UPSAMPLE_STRIDES: list of int, deconv strides. E.g., [1, 2, 4]
        NUM_UPSAMPLE_FILTERS: list of int, deconv output channels. E.g., [32, 32, 32]
        DEPLOY_MODE: bool (default False)
    """

    def __init__(self, model_cfg, input_channels):
        """
        Args:
            model_cfg: EasyDict with backbone config.
            input_channels: int, input BEV feature channels.
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.deploy_mode = model_cfg.get('DEPLOY_MODE', False)

        layer_nums = model_cfg.LAYER_NUMS
        layer_strides = model_cfg.LAYER_STRIDES
        num_filters = model_cfg.NUM_FILTERS

        assert len(layer_nums) == len(layer_strides) == len(num_filters), \
            "LAYER_NUMS, LAYER_STRIDES, NUM_FILTERS must have same length"

        # Upsample config
        upsample_strides = model_cfg.get('UPSAMPLE_STRIDES', [])
        num_upsample_filters = model_cfg.get('NUM_UPSAMPLE_FILTERS', [])

        num_levels = len(layer_nums)
        c_in_list = [input_channels] + list(num_filters[:-1])

        # Build encoder blocks per stage
        self.blocks = nn.ModuleList()
        for stage_idx in range(num_levels):
            stage_blocks = []
            stage_stride = layer_strides[stage_idx]
            stage_c_in = c_in_list[stage_idx]
            stage_c_out = num_filters[stage_idx]

            for block_idx in range(layer_nums[stage_idx]):
                if block_idx == 0 and stage_stride == 2:
                    # First block of stage uses stride 2 (downsample)
                    block = RepDWCDownsample(stage_c_in, stage_c_out, deploy_mode=self.deploy_mode)
                elif block_idx == 0:
                    # First block handles in->out channel transition
                    block = RepDWCBlock(stage_c_in, stage_c_out, stride=1, deploy_mode=self.deploy_mode)
                else:
                    # Subsequent blocks: same in/out channels, stride 1
                    block = RepDWCBlock(stage_c_out, stage_c_out, stride=1, deploy_mode=self.deploy_mode)
                stage_blocks.append(block)

            self.blocks.append(nn.Sequential(*stage_blocks))

        # Build decoder (upsample) blocks
        self.deblocks = nn.ModuleList()
        if len(upsample_strides) > 0:
            for stage_idx in range(num_levels):
                stride = upsample_strides[stage_idx]
                up_channels = num_upsample_filters[stage_idx]

                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[stage_idx], up_channels,
                            stride, stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(up_channels, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride_int = int(round(1.0 / stride))
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[stage_idx], up_channels,
                            stride_int, stride=stride_int, bias=False
                        ),
                        nn.BatchNorm2d(up_channels, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        # Final upsampling if needed
        if len(upsample_strides) > num_levels:
            c_in = sum(num_upsample_filters)
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = sum(num_upsample_filters) if upsample_strides else num_filters[-1]

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features: (B, C_in, H, W) BEV pseudo-image.

        Returns:
            data_dict with:
                spatial_features_2d: (B, C_out, H, W) output features.
                spatial_features_Nx: intermediate multi-scale features.
        """
        spatial_features = data_dict['spatial_features']
        x = spatial_features
        ups = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            data_dict[f'spatial_features_{stride}x'] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        # Concatenate upsampled features
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        # Store multi-scale features for MDFEN neck
        # Block outputs are stored under 'spatial_features_{stride}x' keys
        # where stride is computed from the raw block output (BEFORE upsampling).
        # For stages 1, 2, 3: strides are 1, 2, 4 (or 1, 2, 2 for the standard config).
        if not hasattr(data_dict, 'multi_scale_features'):
            data_dict['multi_scale_features'] = {}
        # The data_dict already stores block outputs at correct stride keys.
        # Look them up by the strides we expect from LAYER_STRIDES:
        stored_strides = set()
        for i in range(len(self.blocks)):
            # Compute actual stride from the data_dict keys
            for key in [f'spatial_features_{s}x' for s in [1, 2, 4, 8]]:
                if key in data_dict:
                    stored_strides.add(int(key.split('_')[-1].rstrip('x')))
            # Use the stride sequence: cumulative product of first-block strides
            # Stage i is at stride = prod(stride_1 ... stride_i)
            if i == 0:
                stride_at_stage = self.model_cfg.LAYER_STRIDES[0]
            else:
                stride_at_stage = stride_at_stage * self.model_cfg.LAYER_STRIDES[i]
            key = f'spatial_features_{stride_at_stage}x'
            if key in data_dict:
                data_dict['multi_scale_features'][f'F{i+1}'] = data_dict[key]

        return data_dict

    def set_deploy_mode(self, deploy=True):
        """Toggle deploy mode for all RepDWC blocks."""
        self.deploy_mode = deploy
        for stage_blocks in self.blocks:
            for block in stage_blocks:
                if hasattr(block, 'deploy_mode'):
                    block.deploy_mode = deploy
                if hasattr(block, 'dwc_block') and hasattr(block.dwc_block, 'deploy_mode'):
                    block.dwc_block.deploy_mode = deploy
        return self
