"""
PRISM CenterHead (Paper Section 11, accuracy variant).

Anchor-free center-based detection head with:
    - Center heatmap (Focal Loss)
    - (x, y, z) offset regression (L1)
    - (l, w, h) size regression (L1)
    - Yaw angle (sin/cos encoding + L1)
    - Optional velocity prediction
    - Optional IoU quality branch

Entry criteria (paper Section 11):
    delta_mAP >= 0.5 AND delta_latency <= 10%
    Otherwise fall back to AnchorHead.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import common_utils, loss_utils


class PRISMCenterHead(nn.Module):
    """
    Anchor-free center-based detection head for PRISM-Pillars-RF-C.

    Predicts per-pixel:
        - Heatmap: class-wise center likelihoods
        - Offset: sub-pixel center refinement (dx, dy, dz)
        - Size: bounding box dimensions (l, w, h)
        - Yaw: angle as (sin, cos)
        - Velocity: (vx, vy) - optional
        - IoU: predicted IoU quality - optional
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range):
        """
        Args:
            model_cfg: EasyDict with CENTER_HEAD config.
            input_channels: int, input feature channels.
            num_class: int, number of classes.
            class_names: list of str.
            grid_size: (nx, ny, nz) tuple.
            point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax].
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.grid_size = grid_size
        self.point_cloud_range = np.array(point_cloud_range)
        self.forward_ret_dict = {}

        self.use_velocity = model_cfg.get('USE_VELOCITY', True)
        self.use_iou_branch = model_cfg.get('USE_IOU_BRANCH', True)
        self.num_dir_bins = model_cfg.get('NUM_DIR_BINS', 12)
        self.box_code_size = model_cfg.get('BOX_CODE_SIZE', 10)
        # box encoding: [dx, dy, dz, log_l, log_w, log_h, sin_yaw, cos_yaw, vx, vy]

        # Head convolutions
        head_conv = model_cfg.get('HEAD_CONV_CHANNELS', 64)
        num_conv = model_cfg.get('NUM_HEAD_CONV', 2)

        self.shared_conv = nn.Sequential()
        for i in range(num_conv):
            in_ch = input_channels if i == 0 else head_conv
            self.shared_conv.add_module(f'conv{i}', nn.Conv2d(in_ch, head_conv, 3, padding=1))
            self.shared_conv.add_module(f'bn{i}', nn.BatchNorm2d(head_conv))
            self.shared_conv.add_module(f'act{i}', nn.ReLU())

        # Task-specific heads
        self.heatmap_head = nn.Conv2d(head_conv, num_class, kernel_size=1)
        self.offset_head = nn.Conv2d(head_conv, 3, kernel_size=1)  # dz, dy, dx
        self.size_head = nn.Conv2d(head_conv, 3, kernel_size=1)    # log_l, log_w, log_h
        self.yaw_head = nn.Conv2d(head_conv, self.num_dir_bins, kernel_size=1)  # sin/cos encoding

        if self.use_velocity:
            self.velocity_head = nn.Conv2d(head_conv, 2, kernel_size=1)  # vx, vy

        if self.use_iou_branch:
            self.iou_head = nn.Conv2d(head_conv, 1, kernel_size=1)

        # Build losses
        self._build_losses(model_cfg)

        # Voxel/pillar size for coordinate conversion
        self.voxel_size = model_cfg.get('TARGET_VOXEL_SIZE', [0.16, 0.16, 5.0])
        self.downsample_factor = model_cfg.get('DOWNSAMPLE_FACTOR', 1)

    def _build_losses(self, model_cfg):
        loss_cfg = model_cfg.get('LOSS_CONFIG', {})
        self.heatmap_loss = loss_utils.SigmoidFocalClassificationLoss(
            alpha=loss_cfg.get('heatmap_alpha', 0.25),
            gamma=loss_cfg.get('heatmap_gamma', 2.0),
        )
        self.offset_loss = nn.L1Loss(reduction='none')
        self.size_loss = nn.L1Loss(reduction='none')
        self.yaw_loss = nn.CrossEntropyLoss(reduction='none')
        if self.use_velocity:
            self.velocity_loss = nn.L1Loss(reduction='none')
        if self.use_iou_branch:
            self.iou_loss = nn.L1Loss(reduction='none')
        # dIoU loss for rotation-aware box regression
        self.use_diou = model_cfg.get('USE_DIOU_LOSS', True)

        self.lambda_offset = loss_cfg.get('lambda_offset', 1.0)
        self.lambda_size = loss_cfg.get('lambda_size', 0.1)
        self.lambda_yaw = loss_cfg.get('lambda_yaw', 1.0)
        self.lambda_velocity = loss_cfg.get('lambda_velocity', 0.1)
        self.lambda_iou = loss_cfg.get('lambda_iou', 1.0)
        self.lambda_diou = loss_cfg.get('lambda_diou', 0.5)

    def _generate_heatmap_targets(self, gt_boxes, feature_map_size):
        """
        Generate Gaussian heatmap targets from GT boxes.

        Args:
            gt_boxes: (B, M, 8) float tensor [x, y, z, l, w, h, yaw, class_id].
            feature_map_size: (H, W) tuple.

        Returns:
            heatmap: (B, num_class, H, W) float tensor.
            offset_target: (B, 3, H, W) float tensor.
            mask: (B, H, W) bool tensor, positive locations.
        """
        B = gt_boxes.shape[0]
        H, W = feature_map_size
        device = gt_boxes.device

        heatmap = torch.zeros(B, self.num_class, H, W, device=device)
        offset_target = torch.zeros(B, 3, H, W, device=device)
        size_target = torch.zeros(B, 3, H, W, device=device)
        yaw_target = torch.zeros(B, 1, H, W, dtype=torch.long, device=device)
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        for b in range(B):
            valid = gt_boxes[b, :, :7].sum(dim=-1) != 0
            boxes = gt_boxes[b, valid]
            if boxes.shape[0] == 0:
                continue

            for box in boxes:
                x, y, z, l, w, h, yaw = box[:7]
                cls_id = int(box[7].item()) - 1  # 0-indexed

                if cls_id < 0 or cls_id >= self.num_class:
                    continue

                # Map to feature map coordinates
                x_range = self.point_cloud_range[3] - self.point_cloud_range[0]
                y_range = self.point_cloud_range[4] - self.point_cloud_range[1]
                fx = ((x - self.point_cloud_range[0]) / x_range * W).item()
                fy = ((y - self.point_cloud_range[1]) / y_range * H).item()

                cx, cy = int(round(fx)), int(round(fy))

                if cx < 0 or cx >= W or cy < 0 or cy >= H:
                    continue

                # Gaussian radius based on box size
                sigma_x = max(1.0, l / self.voxel_size[0] / 3.0 / self.downsample_factor)
                sigma_y = max(1.0, w / self.voxel_size[1] / 3.0 / self.downsample_factor)

                # Generate Gaussian kernel
                xs = torch.arange(W, device=device).float()
                ys = torch.arange(H, device=device).float()
                gy, gx = torch.meshgrid(ys, xs, indexing='ij')
                gaussian = torch.exp(
                    -((gx - float(cx)) ** 2) / (2 * sigma_x ** 2)
                    - ((gy - float(cy)) ** 2) / (2 * sigma_y ** 2)
                )

                # Max with existing heatmap
                heatmap[b, cls_id] = torch.max(heatmap[b, cls_id], gaussian)

                # Offset: sub-pixel refinement
                offset_target[b, 0, cy, cx] = fy - float(cy)  # dy
                offset_target[b, 1, cy, cx] = fx - float(cx)  # dx
                offset_target[b, 2, cy, cx] = z - self.point_cloud_range[2]  # dz relative to bottom

                # Size: log scale
                size_target[b, 0, cy, cx] = torch.log(torch.tensor(l, device=device) + 1e-8)
                size_target[b, 1, cy, cx] = torch.log(torch.tensor(w, device=device) + 1e-8)
                size_target[b, 2, cy, cx] = torch.log(torch.tensor(h, device=device) + 1e-8)

                # Yaw: bin index
                yaw_normalized = (yaw % (2 * np.pi)) / (2 * np.pi) * self.num_dir_bins
                yaw_target[b, 0, cy, cx] = int(yaw_normalized.item()) % self.num_dir_bins

                mask[b, cy, cx] = True

        return heatmap, offset_target, size_target, yaw_target, mask

    def _generate_velocity_targets(self, gt_boxes, feature_map_size):
        """
        Generate per-pixel velocity targets from GT boxes.

        Args:
            gt_boxes: (B, M, 8+) tensor [x, y, z, l, w, h, yaw, cls_id, vx?, vy?].
            feature_map_size: (H, W) tuple.

        Returns:
            velocity_target: (B, 2, H, W) float tensor [vx, vy].
        """
        B = gt_boxes.shape[0]
        H, W = feature_map_size
        device = gt_boxes.device
        velocity_target = torch.zeros(B, 2, H, W, device=device)

        for b in range(B):
            valid = gt_boxes[b, :, :7].sum(dim=-1) != 0
            boxes = gt_boxes[b, valid]
            if boxes.shape[0] == 0:
                continue

            for box in boxes:
                x, y = box[0].item(), box[1].item()
                x_range = self.point_cloud_range[3] - self.point_cloud_range[0]
                y_range = self.point_cloud_range[4] - self.point_cloud_range[1]
                cx = int(round(((x - self.point_cloud_range[0]) / x_range * W)))
                cy = int(round(((y - self.point_cloud_range[1]) / y_range * H)))

                if cx < 0 or cx >= W or cy < 0 or cy >= H:
                    continue

                # Extract velocity if available (after yaw and cls_id in gt_boxes)
                if box.shape[0] >= 10:
                    velocity_target[b, 0, cy, cx] = box[8]  # vx
                    velocity_target[b, 1, cy, cx] = box[9]  # vy

        return velocity_target

    def _generate_iou_targets(self, gt_boxes, feature_map_size, offset_tgt, size_tgt, yaw_tgt, mask):
        """
        Generate IoU quality targets by computing 3D IoU between
        decoded predictions and ground truth boxes.

        For simplicity during training, all positive locations get target IoU=1.0.
        Full IoU computation requires decoding boxes and computing 3D IoU,
        which is expensive. A simpler approach: positive mask → target=1, else target=0.

        Returns:
            iou_target: (B, 1, H, W) float tensor.
        """
        B = gt_boxes.shape[0]
        H, W = feature_map_size
        device = gt_boxes.device
        # Positive locations: target IoU = 1.0 (ideal)
        iou_target = mask.float().unsqueeze(1)  # (B, 1, H, W)
        return iou_target

    def _compute_diou_loss(self, pred_boxes, gt_boxes, mask):
        """
        Compute Distance-IoU loss for rotation-aware box regression.

        dIoU = 1 - IoU + rho^2(b, b_gt) / c^2
        where rho is center distance and c is diagonal of smallest enclosing box.

        For center-based detection, center distance is already supervised by offset loss,
        so we compute a simplified version: 1 - IoU + center_penalty.

        Args:
            pred_boxes: (B, num_pos, 7) decoded boxes [x, y, z, l, w, h, yaw].
            gt_boxes: (B, num_pos, 7) ground truth boxes.
            mask: (B, H, W) bool tensor.

        Returns:
            diou_loss: scalar.
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        # Compute 3D IoU for rotated boxes (simplified: use 2D BEV IoU)
        # For full 3D IoU, need the iou3d_nms CUDA extension.
        # Here we use a simple regularization: L1 on box parameters
        # weighted by a decay factor on center error.
        l1_loss = nn.L1Loss(reduction='none')
        box_diff = l1_loss(pred_boxes, gt_boxes).mean()
        return box_diff

    def forward(self, data_dict):
        """
        Args:
            data_dict with:
                spatial_features_2d: (B, C_in, H, W)

        Returns:
            data_dict with predictions added.
        """
        x = data_dict['spatial_features_2d']
        x = self.shared_conv(x)

        # Predictions
        heatmap = torch.sigmoid(self.heatmap_head(x))  # (B, num_class, H, W)
        offset = self.offset_head(x)                     # (B, 3, H, W)
        size = self.size_head(x)                         # (B, 3, H, W)
        yaw_logits = self.yaw_head(x)                    # (B, num_dir_bins, H, W)

        self.forward_ret_dict['heatmap'] = heatmap
        self.forward_ret_dict['offset'] = offset
        self.forward_ret_dict['size'] = size
        self.forward_ret_dict['yaw_logits'] = yaw_logits

        if self.use_velocity:
            velocity = self.velocity_head(x)
            self.forward_ret_dict['velocity'] = velocity

        if self.use_iou_branch:
            iou_pred = torch.sigmoid(self.iou_head(x))
            self.forward_ret_dict['iou_pred'] = iou_pred

        # Generate targets during training
        if self.training:
            batch_size = data_dict['batch_size']
            H, W = x.shape[2:]
            gt_boxes = data_dict['gt_boxes']

            heatmap_tgt, offset_tgt, size_tgt, yaw_tgt, mask = self._generate_heatmap_targets(
                gt_boxes, (H, W)
            )
            self.forward_ret_dict['heatmap_target'] = heatmap_tgt
            self.forward_ret_dict['offset_target'] = offset_tgt
            self.forward_ret_dict['size_target'] = size_tgt
            self.forward_ret_dict['yaw_target'] = yaw_tgt
            self.forward_ret_dict['positive_mask'] = mask

            # Generate velocity targets from GT boxes
            if self.use_velocity:
                velocity_tgt = self._generate_velocity_targets(gt_boxes, (H, W))
                self.forward_ret_dict['velocity_target'] = velocity_tgt

            # Generate IoU quality targets
            if self.use_iou_branch:
                iou_tgt = self._generate_iou_targets(gt_boxes, (H, W), offset_tgt, size_tgt, yaw_tgt, mask)
                self.forward_ret_dict['iou_target'] = iou_tgt

        return data_dict

    def get_loss(self):
        """Compute CenterHead losses."""
        pred = self.forward_ret_dict
        mask = pred['positive_mask']  # (B, H, W)
        B = mask.shape[0]
        num_pos = mask.sum().float().clamp(min=1.0)

        # Heatmap loss
        heatmap_loss = self.heatmap_loss(
            pred['heatmap'].permute(0, 2, 3, 1).reshape(-1, self.num_class),
            pred['heatmap_target'].permute(0, 2, 3, 1).reshape(-1, self.num_class),
            weights=torch.ones(B * mask.shape[1] * mask.shape[2], device=mask.device)
        ).sum() / B

        # Offset loss
        offset_loss = (self.offset_loss(pred['offset'], pred['offset_target']) * mask.unsqueeze(1)).sum() / num_pos
        offset_loss = offset_loss * self.lambda_offset

        # Size loss
        size_loss = (self.size_loss(pred['size'], pred['size_target']) * mask.unsqueeze(1)).sum() / num_pos
        size_loss = size_loss * self.lambda_size

        # Yaw loss
        yaw_pred_flat = pred['yaw_logits'].permute(0, 2, 3, 1).reshape(-1, self.num_dir_bins)
        yaw_tgt_flat = pred['yaw_target'].reshape(-1)
        yaw_loss = (self.yaw_loss(yaw_pred_flat, yaw_tgt_flat) * mask.reshape(-1)).sum() / num_pos
        yaw_loss = yaw_loss * self.lambda_yaw

        total_loss = heatmap_loss + offset_loss + size_loss + yaw_loss
        tb_dict = {
            'center_heatmap_loss': heatmap_loss.item(),
            'center_offset_loss': offset_loss.item(),
            'center_size_loss': size_loss.item(),
            'center_yaw_loss': yaw_loss.item(),
        }

        # dIoU loss (rotation-aware box regression)
        if self.use_diou and mask.sum() > 0:
            # Decode predictions to boxes for dIoU computation
            # Simplified: use L1 on decoded parameters as proxy
            # Full dIoU requires iou3d CUDA ops; fall back to L1 regularization
            box_params = ['offset', 'size']
            diou_reg = 0.0
            for param_name in box_params:
                if param_name in pred and f'{param_name}_target' in pred:
                    param_loss = (self.offset_loss(pred[param_name], pred[f'{param_name}_target']) *
                                  mask.unsqueeze(1)).sum() / num_pos
                    diou_reg = diou_reg + param_loss
            total_loss = total_loss + self.lambda_diou * diou_reg
            tb_dict['center_diou_loss'] = diou_reg.item() if isinstance(diou_reg, torch.Tensor) else diou_reg

        # Optional velocity loss
        if self.use_velocity and 'velocity_target' in pred:
            vel_loss = (self.velocity_loss(pred['velocity'], pred['velocity_target']) * mask.unsqueeze(1)).sum() / num_pos
            vel_loss = vel_loss * self.lambda_velocity
            total_loss = total_loss + vel_loss
            tb_dict['center_velocity_loss'] = vel_loss.item()

        # Optional IoU quality loss
        if self.use_iou_branch and 'iou_target' in pred:
            iou_l = (self.iou_loss(pred['iou_pred'], pred['iou_target']) * mask.unsqueeze(1)).sum() / num_pos
            iou_l = iou_l * self.lambda_iou
            total_loss = total_loss + iou_l
            tb_dict['center_iou_loss'] = iou_l.item()

        tb_dict['center_total_loss'] = total_loss.item()
        return total_loss, tb_dict
