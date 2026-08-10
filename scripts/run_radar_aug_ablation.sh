#!/bin/bash
# Radar Aug Ablation Study (r1 report §8, 改良建议 P0)
#
# Three conditions to isolate which augmentation component causes
# the r0→r1 regression:
#   Abl-A: No Radar Aug at all
#   Abl-B: Geometric-only (RCS + angle noise), no Doppler/temporal perturbation
#   Abl-C: Half intensity (all params halved)
#
# Each: 30 epochs, bs=8, 1 seed (666), AMP bf16, phased training ON
# ETA (4090D): ~3h per run, ~9h total
#
# Output: compare eval at epoch 30 across A/B/C vs r0 baseline

set -e
cd /root/PRISM-Pillars
PYTHON=/root/miniconda3/bin/python
CFG=tools/cfgs/vod_models/prism_pillars_rf_s.yaml
# Note: --fix_random_seed must come BEFORE --set (argparse REMAINDER quirk)
COMMON_FLAGS="--fix_random_seed --eval_epoch_interval 10"

echo "========================================"
echo "Radar Aug Ablation Study"
echo "Started: $(date)"
echo "========================================"

# ============================================================
# Abl-A: No Radar Process Augmentation
# ============================================================
echo ""
echo "=== Abl-A: No Radar Aug (expect ~r0 mAP level if Aug is the cause) ==="
echo "  Started @ $(date)"
$PYTHON tools/train.py \
    --cfg_file $CFG --extra_tag abl_a_no_aug \
    $COMMON_FLAGS \
    --set \
        DATA_CONFIG.RADAR_AUG.ENABLED False \
        OPTIMIZATION.NUM_EPOCHS 30 \
        OPTIMIZATION.PHASED_TRAINING.ENABLED true

# ============================================================
# Abl-B: Geometric-only Aug (RCS + azimuth, no Doppler/temporal)
# Keeps: RCS scale/shift, angle noise
# Disables: range dropout, Doppler bias/scale, ego-comp noise, sweep dropout, ghost
# ============================================================
echo ""
echo "=== Abl-B: Geo-Only Aug (RCS + azimuth, no Doppler perturbation) ==="
echo "  Started @ $(date)"
$PYTHON tools/train.py \
    --cfg_file $CFG --extra_tag abl_b_geo_only \
    $COMMON_FLAGS \
    --set \
        DATA_CONFIG.RADAR_AUG.RANGE_DROPOUT.ENABLED False \
        DATA_CONFIG.RADAR_AUG.DOPPLER_BIAS_STD 0.0 \
        DATA_CONFIG.RADAR_AUG.EGO_COMP_NOISE_STD 0.0 \
        DATA_CONFIG.RADAR_AUG.SWEEP_DROPOUT_PROB 0.0 \
        DATA_CONFIG.RADAR_AUG.GHOST_PROB 0.0 \
        OPTIMIZATION.NUM_EPOCHS 30 \
        OPTIMIZATION.PHASED_TRAINING.ENABLED true

# ============================================================
# Abl-C: Half-intensity Radar Aug
# All parameters reduced to 50% of default values
# ============================================================
echo ""
echo "=== Abl-C: Half-Intensity Aug (all params ×0.5) ==="
echo "  Started @ $(date)"
$PYTHON tools/train.py \
    --cfg_file $CFG --extra_tag abl_c_half_aug \
    $COMMON_FLAGS \
    --set \
        DATA_CONFIG.RADAR_AUG.RCS_SCALE 0.85 1.15 \
        DATA_CONFIG.RADAR_AUG.RCS_SHIFT -0.5 0.5 \
        DATA_CONFIG.RADAR_AUG.RANGE_DROPOUT.BASE_PROB 0.025 \
        DATA_CONFIG.RADAR_AUG.RANGE_DROPOUT.FAR_GAIN 0.125 \
        DATA_CONFIG.RADAR_AUG.ANGLE_NOISE_STD 0.0015 \
        DATA_CONFIG.RADAR_AUG.DOPPLER_BIAS_STD 0.10 \
        DATA_CONFIG.RADAR_AUG.DOPPLER_SCALE 0.95 1.05 \
        DATA_CONFIG.RADAR_AUG.EGO_COMP_NOISE_STD 0.075 \
        DATA_CONFIG.RADAR_AUG.SWEEP_DROPOUT_PROB 0.10 \
        DATA_CONFIG.RADAR_AUG.GHOST_PROB 0.025 \
        OPTIMIZATION.NUM_EPOCHS 30 \
        OPTIMIZATION.PHASED_TRAINING.ENABLED true

echo ""
echo "========================================"
echo "Radar Aug Ablation complete: $(date)"
echo "========================================"
echo ""
echo "To evaluate:"
echo "  $PYTHON tools/test.py --cfg_file $CFG --eval_all --ckpt_dir output/cfgs/vod_models/prism_pillars_rf_s/abl_a_no_aug/ckpt --start_epoch 30"
echo "  $PYTHON tools/test.py --cfg_file $CFG --eval_all --ckpt_dir output/cfgs/vod_models/prism_pillars_rf_s/abl_b_geo_only/ckpt --start_epoch 30"
echo "  $PYTHON tools/test.py --cfg_file $CFG --eval_all --ckpt_dir output/cfgs/vod_models/prism_pillars_rf_s/abl_c_half_aug/ckpt --start_epoch 30"
