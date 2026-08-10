#!/bin/bash
# P1 Temporal Baseline Chain (converged_experiment_guide.md §3 P1)
# "全方案生死线" — must show monotonic improvement: #3 < #4 < #5
#
# Skip #1(naive) and #2(ego-motion) for now — use standard RadarPillars as proxy.
# Focus on the core PRISM chain:
#   #3: deterministic Doppler + hard assignment (sigma→0)
#   #4: isotropic Gaussian routing (sigma_r = sigma_t)
#   #5: fixed anisotropic routing (sigma_r=0.10, sigma_t=0.50)
#
# Each: 80 epochs, bs=16, AMP, RadarPillars backbone, q=1, no CRLF
# #3 and #5: 3 seeds (666, 42, 2023); #4: 2 seeds (666, 42)
# Expected GPU time (4090D): ~24h (run sequentially)

set -e
cd /root/PRISM-Pillars
PYTHON=/root/miniconda3/bin/python
CFG=tools/cfgs/vod_models/prism_pillars_p1.yaml
FLAGS="--fix_random_seed"

echo "========================================"
echo "P1 Temporal Baseline Chain"
echo "Started: $(date)"
echo "========================================"

# ============================================================
# P1#3: Deterministic Doppler (hard assignment, sigma → 0)
# ============================================================
echo ""
echo "=== P1#3: Deterministic Doppler compensation ==="
for seed in 666 42 2023; do
    echo "  Seed $seed @ $(date)"
    $PYTHON tools/train.py \
        --cfg_file $CFG --extra_tag p1_3_hard_s${seed} \
        $FLAGS \
        --set \
            MODEL.DOPPLER_TUBE.FIXED_SIGMA_R_POSITION 0.001 \
            MODEL.DOPPLER_TUBE.FIXED_SIGMA_T_POSITION 0.001 \
            MODEL.PROBABILISTIC_ROUTING.NEIGHBOR_SIZE 1
done

# ============================================================
# P1#4: Isotropic Gaussian routing (sigma_r == sigma_t)
# ============================================================
echo ""
echo "=== P1#4: Isotropic Gaussian routing ==="
for seed in 666 42; do
    echo "  Seed $seed @ $(date)"
    $PYTHON tools/train.py \
        --cfg_file $CFG --extra_tag p1_4_iso_s${seed} \
        $FLAGS \
        --set \
            MODEL.DOPPLER_TUBE.FIXED_SIGMA_R_POSITION 0.30 \
            MODEL.DOPPLER_TUBE.FIXED_SIGMA_T_POSITION 0.30
done

# ============================================================
# P1#5: Fixed anisotropic routing (sigma_r < sigma_t)
# ============================================================
echo ""
echo "=== P1#5: Anisotropic Gaussian routing ==="
for seed in 666 42 2023; do
    echo "  Seed $seed @ $(date)"
    $PYTHON tools/train.py \
        --cfg_file $CFG --extra_tag p1_5_aniso_s${seed} \
        $FLAGS \
        --set \
            MODEL.DOPPLER_TUBE.FIXED_SIGMA_R_POSITION 0.10 \
            MODEL.DOPPLER_TUBE.FIXED_SIGMA_T_POSITION 0.50
done

echo ""
echo "========================================"
echo "P1 Chain complete: $(date)"
echo "========================================"
echo ""
echo "Evaluation:"
echo "  $PYTHON tools/test.py --cfg_file $CFG --eval_all --ckpt_dir output/cfgs/vod_models/prism_pillars_p1/p1_3_hard_s666/ckpt --start_epoch 60"
echo "  $PYTHON tools/test.py --cfg_file $CFG --eval_all --ckpt_dir output/cfgs/vod_models/prism_pillars_p1/p1_4_iso_s666/ckpt --start_epoch 60"
echo "  $PYTHON tools/test.py --cfg_file $CFG --eval_all --ckpt_dir output/cfgs/vod_models/prism_pillars_p1/p1_5_aniso_s666/ckpt --start_epoch 60"
