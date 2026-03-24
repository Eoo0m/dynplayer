#!/bin/bash
cd /Users/eomjoonseo/dynplayer

echo "=========================================="
echo "CLIP Experiments - 50 epochs each"
echo "=========================================="

# Dot Product Experiments
echo ""
echo "[1/5] Dot Product - L2 reg=0.3"
echo "=========================================="
python clip_simgcl/train_dot.py \
    --output-dir clip_simgcl/exp_dot_reg03 \
    --l2-weight 0.3 \
    --epochs 50 \
    --test-ratio 0.1 \
    --eval-every 5

echo ""
echo "[2/5] Dot Product - L2 reg=0.2"
echo "=========================================="
python clip_simgcl/train_dot.py \
    --output-dir clip_simgcl/exp_dot_reg02 \
    --l2-weight 0.2 \
    --epochs 50 \
    --test-ratio 0.1 \
    --eval-every 5

echo ""
echo "[3/5] Dot Product - L2 reg=0.1"
echo "=========================================="
python clip_simgcl/train_dot.py \
    --output-dir clip_simgcl/exp_dot_reg01 \
    --l2-weight 0.1 \
    --epochs 50 \
    --test-ratio 0.1 \
    --eval-every 5

# Cosine Experiments
echo ""
echo "[4/5] Cosine - Baseline (grad_scale=1.0)"
echo "=========================================="
python clip_simgcl/train.py \
    --output-dir clip_simgcl/exp_cos_baseline \
    --epochs 50 \
    --test-ratio 0.1 \
    --eval-every 5 \
    --playlist-grad-scale 1.0

echo ""
echo "[5/5] Cosine - Text Fast (grad_scale=0.1)"
echo "=========================================="
python clip_simgcl/train.py \
    --output-dir clip_simgcl/exp_cos_textfast \
    --epochs 50 \
    --test-ratio 0.1 \
    --eval-every 5 \
    --playlist-grad-scale 0.1

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - clip_simgcl/exp_dot_reg03/"
echo "  - clip_simgcl/exp_dot_reg02/"
echo "  - clip_simgcl/exp_dot_reg01/"
echo "  - clip_simgcl/exp_cos_baseline/"
echo "  - clip_simgcl/exp_cos_textfast/"
