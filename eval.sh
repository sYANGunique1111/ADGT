#!/bin/bash
# Evaluate a trained checkpoint on Human3.6M (action-wise MPJPE / P-MPJPE, Tables 1-2).
# Usage: ./eval.sh [path/to/ckpt.pth.tar]

CKPT=${1:-checkpoints/adgt_h36m_cpn/ckpt_best.pth.tar}

python main.py \
    --data_path data \
    --keypoints cpn_ft_h36m_dbb \
    --evaluate "$CKPT" \
    --batch_size 256 \
    --dim_model 96 \
    --n_layer 5 \
    --n_register 8 \
    --n_head 4 \
    --h_ca 4 \
    --hops 1 2 3
