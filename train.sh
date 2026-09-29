#!/bin/bash
# Train ADGT on Human3.6M with CPN-detected 2D keypoints (Table 1 of the paper).
# Paper settings (Sec. 4.1): Adam, lr 8e-4 decayed by 0.95 every 2 epochs for CPN input
# (for ground-truth 2D input use --keypoints gt --lr 0.001 --lr_gamma 0.9).
# See README.md for data preparation.

python main.py \
    --data_path data \
    --keypoints cpn_ft_h36m_dbb \
    --checkpoint checkpoints/adgt_h36m_cpn \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.0008 \
    --lr_decay_epochs 2 \
    --lr_gamma 0.95 \
    --dim_model 96 \
    --n_layer 5 \
    --n_register 8 \
    --n_head 4 \
    --h_ca 4 \
    --mlp_ratio 4 \
    --dropout 0.1 \
    --attn_drop 0.0 \
    --drop_path 0.1 \
    --hops 1 2 3
