#!/usr/bin/env bash
export NCCL_DEBUG=INFO

     python -m torch.distributed.launch --nproc_per_node=2 train.py \
        --dataset gtav_jitter \
        --covstat_val_dataset gtav \
        --val_dataset cityscapes bdd100k mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.01 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 40000 \
        --bs_mult 4 \
        --gblur \
        --color_aug 0.5 \
        --relax_denom 0.0 \
        --wt_layer 0 0 1 1 1 0 0 \
        --use_ca \
        --use_cwcl \
        --nce_T 0.07 \
        --contrast_max_classes 15 \
        --contrast_max_view 50 \
        --jit_only \
        --use_sdcl \
        --w1 0.2 \
        --w2 0.2 \
        --w3 0.3 \
        --w4 0.3 \
        --num_patch 20 \
        --date 0326 \
        --exp BlindNet_r50os16_gtav \
        --ckpt ./logs/ \
        --tb_path ./logs/
