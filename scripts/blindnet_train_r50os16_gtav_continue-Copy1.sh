#!/usr/bin/env bash
export NCCL_DEBUG=INFO

     python -m torch.distributed.launch --nproc_per_node=2 --master_port=29400 train.py \
        --dataset cityscapes \
        --covstat_val_dataset gtav \
        --val_dataset cityscapes bdd100k mapillary gtav \
        --arch network.deepv3_blind.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 0.0005 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 1\
        --bs_mult 4 \
        --gblur \
        --color_aug 0.5 \
        --relax_denom 0.0 \
        --wt_layer 0 0 1 1 1 0 0 \
        --date 0504 \
        --exp BlindNet_r50os16_cty_contrain \
        --ckpt ./logs/ \
        --tb_path ./logs/ \
        --snapshot ${1} \
