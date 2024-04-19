#!/usr/bin/env bash
echo "Running inference on" ${1}

     python -m torch.distributed.launch --nproc_per_node=1 --master_port=29400 valid.py \
        --val_dataset cityscapes bdd100k mapillary gtav \
        --arch network.deepv3.DeepR50V3PlusD \
        --wt_layer 0 0 1 1 1 0 0 \
        --date 0101 \
        --exp r50os16_gtav_blindnet \
        --snapshot ${1}
