#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}

     python -m torch.distributed.launch --nproc_per_node=1 --master_port=29401 infer.py \
        --val_dataset cityscapes mapillary bdd-100k \
        --arch network.deepv3.DeepR50V3PlusD \
        --wt_layer 0 0 1 1 1 0 0 \
        --mod blindnet \
        --results ${2} \
        --date 0101 \
        --exp blindnet_r50os16_gtav \
        --snapshot ${1}
