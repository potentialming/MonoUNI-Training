#!/bin/bash

# Train MonoUNI on KITTI format dataset

# export MASTER_PORT=29500

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training with KITTI config
nohup torchrun --standalone --nproc_per_node=8 lib/train_val.py --config config/config_kitti.yaml &> train_kitti.log&
