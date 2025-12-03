# Run torchrun with a single process (auto-sets RANK=0 and WORLD_SIZE=1)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 lib/train_val.py -e
