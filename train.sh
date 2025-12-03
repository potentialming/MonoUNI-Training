# Recommended
# torchrun --standalone --nproc_per_node=8 lib/train_val.py --config config/config.yaml

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# Run the Python script directly; it will use mp.spawn for multi-GPU
nohup torchrun --standalone --nproc_per_node=6 lib/train_val.py --config config/config.yaml &> monouni_train_eval.log&
