python scripts/eval_onnx.py \
    --config config/config_kitti.yaml \
    --onnx checkpoints/dair_v2x_i_epoch_150.onnx \
    --device cuda \
    --batch-size 8 \
    --threshold 0.2
