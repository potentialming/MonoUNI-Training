python scripts/export_onnx.py \
  --config config/config_kitti.yaml \
  --checkpoint ./checkpoints/dair_v2x_i_epoch_150.pth \
  --input-size 512,960 \
  --opset-version 13