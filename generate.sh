CUDA_VISIBLE_DEVICES=0 python generate.py \
  --base_model '../../models/falcon-7b' \
  --adapter_model './output' \
  --load_4bit \
