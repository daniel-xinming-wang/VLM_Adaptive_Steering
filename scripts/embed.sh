#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_paths=(
  "Qwen/Qwen3-VL-4B-Thinking"
)

dataname="${1:-MathVMini}"
if [ "$dataname" = "MATH500" ]; then
  data_path="Data/Questions/test.jsonl"
  images_root=""
  save_path="Data/Representation/MATH500"
else
  data_path="Data/Questions/${dataname}.jsonl"
  images_root="Data/Images/MathV"
  save_path="Data/Representation/${dataname}"
fi

input_mode="gaussian_noise"
batch_size=2 ## Adjust if input_mode is "mixed"

for model_path in "${model_paths[@]}"; do
  echo "Processing model: ${model_path}"
  python embed.py \
    --reasoning True \
    --model_path "${model_path}" \
    --data_path "${data_path}" \
    --images_root "${images_root}" \
    --save_path "${save_path}" \
    --input_mode "${input_mode}" \
    --batch_size "${batch_size}"
done
