#!/bin/bash

# Default parameter values
model_path=${1:-"Qwen/Qwen3-VL-4B-Thinking"}
vote_num=${2:-1}
dataname=${3:-"MathVMini"}
tensor_parallel_size=${4:-1}
steering_strength=${5:--0.5}
steering_vector_type=${6:-"all"}
adaptive_steering=${7:-"False"}
adaptive_reduce=${8:-"last"}
adaptive_eps=${9:-"1e-6"}
adaptive_alpha=${10:-"0.0"}
cot_style=${11:-"concise"}
calibration_vector_path=${12:-"./Data/Representation/MathVMini/Qwen3-VL-4B-Thinking_teacher_forced_think/calibration_vectors.npy"}

model_basename=$(basename "$model_path")
# base_save_path="./Data/Eval/${dataname}/${model_basename}_cos_adaptive"
# base_save_path="./Data/Eval/${dataname}/${model_basename}_concise"
base_save_path="./Data/Eval/${dataname}/${model_basename}_teacher_forcing_concise_triple_layer"
# base_save_path="./Data/Eval/${dataname}/${model_basename}_teacher_forcing_single_layer"
overall_trend_save_path="${base_save_path}/overall_trend_results_${steering_vector_type}_vote_num${vote_num}.json"
generation_save_path="${base_save_path}/${dataname}-${model_basename}_${steering_strength}_${steering_vector_type}_eval_vote_num${vote_num}_${adaptive_alpha}.json"

if [ "$dataname" = "MATH500" ]; then
  data_path="./Data/Questions/test.jsonl"
  images_root=""
else
  data_path="./Data/Questions/${dataname}.jsonl"
  images_root="./Data/Images/MathV"
fi

# Determine steering directory name without using associative arrays
case "$model_basename" in
  "Qwen3-VL-2B-Thinking"|"Qwen3-VL-4B-Thinking"|"Qwen3-VL-8B-Thinking"|"Qwen3-VL-30B-A3B-Thinking"|"Qwen3-VL-235B-A22B-Thinking")
    mapped_name="$model_basename"
    ;;
  *)
    mapped_name="$model_basename"
    ;;
esac

# steering_vector_path="./Assets/MathV/${mapped_name}/mean_steering_vectors_${steering_vector_type}.npy"
# steering_vector_path="./Assets/MATH500/${mapped_name}/mean_steering_vectors_all.npy"
steering_vector_path="./Data/Representation/steering_vectors_teacher_forcing.npy"

# Disable steering if strength is zero (and not adaptive) or file missing
adaptive_flag="$(echo "${adaptive_steering}" | tr '[:upper:]' '[:lower:]')"
if { [ "${steering_strength}" = "0" ] || [ "${steering_strength}" = "0.0" ]; } \
  && [ "${adaptive_flag}" != "true" ] && [ "${adaptive_flag}" != "1" ]; then
  echo "[info] Steering disabled (strength=${steering_strength}, adaptive=${adaptive_steering}, path=${steering_vector_path})."
  steering_vector_path="Empty"
fi
if [ ! -f "${steering_vector_path}" ]; then
  echo "[info] Steering disabled (missing path=${steering_vector_path})."
  steering_vector_path="Empty"
fi

echo "Running generator.py with following parameters:"
echo "Data path: $data_path"
echo "Model path: $model_path"
echo "Generation save path: $generation_save_path"
echo "Vote num: $vote_num"
echo "Dataset name: $dataname"
echo "Tensor parallel size: $tensor_parallel_size"
echo "Steering vector path: $steering_vector_path"
echo "Steering strength: $steering_strength"
echo "Adaptive steering: $adaptive_steering"
echo "Adaptive alpha/reduce/eps: $adaptive_alpha / $adaptive_reduce / $adaptive_eps"
echo "Images root: $images_root"
#echo "Images root: ./Data/Images/${dataname}"
echo "CoT style: $cot_style"
echo "Calibration vector path: $calibration_vector_path"

python generate_and_evaluate_v.py \
  --dataname "$dataname" \
  --data_path "$data_path" \
  --model_path "$model_path" \
  --vote_num "$vote_num" \
  --tensor_parallel_size "$tensor_parallel_size" \
  --base_save_path "$base_save_path" \
  --generation_save_path "$generation_save_path" \
  --overall_trend_save_path "$overall_trend_save_path" \
  --steering_vector_path "$steering_vector_path" \
  --steering_strength "$steering_strength" \
  --adaptive_steering "$adaptive_steering" \
  --adaptive_alpha "$adaptive_alpha" \
  --adaptive_reduce "$adaptive_reduce" \
  --adaptive_eps "$adaptive_eps" \
  --images_root "$images_root" \
  --outputs_dir "./Data/Eval/${dataname}/Outputs" \
  --run_name "${model_basename}_${dataname}_multimodal" \
  --cot_style "$cot_style" \
  --calibration_vector_path "$calibration_vector_path"

