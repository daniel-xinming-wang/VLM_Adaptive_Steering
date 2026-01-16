model_paths=(
    "Qwen/Qwen3-VL-4B-Thinking"
)

vote_num=1
# dataname='MMLU'
# dataname='GPQA_diamond'
# dataname="MATH500"
# dataname="AIME2024"
# dataname="OlympiadBench"
# dataname="Minerva"
dataname="MathVMini"
tensor_parallel_size=1
#steering_strengths=(0.0 -0.05 -0.1 -0.15 -0.2 0.05 0.1 0.15 0.2)
steering_strengths=(-0.1 0.1)
# steering_strengths=(0.0 -0.05 -0.1 -0.15 -0.2)
# steering_strengths=(-0.05)
# steering_strengths=(0.15)
#steering_vector_type="correct"
steering_vector_type="all"


for i in "${!model_paths[@]}"; do
    model_path="${model_paths[i]}"
    model_name=$(basename "$model_path")
    echo "Model name: $model_name"
    for steering_strength in "${steering_strengths[@]}"; do
        echo "Waiting for previous process to fully exit..."
        sleep 10  # Wait for 10 seconds to ensure that the previous process has completely exited.
        bash scripts/generate_and_evaluate.sh "$model_path" "$vote_num" "$dataname" "$tensor_parallel_size" "$steering_strength" "$steering_vector_type"
    done
done
