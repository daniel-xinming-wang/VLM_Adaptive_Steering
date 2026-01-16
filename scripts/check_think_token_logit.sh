#!/bin/bash

# Define the model path array.
model_paths=(
    # "model/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen3-VL-4B-Thinking"
    # "Qwen/Qwen3-VL-8B-Thinking"
)

# Define the dataset.
# dataname="MATH500"
# dataname="GPQA_diamond"
# dataname="AIME2024"
# dataname="OlympiadBench"
# dataname="Minerva"
dataname="MathVMini"

# Define the problem file path + images root.
if [ "$dataname" = "MATH500" ]; then
    question_path="./Data/Questions/test.jsonl"
    images_root=""
else
    # Prefer .jsonl (this repo stores MathV as .jsonl); fall back to .json if needed.
    if [ -f "./Data/Questions/${dataname}.jsonl" ]; then
        question_path="./Data/Questions/${dataname}.jsonl"
    else
        question_path="./Data/Questions/${dataname}.json"
    fi
    images_root="./Data/Images/MathV"
fi

# Define the steering strength array.
# steering_strengths=(0.0 -0.05 -0.1 -0.15 -0.2 0.05 0.1 0.15 0.2)
# steering_strengths=(0.0 -0.1 -0.2 0.1 0.2)
steering_strengths=(-0.05 0.05)
# steering_strengths=(0.15)

# Traverse all models.
for i in "${!model_paths[@]}"; do
    model_path="${model_paths[i]}"
    model_name=$(basename "$model_path")
    echo "Processing model: $model_name"

    if [ "$dataname" = "MATH500" ]; then
        steering_vector_path="./Assets/MATH500/${model_name}/mean_steering_vectors_all.npy"
    else
        #steering_vector_path="./Assets/MathV/${model_name}/mean_steering_vectors_all.npy"
        steering_vector_path="./Data/Representation/steering_vectors.npy"
    fi
    
    # Traverse all steering strength.
    for steering_strength in "${steering_strengths[@]}"; do
        echo "Running with steering strength: $steering_strength"
        echo "Waiting for previous process to fully exit..."
        sleep 10  # Wait for 10 seconds to ensure that the previous process has completely exited.
        
        # Run CheckThinkLogit.py.
        python CheckThinkLogit.py \
            --model_name "$model_name" \
            --model_path "$model_path" \
            --dataset "$dataname" \
            --strength "$steering_strength" \
            --question_path "$question_path" \
            --images_root "$images_root" \
            --steering_vector_path "$steering_vector_path"
            
        echo "Completed run for model $model_name with strength $steering_strength"
    done
done

echo "All runs completed!"
