# VLM Adaptive Steering

This repository contains adaptive steering experiments for multimodal reasoning models, including generation, evaluation, embedding extraction, and analysis scripts.

## Directory Layout
- `scripts/`: Common run scripts
- `Data/Questions/`: Question datasets (JSONL)
- `Data/Images/MathV/`: Corresponding image data
- `Data/Representation/`: Embeddings and intermediate representations
- `Data/Eval/`: Evaluation outputs and statistics
- `Assets/`: Visualization/resources

## Quick Start

### 1) Generate main results
Run the following script to generate the main experimental results:

```bash
bash scripts/generate_and_evaluate.sh
```

### 2) Generate embeddings
Run the following script to generate embeddings:

```bash
bash scripts/embed.sh
```
You can pass a dataset name (e.g., `MathVMini` or `MATH500`).

### 3) Extract steering vectors
Run the following script to extract the steering vector:

```bash
python diff_embeds.py
```
