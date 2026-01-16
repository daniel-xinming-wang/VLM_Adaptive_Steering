# VLM Adaptive Steering

This repository contains adaptive steering experiments for multimodal reasoning models, including generation, evaluation, embedding extraction, and analysis scripts.

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

### 3) Extract steering vectors
Run the following script to extract the steering vector:

```bash
python diff_embeds.py
```
