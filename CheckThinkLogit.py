from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from jinja2 import Template
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from SteerModels import SteerQwen3VLForConditionalGeneration

torch.manual_seed(42)
np.random.seed(42)


END_THINK_TOKEN_ID = 151668  # Qwen3-VL: </think>


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-VL-8B-Thinking",
        help="Model name for saving under Assets/<dataset>/<model_name>/...",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Thinking",
        help="HF repo id or local path.",
    )
    parser.add_argument("--dataset", type=str, default="MathV", help="dataset name.")
    parser.add_argument("--strength", type=float, default=0.0, help="steering strength.")
    parser.add_argument(
        "--question_path",
        type=str,
        default="Data/Questions/MathVMini.jsonl",
        help="question path (.json or .jsonl).",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="",
        help="Base directory for resolving image paths (e.g. ./Data/Images/MathV).",
    )
    parser.add_argument(
        "--steering_vector_path",
        type=str,
        default="",
        help="Path to steering vectors (.npy). If empty, uses dataset-specific defaults.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=384 * 384,
        help="Max image pixels for processor resize (e.g. 512*512).",
    )
    args, _ = parser.parse_known_args()
    return args, _


def load_questions(question_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(question_path):
        # Convenience fallback (some scripts still point to .json while repo uses .jsonl)
        if question_path.endswith(".json") and os.path.exists(question_path + "l"):
            print(f"[warn] question_path not found: {question_path}; falling back to {question_path}l")
            question_path = question_path + "l"
        else:
            raise FileNotFoundError(question_path)
    if question_path.endswith(".jsonl"):
        out: List[Dict[str, Any]] = []
        with open(question_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    with open(question_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_paths(dp: Dict[str, Any], *, question_path: str, images_root: str) -> List[str]:
    imgs: List[str] = []
    if "images" in dp and isinstance(dp["images"], list):
        imgs = dp["images"]
    elif "image" in dp:
        if isinstance(dp["image"], list):
            imgs = dp["image"]
        elif isinstance(dp["image"], str):
            imgs = [dp["image"]]

    if images_root:
        base_dir = images_root
    elif question_path:
        base_dir = os.path.dirname(question_path)
    else:
        base_dir = os.getcwd()

    resolved: List[str] = []
    for p in imgs:
        if not p:
            continue
        if os.path.isabs(p) and os.path.exists(p):
            resolved.append(p)
            continue
        # Match generate_and_evaluate_v.py:
        # 1) try basename under base_dir (handles p like "images/1.jpg")
        cand1 = os.path.join(base_dir, os.path.basename(p))
        if os.path.exists(cand1):
            resolved.append(cand1)
            continue
        # 2) try p relative to base_dir (handles p like "subdir/images/1.jpg")
        cand2 = os.path.join(base_dir, p)
        if os.path.exists(cand2):
            resolved.append(cand2)
            continue
    return resolved


def load_pil_images(paths: List[str]) -> List[Image.Image]:
    out: List[Image.Image] = []
    for p in paths:
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception as err:
            print(f"[warn] failed to load image {p}: {err}")
    return out


args, _ = parse_args()
dataset, model_name, strength, question_path = args.dataset, args.model_name, args.strength, args.question_path
images_root = args.images_root
max_pixels = args.max_pixels

if not images_root and dataset.lower().startswith("mathv"):
    # Default to the same layout as scripts/generate_and_evaluate.sh
    cand = os.path.join("Data", "Images", "MathV")
    if os.path.isdir(cand):
        images_root = cand

print("dataset:", dataset, "model_name:", model_name, "strength:", strength, "question_path:", question_path)
print("images_root:", images_root if images_root else "(empty)")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE, "torch.cuda.device_count():", torch.cuda.device_count())

selected_problem_file_name = os.path.basename(question_path).split(".")[0]
print("selected_problem_file_name:", selected_problem_file_name)

save_path = f"Assets/{dataset}/{model_name}/steering_by_strength/{selected_problem_file_name}"
os.makedirs(save_path, exist_ok=True)

# Load Questions
data_problems = load_questions(question_path)

# Processor + tokenizer (Qwen3-VL)
processor = AutoProcessor.from_pretrained(args.model_path)
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
tokenizer.padding_side = "left"

# Quick sanity: END_THINK_TOKEN_ID should decode to </think> for Qwen3-VL
try:
    decoded = tokenizer.decode([END_THINK_TOKEN_ID])
    if decoded.strip() != "</think>":
        print(f"[warn] token id {END_THINK_TOKEN_ID} decodes to {repr(decoded)} (expected '</think>')")
except Exception as err:
    print(f"[warn] failed to decode END_THINK_TOKEN_ID={END_THINK_TOKEN_ID}: {err}")

eos_token = tokenizer.eos_token_id
print("eos_token_id:", eos_token)

# Load steering vectors (dataset-aware)
steering_vector_path = args.steering_vector_path
fallback_paths: List[str] = []
if not steering_vector_path:
    if dataset.upper() == "MATH500":
        candidate = os.path.join("Assets", "MATH500", model_name, "mean_steering_vectors_all.npy")
    else:
        candidate = os.path.join("Assets", "MathV", model_name, "mean_steering_vectors_all.npy")
    fallback_paths.append(candidate)
    if os.path.exists(candidate):
        steering_vector_path = candidate

if not steering_vector_path:
    fallback = os.path.join("Data", "Representation", "steering_vectors.npy")
    fallback_paths.append(fallback)
    if os.path.exists(fallback):
        steering_vector_path = fallback

if not steering_vector_path or not os.path.exists(steering_vector_path):
    searched = ", ".join(fallback_paths) if fallback_paths else "(none)"
    raise FileNotFoundError(
        f"Steering vector not found. Provided: {args.steering_vector_path!r}; searched: {searched}"
    )
print("Loading steering vectors... from", steering_vector_path)
cfg = AutoConfig.from_pretrained(args.model_path)
text_cfg = getattr(cfg, "text_config", cfg)
num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
if num_layers <= 0:
    raise RuntimeError("num_hidden_layers not found in config")

steering_vectors_np = np.load(steering_vector_path)
steering_vectors = torch.from_numpy(steering_vectors_np).to(DEVICE)
apply_steering_indices = [True] * num_layers
apply_steering_indices[0] = False
layer_wise_strength = [float(strength)] * num_layers

# Construct model
print("Is constructing model...")
device_map = {"": DEVICE} if DEVICE != "cpu" else {"": "cpu"}
model = SteerQwen3VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    steering_vectors=steering_vectors,
    apply_steering_indices=apply_steering_indices,
    strength=layer_wise_strength,
    temperature=1.0,
)
model.eval()
try:
    text_cfg = getattr(model.config, "text_config", model.config)
    setattr(text_cfg, "_attn_implementation", "eager")
except Exception:
    pass

image_token = getattr(processor, "image_token", "<|image_pad|>")
image_token_id = tokenizer.convert_tokens_to_ids(image_token)
if image_token_id is None or image_token_id < 0:
    print(f"[warn] image_token_id not found for {image_token!r}; visual attention ratio disabled.")
    image_token_id = None

attn_store: List[float] = []
current_visual_mask: Optional[torch.Tensor] = None

def _attn_hook(_module, _inputs, outputs):
    if isinstance(outputs, tuple) and len(outputs) > 1:
        attn = outputs[1]
        if attn is None or current_visual_mask is None:
            return
        if not current_visual_mask.any():
            return
        attn_last = attn[:, :, -1, :]
        visual_mass = (attn_last * current_visual_mask[:, None, :]).sum(dim=-1)
        total_mass = attn_last.sum(dim=-1).clamp_min(1e-12)
        attn_store.append((visual_mass / total_mass).mean().item())

for layer in model.model.language_model.layers:
    layer.self_attn.register_forward_hook(_attn_hook)

template_jinja = """\
This is the problem:
{{prompt}}
{{options_block}}
"""
prompt_template = Template(template_jinja)

# Generate / measure logits (original logic: batch_size=1 loop)
time_start = time.time()
batch_size = 1
temperature = 1.0

# Original logic sampled random tokens from [0, eos_token_id)
random_tokens = np.random.randint(0, eos_token, size=2000)

think_token_logits_list = []
think_token_probs_list = []
eos_token_logits_list = []
eos_token_probs_list = []
random_token_logits_list = []
random_token_probs_list = []
last_token_logits_list = []
last_token_probs_list = []
last_token_entropy_list = []
visual_attn_ratio_list = []

for i in tqdm(range(0, len(data_problems), batch_size), desc="Processing", total=(len(data_problems) + batch_size - 1) // batch_size):
    batch_problems = data_problems[i : i + batch_size]
    batch_inputs: List[str] = []
    batch_images: List[Optional[List[Image.Image]]] = []

    for problem_dict in batch_problems:
        problem = problem_dict.get("problem") or problem_dict.get("question") or ""
        # Match generate_and_evaluate_v.py: remove literal placeholders like "<image1>"
        problem = re.sub(r"<image\\d+>", "", problem, flags=re.IGNORECASE)

        options_block = ""
        options = problem_dict.get("options")
        if isinstance(options, list) and options:
            if options == ["A", "B", "C", "D", "E"]:
                options_block = "\nOptions:\n" + "\n".join(options)
            else:
                labels = ["A", "B", "C", "D", "E"]
                lines = []
                for idx, opt in enumerate(options[: len(labels)]):
                    lines.append(f"{labels[idx]}. {opt}")
                if lines:
                    options_block = "\nOptions:\n" + "\n".join(lines)

        prompt_temp = prompt_template.render(prompt=problem, options_block=options_block)
        if options_block:
            system_content = (
                "Please reason step by step. Choose the correct option among "
                "A, B, C, D, and E, and put the chosen letter as your final "
                "answer within \\boxed{}."
            )
        else:
            system_content = "Please reason step by step, and put your final answer within \\boxed{}."

        img_paths = resolve_image_paths(problem_dict, question_path=question_path, images_root=images_root)
        pil_images = load_pil_images(img_paths)
        batch_images.append(pil_images if pil_images else None)

        # Match generate_and_evaluate_v.py: system + user with explicit image entries.
        message = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": ([{"type": "image"}] * len(pil_images))
                + [{"type": "text", "text": prompt_temp}],
            },
        ]

        template_input = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        batch_inputs.append(template_input)

    t1 = time.time()

    # batch_size=1: keep per-sample processing simple and faithful.
    prompt_text = batch_inputs[0]
    imgs = batch_images[0]
    if imgs:
        inputs = processor(
            text=[prompt_text],
            images=[imgs],
            return_tensors="pt",
            padding=True,
            images_kwargs={"max_pixels": max_pixels},
        )
    else:
        inputs = processor(text=[prompt_text], return_tensors="pt", padding=True)
    inputs = inputs.to(DEVICE)

    if image_token_id is not None:
        count = int((inputs["input_ids"] == image_token_id).sum().item())
        #print("image_pad count:", count)

    attn_store.clear()
    if image_token_id is not None:
        current_visual_mask = inputs["input_ids"].eq(image_token_id)
    else:
        current_visual_mask = None
    with torch.no_grad():
        output = model(**inputs)

    """
    print("attn_store len:", len(attn_store))
    if attn_store:
        print("attn shape:", attn_store[0].shape)
    """

    last_token_logits_t = output.logits[:, -1, :].to(torch.float32)
    last_token_probs_t = torch.softmax(last_token_logits_t / temperature, dim=-1)
    last_token_entropy_t = -(last_token_probs_t * torch.log(last_token_probs_t + 1e-12)).sum(dim=-1)

    last_token_logits = last_token_logits_t.detach().cpu().numpy()
    last_token_probs = last_token_probs_t.detach().cpu().numpy()
    last_token_entropy = last_token_entropy_t.detach().cpu().numpy()

    topk = 5
    topk_probs, topk_ids = torch.topk(last_token_probs_t[0], k=topk)
    topk_logits = last_token_logits_t[0, topk_ids]
    topk_probs = topk_probs.detach().cpu().numpy()
    topk_logits = topk_logits.detach().cpu().numpy()
    topk_ids = topk_ids.detach().cpu().numpy()
    topk_tokens = [tokenizer.decode([int(tid)]) for tid in topk_ids]
    print("Top-5 tokens:", [(int(tid), repr(tok), float(prob), float(logit))
                            for tid, tok, prob, logit in zip(topk_ids, topk_tokens, topk_probs, topk_logits)])

    if attn_store:
        visual_attn_ratio_list.append(float(np.mean(attn_store)))

    think_token_logits = last_token_logits[:, END_THINK_TOKEN_ID]
    think_token_probs = last_token_probs[:, END_THINK_TOKEN_ID]

    eos_token_logits = last_token_logits[:, eos_token]
    eos_token_probs = last_token_probs[:, eos_token]

    random_token_logits = last_token_logits[:, random_tokens]
    random_token_probs = last_token_probs[:, random_tokens]
    random_token_logits = np.mean(random_token_logits, axis=-1)
    random_token_probs = np.mean(random_token_probs, axis=-1)

    if i == 0:
        think_token_logits_list = think_token_logits
        think_token_probs_list = think_token_probs
        eos_token_logits_list = eos_token_logits
        eos_token_probs_list = eos_token_probs
        random_token_logits_list = random_token_logits
        random_token_probs_list = random_token_probs
        last_token_logits_list = last_token_logits
        last_token_probs_list = last_token_probs
        last_token_entropy_list = last_token_entropy
    else:
        think_token_logits_list = np.concatenate((think_token_logits_list, think_token_logits), axis=0)
        think_token_probs_list = np.concatenate((think_token_probs_list, think_token_probs), axis=0)
        eos_token_logits_list = np.concatenate((eos_token_logits_list, eos_token_logits), axis=0)
        eos_token_probs_list = np.concatenate((eos_token_probs_list, eos_token_probs), axis=0)
        random_token_logits_list = np.concatenate((random_token_logits_list, random_token_logits), axis=0)
        random_token_probs_list = np.concatenate((random_token_probs_list, random_token_probs), axis=0)
        last_token_logits_list = np.concatenate((last_token_logits_list, last_token_logits), axis=0)
        last_token_probs_list = np.concatenate((last_token_probs_list, last_token_probs), axis=0)
        last_token_entropy_list = np.concatenate((last_token_entropy_list, last_token_entropy), axis=0)

    t2 = time.time()
    #print(f"Batch {i//batch_size + 1} time cost: {t2 - t1}")

mean_last_token_logits = np.mean(last_token_logits_list, axis=0)
mean_last_token_probs = np.mean(last_token_probs_list, axis=0)
mean_last_token_entropy = np.mean(last_token_entropy_list, axis=0)
mean_visual_attn_ratio = float(np.mean(visual_attn_ratio_list)) if visual_attn_ratio_list else None

time_end = time.time()
print(f"Whole time cost: {time_end - time_start} seconds")

print("Average </think> token logits:", np.mean(think_token_logits_list))
print("Average </think> token probs:", np.mean(think_token_probs_list))
print("Average eos token logits:", np.mean(eos_token_logits_list))
print("Average eos token probs:", np.mean(eos_token_probs_list))
print("Average random token logits:", np.mean(random_token_logits_list))
print("Average random token probs:", np.mean(random_token_probs_list))
print("Average last-token entropy:", mean_last_token_entropy)
print("Average visual attention ratio:", mean_visual_attn_ratio)

token_logits_probs_dict = {
    # Keep original keys for downstream analysis scripts.
    "think_token_logits": think_token_logits_list,
    "think_token_probs": think_token_probs_list,
    "eos_token_logits": eos_token_logits_list,
    "eos_token_probs": eos_token_probs_list,
    "random_token_logits": random_token_logits_list,
    "random_token_probs": random_token_probs_list,
    "mean_last_token_logits": mean_last_token_logits,
    "mean_last_token_probs": mean_last_token_probs,
    "last_token_entropy": last_token_entropy_list,
    "mean_last_token_entropy": mean_last_token_entropy,
    "visual_attn_ratio": visual_attn_ratio_list,
    "mean_visual_attn_ratio": mean_visual_attn_ratio,
    # Extra metadata (harmless for np.load(...).item()).
    "meta": {
        "target_token": "</think>",
        "target_token_id": END_THINK_TOKEN_ID,
        "temperature": temperature,
        "batch_size": batch_size,
        "random_token_count": int(random_tokens.shape[0]),
        "random_token_upper_bound": int(eos_token),
        "model_path": args.model_path,
        "question_path": question_path,
    },
}

out_file = os.path.join(save_path, f"token_logits_probs_{strength}.npy")
np.save(out_file, token_logits_probs_dict)
print("Saved:", out_file)
