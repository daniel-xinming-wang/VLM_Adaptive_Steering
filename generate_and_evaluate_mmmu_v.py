from typing import Any, Dict, List, Optional, Tuple
from string import ascii_uppercase
from PIL import Image
from transformers import AutoProcessor, AutoConfig
from jinja2 import Template
import ast
import json
import fire
import os
import re
import inspect
from tqdm import tqdm

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_boxed_content(text: str) -> str:
    if not isinstance(text, str) or "\\boxed{" not in text:
        return "None"
    try:
        matches = _BOXED_RE.findall(text)
        if matches:
            return matches[-1].strip()
    except Exception:
        pass
    try:
        start = text.rfind("\\boxed{")
        if start == -1:
            return "None"
        i = start + len("\\boxed{")
        depth = 1
        j = i
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i:j].strip()
            j += 1
    except Exception:
        pass
    return "None"


def normalize_answer(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def split_options(value: Any) -> Tuple[List[str], str]:
    options_list: Optional[List[str]] = None
    options_text = ""
    if isinstance(value, list):
        options_list = value
    elif isinstance(value, str):
        text = value.strip()
        if text:
            parsed = None
            if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = None
            if isinstance(parsed, list):
                options_list = parsed
            else:
                options_text = text
    return options_list or [], options_text


def extract_choice_letter(text: str, options_list: List[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    letters = ascii_uppercase[: max(1, len(options_list))]
    match = re.findall(
        r"(?:correct answer is|answer[:：]?)\s*(?:\*\*)?[\(\[]?([A-Z])[\)\]\.\s]?",
        text,
        re.IGNORECASE,
    )
    for candidate in reversed(match):
        cand = candidate.upper()
        if cand in letters:
            return cand
    match2 = re.findall(r"\b([A-Z])\b", text, re.IGNORECASE)
    for candidate in reversed(match2):
        cand = candidate.upper()
        if cand in letters:
            return cand
    return None


def match_option_text(text: str, options_list: List[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    hay = text.lower()
    best = None
    best_len = 0
    for opt in options_list:
        if not isinstance(opt, str):
            continue
        needle = opt.strip().lower()
        if not needle:
            continue
        if needle in hay and len(needle) > best_len:
            best = opt
            best_len = len(needle)
    return best


def build_gt_candidates(raw_answer: Any, options_list: List[str]) -> List[str]:
    candidates: List[str] = []
    if isinstance(raw_answer, list):
        candidates.extend([str(a) for a in raw_answer if a is not None])
    elif raw_answer is not None:
        candidates.append(str(raw_answer))
    if isinstance(raw_answer, str):
        letter = raw_answer.strip().upper()
        letters = ascii_uppercase[: len(options_list)]
        if len(letter) == 1 and letter in letters:
            idx = letters.index(letter)
            if 0 <= idx < len(options_list):
                candidates.append(str(options_list[idx]))
    return candidates


def build_pred_candidates(pred: str, options_list: List[str]) -> List[str]:
    candidates = []
    if pred is None:
        return candidates
    candidates.append(str(pred))
    letter = str(pred).strip().upper()
    letters = ascii_uppercase[: len(options_list)]
    if len(letter) == 1 and letter in letters:
        idx = letters.index(letter)
        if 0 <= idx < len(options_list):
            candidates.append(str(options_list[idx]))
    return candidates


def extract_final_answer(text: str, options_list: List[str]) -> str:
    ans = extract_boxed_content(text)
    if ans != "None":
        return ans
    letter = extract_choice_letter(text, options_list)
    if letter:
        return letter
    option_match = match_option_text(text, options_list)
    if option_match:
        return option_match
    return "None"


# ---------------------------- Main pipeline ----------------------------

def generate_and_evaluate(
    model_path="Qwen/Qwen3-VL-8B-Thinking",
    dataname="MMMU_Validation",
    data_path="./Data/Questions/MMMU_Validation.jsonl",
    base_save_path="",
    generation_save_path="",
    overall_trend_save_path="",
    batch_size=64,
    vote_num=1,
    tensor_parallel_size=1,
    max_tokens=40960,
    steering_vector_path="Empty",
    steering_strength=0.0,
    adaptive_steering=False,
    adaptive_alpha=0.1,
    adaptive_reduce="last",
    adaptive_eps=1e-6,
    calibration_vector_path="",
    images_root="./Data/Images/MMMU_Val",
    outputs_dir="",
    run_name="Qwen3VL_MMMU_Validation",
    enforce_eager=False,
):
    from vllm import LLM, SamplingParams, ModelRegistry

    if base_save_path and not os.path.exists(base_save_path):
        os.makedirs(base_save_path, exist_ok=True)

    if steering_vector_path in (None, "", "Empty"):
        print("[info] No steering vector path provided; steering disabled.")
    path_provided = steering_vector_path not in (None, "", "Empty")
    path_exists = os.path.exists(steering_vector_path) if path_provided else False
    strength_nonzero = float(steering_strength) != 0.0 if steering_strength is not None else False
    if path_provided and not path_exists:
        print(f"[warn] Steering vector path not found: {steering_vector_path}. Disabling steering.")
    if path_provided and not strength_nonzero and not adaptive_steering:
        print("[info] Steering strength is 0.0; steering disabled.")
    if path_provided and adaptive_steering and float(adaptive_alpha) == 0.0:
        print("[warn] adaptive_alpha is 0.0; adaptive steering will be a no-op.")

    steering_enabled = path_provided and path_exists and (strength_nonzero or adaptive_steering)
    if steering_enabled:
        ModelRegistry.register_model(
            "Qwen3VLForConditionalGeneration",
            "steer_qwen3_vl_vllm:SteerQwen3VLForConditionalGeneration",
        )
        print("Registered SteerQwen3VLForConditionalGeneration for vLLM.")
    else:
        os.environ.pop("steering_vector_path", None)
        os.environ.pop("steering_strength_list", None)

    available_params = set(inspect.signature(SamplingParams).parameters)
    sampling_args = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": max_tokens,
        "n": vote_num,
        "stop": ["<|im_end|>"],
    }
    sampling_args = {k: v for k, v in sampling_args.items() if k in available_params}
    sampling_params = SamplingParams(**sampling_args)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if not getattr(processor, "apply_chat_template", None):
        raise RuntimeError("Loaded processor does not support apply_chat_template.")
    if not getattr(processor, "chat_template", None) and getattr(tokenizer, "chat_template", None):
        processor.chat_template = tokenizer.chat_template

    template_jinja = """\
    This is the problem:
    {{prompt}}
    {{options_block}}
    """
    prompt_template = Template(template_jinja)

    def create_llm(steering_strength, steering_vector_path, model_path, tensor_parallel_size=1):
        os.environ.pop("steering_strength", None)
        os.environ.pop("steering_strength_list", None)
        os.environ.pop("steering_vector_path", None)
        os.environ.pop("steering_adaptive", None)
        os.environ.pop("steering_adaptive_config", None)
        os.environ.pop("steering_adaptive_alpha", None)
        os.environ.pop("steering_calibration_path", None)

        if steering_enabled:
            config = AutoConfig.from_pretrained(model_path)
            num_layers = getattr(
                getattr(config, "text_config", config),
                "num_hidden_layers",
                getattr(config, "num_hidden_layers", 1),
            )
            os.environ["steering_vector_path"] = steering_vector_path
            os.environ["steering_strength"] = str(steering_strength)
            os.environ["steering_strength_list"] = ",".join([str(steering_strength)] * num_layers)
            os.environ["steering_adaptive"] = "1" if adaptive_steering else "0"
            os.environ["steering_adaptive_alpha"] = str(adaptive_alpha)
            os.environ["steering_adaptive_config"] = adaptive_reduce
            if calibration_vector_path:
                os.environ["steering_calibration_path"] = calibration_vector_path

        import gc
        gc.collect()

        return LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=max_tokens,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": 8},
            enforce_eager=enforce_eager,
        )

    llm = create_llm(steering_strength, steering_vector_path, model_path, tensor_parallel_size)

    if data_path.endswith(".jsonl"):
        question_dataset = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                question_dataset.append(json.loads(line))
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            question_dataset = json.load(f)

    original_data = question_dataset.copy()

    expect_images = bool(images_root) or any(
        isinstance(dp, dict) and any(str(k).startswith("image_") for k in dp.keys())
        for dp in question_dataset
    )

    print(f"Preprocessing {len(question_dataset)} datapoints")
    processed_prompts = []
    images_batch = []
    options_cache: List[List[str]] = []

    def _resolve_image_paths(dp: Dict[str, Any]) -> List[str]:
        imgs: List[str] = []
        if "images" in dp:
            if isinstance(dp["images"], list):
                imgs = dp["images"]
            elif isinstance(dp["images"], str):
                imgs = [dp["images"]]
        elif "image" in dp:
            if isinstance(dp["image"], list):
                imgs = dp["image"]
            elif isinstance(dp["image"], str):
                imgs = [dp["image"]]
        else:
            for i in range(1, 8):
                key = f"image_{i}"
                if key not in dp:
                    continue
                val = dp[key]
                if isinstance(val, list):
                    imgs.extend(val)
                elif isinstance(val, str):
                    imgs.append(val)

        if images_root:
            base_dir = images_root
        elif data_path:
            base_dir = os.path.dirname(data_path)
        else:
            base_dir = os.getcwd()

        resolved: List[str] = []
        for p in imgs:
            if not p:
                continue
            if os.path.isabs(p) and os.path.exists(p):
                resolved.append(p)
                continue
            base_name = os.path.basename(p)
            cand1 = os.path.join(base_dir, base_name)
            if os.path.exists(cand1):
                resolved.append(cand1)
                continue
            cand2 = os.path.join(base_dir, p)
            if os.path.exists(cand2):
                resolved.append(cand2)
                continue
        return resolved

    zero_image_qids = []

    for datapoint in tqdm(question_dataset):
        problem = datapoint.get("problem") or datapoint.get("question") or ""

        options_list, options_text = split_options(datapoint.get("options"))
        options_cache.append(options_list)

        options_block = ""
        if options_list:
            labels = list(ascii_uppercase[: len(options_list)])
            lines = [f"{label}. {opt}" for label, opt in zip(labels, options_list)]
            if lines:
                options_block = "\nOptions:\n" + "\n".join(lines)
        elif options_text:
            options_block = "\nOptions:\n" + options_text

        prompt_temp = prompt_template.render(prompt=problem, options_block=options_block)

        if options_list:
            labels = ", ".join(ascii_uppercase[: max(1, len(options_list))])
            system_content = (
                "Please reason step by step. Choose the correct option among "
                f"{labels}, and put the chosen letter as your final answer within \\boxed{{}}."
            )
        else:
            system_content = "Please reason step by step, and put your final answer within \\boxed{}."

        pil_images: List[Image.Image] = []
        for img_path in _resolve_image_paths(datapoint):
            try:
                im = Image.open(img_path)
                if im.mode == "P" and "transparency" in im.info:
                    im = im.convert("RGBA")
                im = im.convert("RGB")
                pil_images.append(im)
            except Exception as err:
                print(f"[warn] failed to load image {img_path}: {err}")
                continue

        if not pil_images and expect_images:
            qid = datapoint.get("question_id") or datapoint.get("id") or datapoint.get("uid")
            print(f"[warn] no image loaded for question {qid}")
            if qid is not None:
                zero_image_qids.append(qid)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": ([{"type": "image"}] * len(pil_images)) + [{"type": "text", "text": prompt_temp}]},
        ]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        placeholder_count = prompt_text.count("<|image_pad|>")
        if placeholder_count != len(pil_images):
            qid = datapoint.get("question_id") or datapoint.get("id") or datapoint.get("uid")
            print(
                f"[warn] placeholder/image mismatch for question {qid}: "
                f"placeholders={placeholder_count}, images={len(pil_images)}"
            )
        processed_prompts.append(prompt_text)
        images_batch.append(pil_images if pil_images else None)

    print("len(processed_prompts):", len(processed_prompts))
    if zero_image_qids:
        print(f"[summary] {len(zero_image_qids)} samples had no images attached. Example IDs: {zero_image_qids[:5]}")

    structured_inputs = []
    for p_text, imgs in zip(processed_prompts, images_batch):
        if imgs:
            structured_inputs.append({"prompt": p_text, "multi_modal_data": {"image": imgs}})
        else:
            structured_inputs.append({"prompt": p_text})

    all_generated_texts: List[List[str]] = []
    total = len(structured_inputs)
    for start in range(0, total, batch_size):
        batch = structured_inputs[start:start + batch_size]
        outputs = llm.generate(prompts=batch, sampling_params=sampling_params)
        for req_out in outputs:
            texts = [o.text for o in req_out.outputs]
            all_generated_texts.append(texts)

        done = min(start + batch_size, total)
        if done % max(10, batch_size) == 0 or done == total:
            print(f"Processed {done}/{total} samples (batched)")

    results_for_saving = []
    for i in range(len(all_generated_texts)):
        datapoint = original_data[i]
        options_list = options_cache[i]
        gt_candidates = build_gt_candidates(datapoint.get("answer"), options_list)
        gt_norms = {normalize_answer(c) for c in gt_candidates if c is not None}

        one_record = datapoint.copy()
        qid = datapoint.get("question_id") or datapoint.get("id") or datapoint.get("uid") or f"idx_{i}"
        one_record["question_id"] = qid
        one_record["llm_reasoning"], one_record["llm_answer"], one_record["llm_final_answer"], one_record["is_correct"] = [], [], [], []
        one_record["llm_reasoning_token_num"], one_record["llm_answer_token_num"] = [], []

        one_generation = all_generated_texts[i]
        for rollout in one_generation:
            think_idx = rollout.find("</think>")
            if think_idx != -1:
                llm_reasoning = rollout[:think_idx]
                post_think = rollout[think_idx + len("</think>"):]
                llm_answer = post_think.strip()
            else:
                llm_reasoning = rollout
                post_think = rollout
                llm_answer = rollout.strip()

            llm_final_answer = extract_final_answer(post_think, options_list)
            if llm_final_answer == "None":
                llm_final_answer = extract_final_answer(rollout, options_list)

            llm_reasoning_token_num = len(tokenizer.encode(llm_reasoning))
            llm_answer_token_num = len(tokenizer.encode(llm_answer))

            pred_candidates = build_pred_candidates(llm_final_answer, options_list)
            pred_norms = {normalize_answer(c) for c in pred_candidates if c is not None}
            is_correct = bool(gt_norms.intersection(pred_norms))

            one_record["llm_reasoning"].append(llm_reasoning)
            one_record["llm_answer"].append(llm_answer)
            one_record["llm_final_answer"].append(llm_final_answer)
            one_record["is_correct"].append(is_correct)
            one_record["llm_reasoning_token_num"].append(llm_reasoning_token_num)
            one_record["llm_answer_token_num"].append(llm_answer_token_num)

        def _avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        one_record["avg_llm_reasoning_token_num"] = _avg(one_record["llm_reasoning_token_num"])
        one_record["avg_llm_answer_token_num"] = _avg(one_record["llm_answer_token_num"])
        one_record["accuracy"] = _avg(one_record["is_correct"])
        results_for_saving.append(one_record)

    if outputs_dir:
        eval_out = os.path.join(outputs_dir, f"{run_name}.jsonl")
        os.makedirs(os.path.dirname(eval_out), exist_ok=True)
        with open(eval_out, "w", encoding="utf-8") as wf:
            for rec in results_for_saving:
                fa_list = rec.get("llm_final_answer", []) or ["None"]
                majority = max(set(fa_list), key=fa_list.count) if fa_list else "None"
                wf.write(json.dumps({"question_id": rec.get("question_id"), "prediction": majority}, ensure_ascii=False) + "\n")
        print(f"Wrote evaluator file to: {eval_out}")
    else:
        print("[info] outputs_dir not set; skipping evaluator JSONL.")

    if generation_save_path:
        with open(generation_save_path, "w", encoding="utf-8") as f:
            json.dump(results_for_saving, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {generation_save_path}")

    total_accuracy = sum([datapoint["accuracy"] for datapoint in results_for_saving]) / len(results_for_saving) if results_for_saving else 0.0
    print(f"Total accuracy: {total_accuracy}")

    def _avg_all(key):
        return sum([dp.get(key, 0.0) for dp in results_for_saving]) / len(results_for_saving) if results_for_saving else 0.0

    total_llm_reasoning_token_num = _avg_all("avg_llm_reasoning_token_num")
    total_llm_answer_token_num = _avg_all("avg_llm_answer_token_num")
    print(f"Average reasoning token num: {total_llm_reasoning_token_num}")
    print(f"Average answer token num: {total_llm_answer_token_num}")

    if overall_trend_save_path:
        if os.path.exists(overall_trend_save_path) and os.path.getsize(overall_trend_save_path) > 0:
            with open(overall_trend_save_path, "r", encoding="utf-8") as f:
                overall_trend_data = json.load(f)
        else:
            overall_trend_data = []

        new_entry = {
            "strength": steering_strength,
            "total_accuracy": total_accuracy,
            "average_reasoning_token_num": total_llm_reasoning_token_num,
            "average_answer_token_num": total_llm_answer_token_num,
            "alpha": adaptive_alpha,
        }

        existing_entry = next(
            (entry for entry in overall_trend_data if entry["strength"] == steering_strength and entry["alpha"] == adaptive_alpha),
            None,
        )
        if existing_entry:
            existing_entry.update(new_entry)
        else:
            overall_trend_data.append(new_entry)

        with open(overall_trend_save_path, "w", encoding="utf-8") as f:
            json.dump(overall_trend_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    fire.Fire(generate_and_evaluate)
