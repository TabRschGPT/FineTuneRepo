#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

HF_CACHE_ROOT = "./hf_cache"
os.environ["HF_HOME"]            = HF_CACHE_ROOT
os.environ["HF_DATASETS_CACHE"]  = f"{HF_CACHE_ROOT}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_ROOT}/transformers"
os.environ["HF_HUB_CACHE"]       = f"{HF_CACHE_ROOT}/hub"

import gc
import json
import time
import hashlib
import datetime
from threading import Lock
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import load_dataset
from unsloth import FastVisionModel, is_bf16_supported
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback, TextStreamer
from openai import OpenAI

# ================================================================
# CONFIG
# ================================================================
PREPARED_PATH         = "vietmed/sft_16k_mix"
DATASET_SPLIT         = "train"
MODEL_NAME            = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
SFT_ADAPTER           = "vietmed/qwen-7b-16k-sft"
OUT_DIR               = "./outputs_grpo"
MAX_SEQ_LEN           = 2048
LOAD_4BIT             = True
LORA_R                = 16
LORA_ALPHA            = 16
NUM_EPOCHS            = 1
BATCH_SIZE            = 1
GRAD_ACCUM            = 1
LR                    = 5e-6
ADAM_BETA1            = 0.9
ADAM_BETA2            = 0.99
WEIGHT_DECAY          = 0.1
WARMUP_RATIO          = 0.1
MAX_GRAD_NORM         = 0.1
SEED                  = 3407
SAVE_STEPS            = 60
LOG_STEPS             = 1
RESUME_FROM           = None
NUM_GENERATIONS       = 2
MAX_PROMPT_LENGTH     = 1024
MAX_COMPLETION_LENGTH = 1024
W_ESSENTIAL           = 1.0
W_IMPORTANT           = 0.7
W_OPTIONAL            = 0.3
W_STYLE               = 0.2
PITFALL_PENALTY       = 0.8
REWARD_WEIGHTS        = [W_ESSENTIAL, W_IMPORTANT, W_OPTIONAL, W_STYLE, 1.0]
MAX_IMAGE_SIZE        = 512
MAX_RATIO             = 150
MIN_SHORT_SIDE        = 8
JUDGE_API_KEY         = "sk-jhK11Lj4xJnjPEeOPxLGd8nKiv4Zf2ScB8IhHog3VsfS0lhD"
JUDGE_BASE_URL        = "https://api.braintrust.dev/v1/proxy"
JUDGE_MODEL           = "gpt-4.1-mini"
JUDGE_TEMP            = 0.0
JUDGE_MAX_TOK         = 640
JUDGE_RETRIES         = 3

_FALLBACK_SCORES = {
    "factual_correctness": 0.0,
    "no_hallucination"   : 0.0,
    "reasoning_quality"  : 0.5,
    "question_coverage"  : 0.5,
    "nomenclature_units" : 0.5,
    "conciseness"        : 0.5,
    "context_richness"   : 0.5,
    "formatting"         : 0.5,
    "language_clarity"   : 0.5,
    "appropriate_length" : 0.5,
    "fabricates_values"  : False,
    "misaligns_data"     : False,
    "reason"             : "judge unavailable",
}

# ================================================================
# IMAGE UTILS
# ================================================================
def fix_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = max(img.size[0], 1), max(img.size[1], 1)
    if max(w, h) / min(w, h) > MAX_RATIO:
        if w >= h:
            new_h  = max(MIN_SHORT_SIDE, -(-w // MAX_RATIO))
            canvas = Image.new("RGB", (w, new_h), (255, 255, 255))
            canvas.paste(img, (0, (new_h - h) // 2))
            img, w, h = canvas, w, new_h
        else:
            new_w  = max(MIN_SHORT_SIDE, -(-h // MAX_RATIO))
            canvas = Image.new("RGB", (new_w, h), (255, 255, 255))
            canvas.paste(img, ((new_w - w) // 2, 0))
            img, w, h = canvas, new_w, h
    long_side = max(w, h)
    if long_side > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / long_side
        new_w = max(MIN_SHORT_SIDE, int(w * scale))
        new_h = max(MIN_SHORT_SIDE, int(h * scale))
        img   = img.resize((new_w, new_h), Image.LANCZOS)
        w, h  = new_w, new_h
    if max(w, h) / max(min(w, h), 1) > MAX_RATIO:
        if w >= h:
            new_h  = max(MIN_SHORT_SIDE, -(-w // MAX_RATIO))
            canvas = Image.new("RGB", (w, new_h), (255, 255, 255))
            canvas.paste(img, (0, (new_h - h) // 2))
            img = canvas
        else:
            new_w  = max(MIN_SHORT_SIDE, -(-h // MAX_RATIO))
            canvas = Image.new("RGB", (new_w, h), (255, 255, 255))
            canvas.paste(img, ((new_w - w) // 2, 0))
            img = canvas
    w, h = img.size
    if min(w, h) < MIN_SHORT_SIDE:
        img = img.resize(
            (MIN_SHORT_SIDE, h) if w < h else (w, MIN_SHORT_SIDE),
            Image.LANCZOS,
        )
    return img


def validate_image(img, idx):
    w, h = img.size
    if max(w, h) / max(min(w, h), 1) >= 200:
        print(f"  [SKIP] sample {idx}: bad ratio", flush=True)
        return False
    if min(w, h) < 1:
        print(f"  [SKIP] sample {idx}: degenerate", flush=True)
        return False
    return True

# ================================================================
# LLM JUDGE
# ================================================================
_JUDGE_SYSTEM = f"""You are an expert evaluator scoring AI-generated medical/clinical answers.
RUBRIC
======
Essential (weight={W_ESSENTIAL}):
  [factual_correctness] Values exactly match the reference.
  [no_hallucination]    No invented facts. Every claim is traceable.
Important (weight={W_IMPORTANT}):
  [reasoning_quality]   Reasoning is logically sound and complete.
  [question_coverage]   All parts of the question are addressed.
  [nomenclature_units]  Correct symbols and units used.
Optional (weight={W_OPTIONAL}):
  [conciseness]         Concise, no padding.
  [context_richness]    Relevant context included where appropriate.
Style (weight={W_STYLE}):
  [formatting]          Good structure, no walls of text.
  [language_clarity]    Clear and unambiguous language.
  [appropriate_length]  Length matches question complexity.
Pitfalls (penalty={PITFALL_PENALTY} each):
  [fabricates_values]   Fabricated numbers or data not in reference.
  [misaligns_data]      Confused rows/columns or misaligned data.
RULES
=====
1. Score each criterion 0.0 to 1.0.
2. If factual_correctness < 1.0 OR no_hallucination < 1.0 -> failed essential.
3. Only apply pitfall penalties when clearly observed.
OUTPUT: valid JSON only, no markdown fences, no extra text:
{{
  "factual_correctness": <float 0-1>,
  "no_hallucination":    <float 0-1>,
  "reasoning_quality":   <float 0-1>,
  "question_coverage":   <float 0-1>,
  "nomenclature_units":  <float 0-1>,
  "conciseness":         <float 0-1>,
  "context_richness":    <float 0-1>,
  "formatting":          <float 0-1>,
  "language_clarity":    <float 0-1>,
  "appropriate_length":  <float 0-1>,
  "fabricates_values":   <bool>,
  "misaligns_data":      <bool>,
  "reason":              "<one sentence>"
}}"""

_judge_cache  = {}
_cache_lock   = Lock()
_judge_client = None
_judge_ok     = True


def _get_client():
    global _judge_client
    if _judge_client is None:
        if not JUDGE_API_KEY:
            return None
        _judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
    return _judge_client


def _cache_key(question, completion):
    return hashlib.md5(f"{question}|||{completion}".encode()).hexdigest()


def _judge(question, ground_truth, completion):
    global _judge_ok
    key = _cache_key(question, completion)
    with _cache_lock:
        if key in _judge_cache:
            return _judge_cache[key]

    client = _get_client()
    if client is None:
        with _cache_lock:
            _judge_cache[key] = _FALLBACK_SCORES
        return _FALLBACK_SCORES

    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"GROUND TRUTH:\n{ground_truth}\n\n"
        f"MODEL ANSWER:\n{completion}\n\n"
        "Score the model answer according to the rubric."
    )

    result     = None
    last_error = ""

    for attempt in range(1, JUDGE_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model           = JUDGE_MODEL,
                temperature     = JUDGE_TEMP,
                max_tokens      = JUDGE_MAX_TOK,
                response_format = {"type": "json_object"},
                messages        = [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                last_error = "empty response"
                if attempt < JUDGE_RETRIES:
                    time.sleep(1.5 * attempt)
                continue
            result    = json.loads(raw.strip())
            _judge_ok = True
            break
        except json.JSONDecodeError as e:
            last_error = f"JSON: {e}"
            if attempt < JUDGE_RETRIES:
                time.sleep(1.5 * attempt)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < JUDGE_RETRIES:
                time.sleep(2.0 * attempt)

    if result is None:
        if _judge_ok:
            print(
                f"  [judge] FAILED: {last_error} — using fallback scores",
                flush=True,
            )
            _judge_ok = False
        result = dict(_FALLBACK_SCORES)

    with _cache_lock:
        _judge_cache[key] = result
    return result


def _get_question(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return " ".join(parts).strip()
    return str(prompt)

# ================================================================
# REWARD FUNCTIONS
# ================================================================
def _nc(c):
    if isinstance(c, list):
        return c[0]["content"] if c else ""
    return c


def reward_essential(prompts, completions, answer=None, **kwargs):
    """MUST be first — clears cache each batch."""
    with _cache_lock:
        _judge_cache.clear()
    if answer is None:
        return [0.0] * len(completions)
    rewards = []
    for prompt, completion, gt in zip(prompts, completions, answer):
        completion = _nc(completion)
        if len(completion) > 0:
            cleaned = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(cleaned)) / len(completion) >= 0.5:
                rewards.append(0.0)
                continue
        data = _judge(_get_question(prompt), gt, completion)
        fc   = float(data.get("factual_correctness", 0.0))
        nh   = float(data.get("no_hallucination",    0.0))
        rewards.append(
            0.0 if fc < 1.0 or nh < 1.0 else round((fc + nh) / 2, 4)
        )
    print(
        f"  [essential] mean={sum(rewards)/max(len(rewards),1):.3f} "
        f"{[round(r,2) for r in rewards]}",
        flush=True,
    )
    return rewards


def reward_important(prompts, completions, answer=None, **kwargs):
    if answer is None:
        return [0.0] * len(completions)
    rewards = []
    for prompt, completion, gt in zip(prompts, completions, answer):
        data = _judge(_get_question(prompt), gt, _nc(completion))
        rq   = float(data.get("reasoning_quality",  0.0))
        qc   = float(data.get("question_coverage",  0.0))
        nu   = float(data.get("nomenclature_units", 0.0))
        rewards.append(round((rq + qc + nu) / 3, 4))
    print(
        f"  [important] mean={sum(rewards)/max(len(rewards),1):.3f} "
        f"{[round(r,2) for r in rewards]}",
        flush=True,
    )
    return rewards


def reward_optional(prompts, completions, answer=None, **kwargs):
    if answer is None:
        return [0.0] * len(completions)
    rewards = []
    for prompt, completion, gt in zip(prompts, completions, answer):
        data = _judge(_get_question(prompt), gt, _nc(completion))
        con  = float(data.get("conciseness",      0.0))
        cr   = float(data.get("context_richness", 0.0))
        rewards.append(round((con + cr) / 2, 4))
    print(
        f"  [optional]  mean={sum(rewards)/max(len(rewards),1):.3f} "
        f"{[round(r,2) for r in rewards]}",
        flush=True,
    )
    return rewards


def reward_style(prompts, completions, answer=None, **kwargs):
    if answer is None:
        return [0.0] * len(completions)
    rewards = []
    for prompt, completion, gt in zip(prompts, completions, answer):
        data = _judge(_get_question(prompt), gt, _nc(completion))
        fmt  = float(data.get("formatting",         0.0))
        lc   = float(data.get("language_clarity",   0.0))
        al   = float(data.get("appropriate_length", 0.0))
        rewards.append(round((fmt + lc + al) / 3, 4))
    print(
        f"  [style]     mean={sum(rewards)/max(len(rewards),1):.3f} "
        f"{[round(r,2) for r in rewards]}",
        flush=True,
    )
    return rewards


def reward_pitfall(prompts, completions, answer=None, **kwargs):
    if answer is None:
        return [0.0] * len(completions)
    rewards = []
    for prompt, completion, gt in zip(prompts, completions, answer):
        data    = _judge(_get_question(prompt), gt, _nc(completion))
        penalty = 0.0
        if bool(data.get("fabricates_values", False)): penalty -= PITFALL_PENALTY
        if bool(data.get("misaligns_data",    False)): penalty -= PITFALL_PENALTY
        rewards.append(round(penalty, 4))
    violations = sum(1 for r in rewards if r < 0)
    print(
        f"  [pitfall]   violations={violations}/{len(rewards)} "
        f"{[round(r,2) for r in rewards]}",
        flush=True,
    )
    return rewards

# ================================================================
# CALLBACK
# ================================================================
class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.datetime.now()
        print(
            f"\n  Training started at "
            f"{self.start_time.strftime('%H:%M:%S')}\n",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        secs    = (datetime.datetime.now() - self.start_time).seconds
        elapsed = f"{secs//60:02d}:{secs%60:02d}"
        parts   = [f"step {state.global_step}/{state.max_steps}"]
        for key in ["epoch", "loss", "reward", "reward_std", "kl", "learning_rate"]:
            val = logs.get(key)
            if val is not None:
                parts.append(
                    f"{key}={val:.4f}" if isinstance(val, float)
                    else f"{key}={val}"
                )
        parts.append(elapsed)
        print("  " + " | ".join(parts), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        duration = str(datetime.datetime.now() - self.start_time)
        print(
            f"\n  Done | steps={state.global_step} | time={duration}\n",
            flush=True,
        )

# ================================================================
# DATA LOADER — plain list, no PyArrow
# ================================================================
def load_data(path, split="train"):
    print(f"\n  Loading '{path}' ...", flush=True)
    raw     = load_dataset(path)
    dataset = raw.get(split) or raw.get("train") or raw[list(raw.keys())[0]]
    print(f"  {len(dataset):,} samples | columns: {dataset.column_names}", flush=True)

    samples = []
    skipped = 0

    for i, sample in enumerate(dataset):
        try:
            raw_conv      = sample["conversations"]
            conversations = (
                json.loads(raw_conv) if isinstance(raw_conv, str) else list(raw_conv)
            )
            img = sample["image"]
            if isinstance(img, str):
                img = Image.open(img)
            if not isinstance(img, Image.Image):
                raise ValueError(f"Bad image type: {type(img)}")
            img = fix_image(img)
            if not validate_image(img, i):
                skipped += 1
                continue

            user_content = None
            ground_truth = None
            for turn in conversations:
                role    = turn.get("role", turn.get("from", ""))
                content = str(turn.get("content", turn.get("value", "")))
                if role in ("human", "user"):
                    user_content = content
                elif role in ("gpt", "assistant"):
                    ground_truth = content

            if user_content is None:
                raise ValueError("No user turn")
            if ground_truth is None:
                raise ValueError("No assistant turn")

            samples.append({
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_content},
                        ],
                    }
                ],
                "image" : img,
                "answer": ground_truth,
            })

        except Exception as e:
            if skipped < 10:
                print(f"  [WARN] sample {i}: {e}", flush=True)
            skipped += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(dataset)} ...", flush=True)

    print(f"  Ready: {len(samples):,} | skipped: {skipped:,}", flush=True)
    if not samples:
        raise ValueError("All samples skipped!")
    return samples

# ================================================================
# INFERENCE TEST
# ================================================================
def run_inference_test(model, tokenizer, samples):
    print("\n" + "="*60, flush=True)
    print("  INFERENCE TEST (before training)", flush=True)
    print("="*60, flush=True)

    FastVisionModel.for_inference(model)

    for idx in [0, min(100, len(samples) - 1)]:
        sample  = samples[idx]
        img     = sample["image"]
        prompt  = sample["prompt"]
        answer  = sample["answer"]

        question = ""
        for msg in prompt:
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    question = block["text"]

        print(f"\n  --- Sample {idx} ---",                   flush=True)
        print(f"  Image    : {img.size} {img.mode}",         flush=True)
        print(f"  Question : {question[:150]}...",           flush=True)
        print(f"  GT answer: {answer[:100]}",                flush=True)
        print(f"  Model out:", flush=True)

        try:
            inputs = tokenizer(
                img,
                prompt,
                add_special_tokens = False,
                return_tensors     = "pt",
            ).to("cuda")

            streamer = TextStreamer(tokenizer, skip_prompt=True)
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    streamer       = streamer,
                    max_new_tokens = 128,
                    use_cache      = True,
                    temperature    = 0.7,
                    top_p          = 0.8,
                )
        except Exception as e:
            print(f"\n  [ERROR] inference failed: {e}", flush=True)

        print(flush=True)

    print("="*60, flush=True)
    print("  Inference test done.", flush=True)
    print("="*60 + "\n", flush=True)

    FastVisionModel.for_training(model)

# ================================================================
# MODEL LOADER
# ================================================================
def load_model():
    print(f"\n  [1/4] Loading base: {MODEL_NAME}", flush=True)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name             = MODEL_NAME,
        max_seq_length         = MAX_SEQ_LEN,
        load_in_4bit           = LOAD_4BIT,
        fast_inference         = False,
        gpu_memory_utilization = 0.8,
    )
    print(f"  [2/4] Loading SFT adapter: {SFT_ADAPTER}", flush=True)
    model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=False)
    print(f"  [3/4] Merging SFT adapter ...", flush=True)
    model = model.merge_and_unload()
    print(f"  [4/4] Adding GRPO LoRA ...", flush=True)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r                          = LORA_R,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = 0,
        bias                       = "none",
        random_state               = SEED,
        use_rslora                 = False,
        loftq_config               = None,
        use_gradient_checkpointing = "unsloth",
    )
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"  Trainable: {trainable:,}/{total:,} "
        f"({trainable/total*100:.2f}%)",
        flush=True,
    )
    return model, tokenizer

# ================================================================
# MAIN
# ================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── print env / config first ───────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("  ENV / CONFIG", flush=True)
    print("="*60, flush=True)
    print(
        f"  OPENAI_API_KEY  : "
        f"{'SET (' + JUDGE_API_KEY[:8] + '...)' if JUDGE_API_KEY else 'NOT SET !!!'}",
        flush=True,
    )
    print(f"  OPENAI_BASE_URL : {JUDGE_BASE_URL}",    flush=True)
    print(f"  JUDGE_MODEL     : {JUDGE_MODEL}",        flush=True)
    print(f"  MODEL_NAME      : {MODEL_NAME}",         flush=True)
    print(f"  SFT_ADAPTER     : {SFT_ADAPTER}",        flush=True)
    print(f"  PREPARED_PATH   : {PREPARED_PATH}",      flush=True)
    print(f"  OUT_DIR         : {OUT_DIR}",             flush=True)
    print(f"  MAX_SEQ_LEN     : {MAX_SEQ_LEN}",        flush=True)
    print(f"  BATCH_SIZE      : {BATCH_SIZE}",          flush=True)
    print(f"  GRAD_ACCUM      : {GRAD_ACCUM}",          flush=True)
    print(f"  NUM_GENERATIONS : {NUM_GENERATIONS}",     flush=True)
    print(f"  MAX_PROMPT_LEN  : {MAX_PROMPT_LENGTH}",  flush=True)
    print(f"  MAX_COMP_LEN    : {MAX_COMPLETION_LENGTH}", flush=True)
    print(f"  LR              : {LR}",                  flush=True)
    print(f"  NUM_EPOCHS      : {NUM_EPOCHS}",          flush=True)
    print(f"  LORA_R          : {LORA_R}",              flush=True)
    print(f"  SEED            : {SEED}",                flush=True)
    print(f"  REWARD_WEIGHTS  : {REWARD_WEIGHTS}",      flush=True)

    print("\n  --- .env file contents ---", flush=True)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if any(s in k.upper() for s in ["KEY", "SECRET", "TOKEN", "PASSWORD"]):
                        masked = v[:6] + "..." if len(v) > 6 else "***"
                        print(f"    {k} = {masked}", flush=True)
                    else:
                        print(f"    {k} = {v}", flush=True)
                else:
                    print(f"    {line}", flush=True)
    else:
        print(f"    .env NOT FOUND (looked at: {env_path})", flush=True)

    print("="*60 + "\n", flush=True)
    # ── end env print ──────────────────────────────────────────────

    # gpu info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name} | {props.total_memory/1024**3:.1f} GB\n", flush=True)

    # judge api test
    if not JUDGE_API_KEY:
        print(
            "  WARNING: OPENAI_API_KEY not set!\n"
            "  Using fallback scores. Add to .env for real judging.\n",
            flush=True,
        )
    else:
        print(f"  Testing judge API ({JUDGE_MODEL}) ...", flush=True)
        try:
            test = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
            test.chat.completions.create(
                model      = JUDGE_MODEL,
                max_tokens = 5,
                messages   = [{"role": "user", "content": "hi"}],
            )
            print("  Judge API: OK\n", flush=True)
        except Exception as e:
            print(f"  Judge API FAILED: {e}\n  Using fallback scores.\n", flush=True)

    # load data
    train_dataset = load_data(PREPARED_PATH, split=DATASET_SPLIT)

    torch.cuda.empty_cache()
    gc.collect()

    # load model
    model, tokenizer = load_model()

    # inference test before training
    run_inference_test(model, tokenizer, train_dataset)

    # grpo config
    training_args = GRPOConfig(
        learning_rate               = LR,
        adam_beta1                  = ADAM_BETA1,
        adam_beta2                  = ADAM_BETA2,
        weight_decay                = WEIGHT_DECAY,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_8bit",
        logging_steps               = LOG_STEPS,
        log_completions             = False,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_generations             = NUM_GENERATIONS,
        max_prompt_length           = MAX_PROMPT_LENGTH,
        max_completion_length       = MAX_COMPLETION_LENGTH,
        num_train_epochs            = NUM_EPOCHS,
        save_steps                  = SAVE_STEPS,
        max_grad_norm               = MAX_GRAD_NORM,
        report_to                   = "none",
        output_dir                  = OUT_DIR,
        seed                        = SEED,
        importance_sampling_level   = "sequence",
        mask_truncated_completions  = False,
        loss_type                   = "dr_grpo",
        remove_unused_columns       = False,
        bf16                        = is_bf16_supported(),
        fp16                        = not is_bf16_supported(),
    )

    trainer = GRPOTrainer(
        model            = model,
        args             = training_args,
        processing_class = tokenizer,
        reward_funcs     = [
            reward_essential,   # MUST be first — clears cache
            reward_important,
            reward_optional,
            reward_style,
            reward_pitfall,
        ],
        reward_weights   = REWARD_WEIGHTS,
        train_dataset    = train_dataset,
        callbacks        = [ProgressCallback()],
    )

    if torch.cuda.is_available():
        res = torch.cuda.memory_reserved(0)  / 1024**3
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU free: {tot-res:.1f} / {tot:.1f} GB\n", flush=True)

    print("  Starting GRPO training ...\n", flush=True)
    trainer.train(resume_from_checkpoint=RESUME_FROM)

    save_path = os.path.join(OUT_DIR, "final_model")
    print(f"\n  Saving -> {save_path}", flush=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved(0) / 1024**3
        print(f"  Peak VRAM: {peak:.1f} GB", flush=True)

    print("  Done.\n", flush=True)


if __name__ == "__main__":
    main()
