#!/usr/bin/env python
# coding: utf-8

# ============================================================
# ENV (CRITICAL FOR MoE + GRPO)
# ============================================================

import os
os.environ["TRANSFORMERS_NO_MAMBA"] = "1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_CACHE"] = "./hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "./hf_cache/datasets"


# ============================================================
# IMPORTS
# ============================================================
import re
import io
import json
import math
import random
import argparse
import gc
from pathlib import Path
from typing import Any, List, Set

import torch
from PIL import Image
from datasets import Dataset

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import GRPOTrainer, GRPOConfig

# ============================================================
# CONSTANTS
# ============================================================
MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
MAX_SEQ_LEN = 512
MAX_IMAGE_SIDE = 512   # IMPORTANT FOR VRAM
SEED = 42

CID_REGEX = re.compile(r"\bCID\s*\d+\b", re.IGNORECASE)

PROMPT_TEMPLATE = """You must follow these steps exactly:

STEP 1: Identify or infer the chemical compound CID(s).
STEP 2: Perform the task using the CID(s).
STEP 3: Re-attach CID(s) to chemical names.
STEP 4: Verification.

{question}{context}

Use the format:

<REASONING>
...
</REASONING>

<SOLUTION>
...
</SOLUTION>

<VERIFY>
...
</VERIFY>
"""

# ============================================================
# IMAGE HELPERS (LAZY)
# ============================================================
def resize_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= MAX_IMAGE_SIDE:
        return img
    s = MAX_IMAGE_SIDE / m
    return img.resize((int(w * s), int(h * s)))

def load_image(obj: Any) -> Image.Image:
    try:
        if isinstance(obj, Image.Image):
            return resize_image(obj.convert("RGB"))
        if isinstance(obj, str):
            return resize_image(Image.open(obj).convert("RGB"))
        if isinstance(obj, dict) and "bytes" in obj:
            return resize_image(Image.open(io.BytesIO(obj["bytes"])).convert("RGB"))
    except Exception:
        pass
    return Image.new("RGB", (224, 224), "black")

# ============================================================
# DATASET
# ============================================================
def load_jsonl_dataset(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            rows.append({
                "image": x.get("file", ""),
                "question": x.get("question", ""),
                "context": x.get("context", ""),
                "answer": x.get("answer", ""),
            })
    return Dataset.from_list(rows)

def lazy_transform(batch):
    prompts, answers = [], []

    for img, q, ctx, ans in zip(
        batch["image"], batch["question"], batch["context"], batch["answer"]
    ):
        image = load_image(img)
        ctx_block = f"\nContext: {ctx}" if ctx else ""
        text = PROMPT_TEMPLATE.format(question=q, context=ctx_block)

        prompts.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ])
        answers.append(ans)

    return {"prompt": prompts, "reference_answer": answers}

# ============================================================
# TEXT HELPERS
# ============================================================
def get_content(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)

def extract_block(text: str, tag: str) -> str:
    m = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def extract_cids(text: str) -> Set[str]:
    return set(m.upper() for m in CID_REGEX.findall(text))

# ============================================================
# REWARD FUNCTIONS (ALL STANDALONE)
# ============================================================
def reward_format(prompts, completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        t = get_content(c)
        rewards.append(
            1.0 if extract_block(t, "REASONING") and extract_block(t, "SOLUTION") else 0.0
        )
    return rewards

def reward_cid_introduced(prompts, completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        t = get_content(c)
        rewards.append(0.3 if extract_cids(extract_block(t, "REASONING")) else 0.0)
    return rewards

def reward_cid_propagated(prompts, completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        t = get_content(c)
        r = extract_cids(extract_block(t, "REASONING"))
        s = extract_cids(extract_block(t, "SOLUTION"))
        rewards.append(0.4 if (r and r & s) else 0.0)
    return rewards

def reward_cid_hallucinated(prompts, completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        t = get_content(c)
        r = extract_cids(extract_block(t, "REASONING"))
        s = extract_cids(extract_block(t, "SOLUTION"))
        rewards.append(-0.5 if (s - r) else 0.0)
    return rewards

def reward_justification(prompts, completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        reasoning = extract_block(get_content(c), "REASONING").lower()
        r_cids = extract_cids(reasoning)
        if r_cids and not any(w in reasoning for w in ["infer", "based", "identified"]):
            rewards.append(-0.3)
        else:
            rewards.append(0.0)
    return rewards

def reward_verify(prompts, completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        verify = extract_block(get_content(c), "VERIFY").lower()
        if not verify:
            rewards.append(0.0)
        elif any(w in verify for w in ["correct", "verified", "confirmed", "valid"]):
            rewards.append(0.5)
        elif any(w in verify for w in ["incorrect", "wrong", "invalid"]):
            rewards.append(-0.5)
        else:
            rewards.append(0.2)
    return rewards

def reward_correctness(prompts, completions, reference_answer, **kwargs) -> List[float]:
    rewards = []
    for c, ref in zip(completions, reference_answer):
        rewards.append(
            1.0 if ref and ref.lower() in get_content(c).lower() else 0.0
        )
    return rewards

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl_dataset(args.data)
    dataset.set_transform(lazy_transform)

    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LEN,
        device_map="auto",
        use_gradient_checkpointing="unsloth",
    )

    # 🚨 MoE CRITICAL
    model.config.use_cache = False

    model = FastVisionModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        reward_funcs=[
            reward_format,
            reward_cid_introduced,
            reward_cid_propagated,
            reward_cid_hallucinated,
            reward_justification,
            reward_verify,
            reward_correctness,
        ],
        args=GRPOConfig(
            output_dir=args.out,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_generations=2,
            learning_rate=2e-4,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            multi_objective_aggregation="normalize_then_sum",
        ),
    )

    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
