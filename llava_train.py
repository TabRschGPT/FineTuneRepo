#!/usr/bin/env python
# coding: utf-8

# ============================================================
# LEGACY MODE: Transformers 4.37.x | TRL | LLaVA-Med
# ============================================================

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import io
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from datasets import IterableDataset

from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ============================================================
# Constants
# ============================================================

MODEL_ID = "Eren-Senoglu/llava-med-v1.5-mistral-7b-hf"
MAX_IMAGE_SIDE = 1024
MAX_SEQ_LEN = 2048

PROMPT_STYLE = """You are an expert assistant.

Question:
{question}

Context:
{context}
"""

# ============================================================
# Helpers
# ============================================================

def resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= MAX_IMAGE_SIDE:
        return img
    s = MAX_IMAGE_SIDE / float(m)
    return img.resize((int(w * s), int(h * s)))

def load_image(path: str) -> Image.Image:
    return resize_if_needed(Image.open(path).convert("RGB"))

def resolve_image_path(jsonl_path: str, img_path: str) -> str:
    base = Path(jsonl_path).parent.resolve()
    return str(base / img_path) if not os.path.isabs(img_path) else img_path

def count_samples(jsonl_path: str) -> int:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

# ============================================================
# Dataset
# ============================================================

def build_example(row, image):
    prompt = PROMPT_STYLE.format(
        question=row.get("question", ""),
        context=row.get("context", ""),
    )
    return {
        "image": image,
        "prompt": prompt,
        "answer": row.get("answer", ""),
    }

def dataset_generator(jsonl_path: str, seed: int, epochs: int):
    rng = random.Random(seed)
    for _ in range(epochs):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rng.shuffle(rows)
        for row in rows:
            if not row.get("file"):
                continue
            try:
                image = load_image(
                    resolve_image_path(jsonl_path, row["file"])
                )
            except Exception:
                continue
            yield build_example(row, image)

# ============================================================
# Collator
# ============================================================

class VisionCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, batch):
        texts, prompts, images = [], [], []

        for ex in batch:
            prompt = f"<image>\nUSER: {ex['prompt']}\nASSISTANT:"
            full_text = prompt + " " + ex["answer"]

            prompts.append(prompt)
            texts.append(full_text)
            images.append(ex["image"])

        enc = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )

        prompt_ids = self.tokenizer(
            prompts, padding=False, add_special_tokens=True
        )["input_ids"]

        labels = enc["input_ids"].clone()
        for i, p in enumerate(prompt_ids):
            labels[i, : len(p)] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        enc["labels"] = labels
        return enc

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Compute max_steps (REQUIRED for IterableDataset)
    # ============================================================
    num_samples = count_samples(args.domain_data)
    effective_bs = args.bs * args.grad_accum
    max_steps = math.ceil(num_samples / effective_bs) * args.epochs

    print(f"Samples: {num_samples}")
    print(f"Max steps: {max_steps}")

    train_ds = IterableDataset.from_generator(
        lambda: dataset_generator(
            args.domain_data, args.seed, args.epochs
        )
    )

    # ============================================================
    # Load processor + PATCH
    # ============================================================
    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )

    # 🔥 THE ONLY PATCH THAT MATTERS
    processor.patch_size = 14
    assert processor.patch_size == 14

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ============================================================
    # Load model
    # ============================================================
    print("Loading model")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )

    # ============================================================
    # Trainer
    # ============================================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=500,
        remove_unused_columns=False,
        report_to="wandb",
        fp16=False,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        data_collator=VisionCollator(processor),
        args=training_args,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("✅ Training complete")

if __name__ == "__main__":
    main()
