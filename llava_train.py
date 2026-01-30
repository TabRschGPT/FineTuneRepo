#!/usr/bin/env python
# coding: utf-8

# ============================================================
# QLoRA + BitsAndBytes + LLaVA-Med (MAP-STYLE DATASET)
# ============================================================

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DISABLE_CACHING_ALLOCATOR_WARMUP"] = "1"

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ============================================================
# Constants
# ============================================================

MODEL_ID = "Eren-Senoglu/llava-med-v1.5-mistral-7b-hf"

MAX_IMAGE_SIDE = 640          # SAFE
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

# ============================================================
# Map-style Dataset
# ============================================================

class VisionJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        image = load_image(
            resolve_image_path(self.jsonl_path, row["file"])
        )

        prompt = PROMPT_STYLE.format(
            question=row.get("question", ""),
            context=row.get("context", ""),
        )

        return {
            "image": image,
            "prompt": prompt,
            "answer": row.get("answer", ""),
        }

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

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    train_ds = VisionJsonlDataset(args.domain_data)

    # ------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )

    processor.patch_size = 14
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ------------------------------------------------------------
    # BitsAndBytes (4-bit QLoRA)
    # ------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}

    print("Loading model (bnb 4-bit)")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------
    # Memory Savers
    # ------------------------------------------------------------
    model.vision_tower.requires_grad_(False)
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # LoRA (SAFE)
    # ------------------------------------------------------------
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        remove_unused_columns=False,
        bf16=True,
        report_to="wandb",
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
