#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA fine-tuning for Qwen3-VL-8B with Unsloth + DDP
USAGE:
  CUDA_VISIBLE_DEVICES=0 python train_8b.py
  torchrun --nproc_per_node=4 train_8b.py
"""

# ===================== ENV =====================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "./hf_cache")
os.environ["HF_HOME"] = HF_CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_ROOT}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{HF_CACHE_ROOT}/datasets"

# =================================================

import torch
import torch.distributed as dist
import logging
from datasets import load_dataset
from pathlib import Path
from PIL import Image

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ===================== DDP =====================
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK       = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DIST    = WORLD_SIZE > 1

if IS_DIST and not dist.is_initialized():
    dist.init_process_group("nccl")

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

logging.basicConfig(level=logging.INFO if RANK == 0 else logging.WARNING)
log = logging.getLogger("train")

# ===================== CONFIG =====================
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
DATA_FILE  = "./aggregate_converted.jsonl"
OUT_DIR    = "./vl_qwen3_lora"

MAX_SEQ_LEN = 2048
LORA_RANK   = 8
NUM_EPOCHS  = 3
SEED        = 1337

LR            = 2e-4
WEIGHT_DECAY  = 0.1
MAX_GRAD_NORM = 1.0

PER_DEVICE_BATCH = 1
GRAD_ACCUM       = 2

torch.manual_seed(SEED)

# ===================== MODEL =====================
model, tokenizer = FastVisionModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,
    device_map     = "auto",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = False,
    r                          = LORA_RANK,
    lora_alpha                 = LORA_RANK,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = SEED,
)

model.config.use_cache = False

# ===================== COLLATOR =====================
data_collator = UnslothVisionDataCollator(model, tokenizer)

# ===================== DATA =====================
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
BASE_IMAGE_DIR = Path(".")

def build_messages(ex):
    img_path = ex.get("file", None)
    if img_path is None:
        return None

    full_path = BASE_IMAGE_DIR / img_path
    if not full_path.exists():
        return None

    try:
        image = Image.open(full_path).convert("RGB")
    except Exception:
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        # 🔑 THIS IS THE IMPORTANT PART
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": f"""Question:
{ex["question"]}

Context:
{ex.get("context", "")}
"""
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Rationale:
{ex.get("rationale", "")}

Answer:
{ex["answer"]}
"""
                    }
                ],
            },
        ]
    }

dataset = dataset.map(
    build_messages,
    remove_columns=dataset.column_names,
    desc="Building messages",
)

dataset = dataset.filter(lambda x: x is not None)
dataset = dataset.shuffle(seed=SEED)

# ===================== TRAIN ARGS =====================
training_args = SFTConfig(
    dataset_text_field          = None,
    remove_unused_columns       = False,
    per_device_train_batch_size = PER_DEVICE_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs            = NUM_EPOCHS,
    learning_rate               = LR,
    lr_scheduler_type           = "linear",
    max_grad_norm               = MAX_GRAD_NORM,
    weight_decay                = WEIGHT_DECAY,
    seed                        = SEED,
    output_dir                  = OUT_DIR,
    save_strategy               = "epoch",
    logging_steps               = 10,
    optim                       = "paged_adamw_8bit",
    gradient_checkpointing      = True,
    ddp_find_unused_parameters  = False,
    report_to                   = "none",
)

# ===================== TRAIN =====================
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    data_collator = data_collator,
    args          = training_args,
)

if RANK == 0:
    log.info("🚀 Training started")

trainer.train()

# ===================== SAVE =====================
if RANK == 0:
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\n✅ DONE. Saved to {OUT_DIR}\n")

# ===================== CLEANUP =====================
if IS_DIST and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
