#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA fine-tuning for Qwen3-VL with Unsloth + DDP (torchrun)
=========================================================
Usage (example, 4 GPUs):
  torchrun --nproc_per_node=4 train_vl_ddp.py
"""

# ===================== HF CACHE (SET FIRST) =====================
import os
HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "./hf_cache")
os.environ["HF_HOME"] = HF_CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_ROOT}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{HF_CACHE_ROOT}/datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ===============================================================

import torch
import torch.distributed as dist
import logging
from datasets import load_dataset
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig

# ===================== DDP BOOTSTRAP =====================
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK       = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DIST    = WORLD_SIZE > 1

if IS_DIST and not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="env://")

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO if RANK == 0 else logging.WARNING)
log = logging.getLogger("train_vl_ddp")
log.info(f"RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE} device={device}")

# ===================== USER CONFIG =====================
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
DATA_FILE  = "./final_train_normalized.jsonl"
OUT_DIR    = "./vl_qwen3_ddp_lora"

MAX_SEQ_LEN = 4096
NUM_EPOCHS  = 3
SEED        = 1337

LR            = 3e-4
WEIGHT_DECAY  = 0.1
MAX_GRAD_NORM = 1.0

LORA_RANK = 16

TARGET_GLOBAL_BATCH = 8
PER_DEVICE_BATCH   = max(1, TARGET_GLOBAL_BATCH // max(1, WORLD_SIZE))
GRAD_ACCUM          = 1

torch.manual_seed(SEED)

# ===================== MODEL LOAD (PER-RANK) =====================
device_map = {"": f"cuda:{LOCAL_RANK}"} if torch.cuda.is_available() else None

model, tokenizer = FastVisionModel.from_pretrained(
    model_name       = MODEL_NAME,
    max_seq_length   = MAX_SEQ_LEN,
    load_in_4bit     = True,
    fast_inference   = False,
    device_map       = device_map,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r                          = LORA_RANK,
    lora_alpha                 = LORA_RANK,
    lora_dropout               = 0.0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = SEED,
)

# ===================== DATA =====================
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def build_text(ex):
    text = f"""<image>
Answer the following visual-document question.

Explain your reasoning in <THINKING> tags, then give the final answer in <ANSWER> tags.

Artefact type: {ex.get("artefact_type", "unknown")}
Context: {ex.get("context", "")}

Question:
{ex["question"]}

<THINKING>
{ex.get("rationale", "")}
</THINKING>

<ANSWER>
{ex["answer"]}
</ANSWER>
"""
    return {"text": text}

dataset = dataset.map(build_text, desc="Build VL text")
dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])
dataset = dataset.shuffle(seed=SEED)

# ===================== TRAIN CONFIG =====================
training_args = SFTConfig(
    dataset_text_field          = "text",
    per_device_train_batch_size = PER_DEVICE_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs            = NUM_EPOCHS,
    learning_rate               = LR,
    lr_scheduler_type           = "linear",
    warmup_ratio                = 0.0,
    max_grad_norm               = MAX_GRAD_NORM,
    weight_decay                = WEIGHT_DECAY,
    seed                        = SEED,
    output_dir                  = OUT_DIR,
    save_strategy               = "epoch",
    logging_steps               = 10,
    optim                       = "adamw_8bit",
    ddp_find_unused_parameters  = False,
    gradient_checkpointing      = True,
    report_to                   = "wandb",
)

# ===================== TRAINER =====================
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args          = training_args,
)

if RANK == 0:
    log.info("Starting training...")

trainer.train()

# ===================== SAVE (RANK 0 ONLY) =====================
if RANK == 0:
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"✅ DONE. Saved to {OUT_DIR}")

# ===================== CLEANUP =====================
if IS_DIST and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
