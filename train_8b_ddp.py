#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B Vision LoRA fine-tune with Unsloth + DDP
USAGE:
  Single GPU : python train.py
  Multi  GPU : torchrun --nproc_per_node=8 --master_port=5553 train.py
"""
# ================================================================
# SECTION 0: ENV (before ALL imports)
# ================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["NCCL_TIMEOUT"]            = "7200"
os.environ["NCCL_DEBUG"]              = "WARN"
os.environ["NCCL_IB_DISABLE"]         = "1"
os.environ["NCCL_P2P_LEVEL"]          = "NVL"
os.environ["TORCH_NCCL_BLOCKING_WAIT"]= "1"

HF_CACHE_ROOT = "./hf_cache"
os.environ["HF_HOME"]            = HF_CACHE_ROOT
os.environ["HF_DATASETS_CACHE"]  = f"{HF_CACHE_ROOT}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_ROOT}/transformers"
os.environ["HF_HUB_CACHE"]       = f"{HF_CACHE_ROOT}/hub"

# ================================================================
# SECTION 1: DDP BOOTSTRAP
# ================================================================
import torch
import torch.distributed as dist
import logging
import json
import datetime
import gc
from PIL import Image

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK       = int(os.environ.get("RANK",       0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DIST    = WORLD_SIZE > 1
IS_MAIN    = RANK == 0

if IS_DIST and not dist.is_initialized():
    dist.init_process_group(
        backend = "nccl",
        timeout = datetime.timedelta(hours=2),
    )

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

logging.basicConfig(
    level  = logging.INFO if IS_MAIN else logging.WARNING,
    format = "%(message)s",
)
log = logging.getLogger("train")
log.info(f"RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE}")

# ================================================================
# SECTION 2: IMPORTS
# ================================================================
from datasets import load_from_disk
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainerState, TrainerControl

# ================================================================
# SECTION 3: CONFIG — ONLY CHANGE THESE
# ================================================================
PREPARED_PATH = "./hf_cache/mixed_datasets/abfec89427e45388_prepared"
MODEL_NAME    = "unsloth/Qwen3-VL-8B-Instruct"
OUT_DIR       = "./outputs"

MAX_SEQ_LEN  = 2048
LOAD_4BIT    = True
LORA_R       = 16
LORA_ALPHA   = 16
NUM_EPOCHS   = 3
BATCH_SIZE   = 2       # per device
GRAD_ACCUM   = 4
LR           = 3e-4
SEED         = 42
SAVE_STEPS   = 100
LOG_STEPS    = 5
SAVE_METHOD  = "lora"  # "lora" | "merged_16bit" | "gguf"
RESUME_FROM  = None    # e.g. "./outputs/checkpoint-500"

# ── Image fix config ─────────────────────────────────────────────
MAX_IMAGE_SIZE = 1024   # resize longest side to this
MAX_RATIO      = 180    # pad if aspect ratio exceeds this
                        # Unsloth limit is 200 — we use 180 for safety

# ================================================================
# SECTION 4: IMAGE FIX
# Fixes two problems that cause Unsloth collation to crash:
#   1. Image too large  → slow collation + OOM
#   2. Extreme aspect ratio (e.g. 5000x20 table) → ValueError
#      "absolute aspect ratio must be smaller than 200"
# ================================================================
def fix_image(img: Image.Image) -> Image.Image:
    """
    Fix extreme aspect ratios and oversized images.

    Problem:
        Patent table images can be very wide/thin:
          e.g. 4910 x 20 px → ratio = 245.5
        Unsloth smart_resize rejects ratio > 200:
          ValueError: absolute aspect ratio must be smaller than 200

    Fix:
        Step 1 — resize longest side to MAX_IMAGE_SIZE
        Step 2 — pad short side if ratio still > MAX_RATIO
    """
    # ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size

    # ── Step 1: resize if too large ──────────────────────────────
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        w     = max(1, int(w * scale))
        h     = max(1, int(h * scale))
        img   = img.resize((w, h), Image.LANCZOS)

    # ── Step 2: fix extreme aspect ratio ─────────────────────────
    if min(w, h) > 0:
        ratio = max(w, h) / min(w, h)
        if ratio > MAX_RATIO:
            if w > h:
                # wide image → pad height
                new_h   = max(1, int(w / MAX_RATIO))
                pad     = new_h - h
                new_img = Image.new("RGB", (w, new_h), (255, 255, 255))
                new_img.paste(img, (0, pad // 2))
            else:
                # tall image → pad width
                new_w   = max(1, int(h / MAX_RATIO))
                pad     = new_w - w
                new_img = Image.new("RGB", (new_w, h), (255, 255, 255))
                new_img.paste(img, (pad // 2, 0))
            img = new_img

    return img

# ================================================================
# SECTION 5: LOSS CALLBACK (rank 0 only)
# ================================================================
class VerboseLossCallback(TrainerCallback):

    def __init__(self):
        self.start_time = None
        self.last_loss  = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not IS_MAIN: return
        self.start_time = datetime.datetime.now()
        print(
            f"\n  Started at {self.start_time.strftime('%H:%M:%S')}",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not IS_MAIN or not logs: return
        loss  = logs.get("loss")
        lr    = logs.get("learning_rate")
        epoch = logs.get("epoch")
        if loss is None: return
        self.last_loss = loss
        secs    = (datetime.datetime.now() - self.start_time).seconds
        elapsed = f"{secs//60:02d}:{secs%60:02d}"
        print(
            f"  step {state.global_step:>5}/{state.max_steps}"
            f" | epoch {epoch:.2f}"
            f" | loss {loss:.4f}"
            f" | lr {lr:.2e}"
            f" | {elapsed}",
            flush=True,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        if not IS_MAIN: return
        print(
            f"\n  ✓ Epoch {int(state.epoch)} done"
            f" | loss {self.last_loss}\n",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not IS_MAIN: return
        duration = str(datetime.datetime.now() - self.start_time)
        print(
            f"\n  ✅ Done"
            f" | steps={state.global_step}"
            f" | loss={self.last_loss}"
            f" | {duration}\n",
            flush=True,
        )

# ================================================================
# SECTION 6: DATA FORMATTER
# ================================================================
def load_and_format(path: str) -> list:
    """
    Load Arrow dataset + convert to Unsloth vision format.
    Fixes all images before storing:
      - resize oversized images
      - pad extreme aspect ratios
    This prevents Unsloth collation crash mid-training.
    """
    if IS_MAIN:
        print(f"\n  Loading dataset from {path} ...", flush=True)

    dataset   = load_from_disk(path)
    formatted = []
    skipped   = 0
    fixed     = 0   # images that needed fixing

    for i, sample in enumerate(dataset):
        try:
            conversations = json.loads(sample["conversations"])
            messages      = []
            image_added   = False

            # ── fix image BEFORE attaching to messages ────────────
            # This prevents ValueError mid-training at collation time
            img           = sample["image"]
            original_size = img.size
            img           = fix_image(img)
            if img.size != original_size:
                fixed += 1

            for turn in conversations:
                if turn["role"] == "user" and not image_added:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text",  "text":  turn["content"]},
                        ],
                    })
                    image_added = True
                else:
                    messages.append({
                        "role":    turn["role"],
                        "content": turn["content"],
                    })

            formatted.append({"messages": messages})

        except Exception as e:
            if IS_MAIN:
                print(f"  [WARN] sample {i}: {e}", flush=True)
            skipped += 1

        if IS_MAIN and (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(dataset)} done ...", flush=True)

    if IS_MAIN:
        print(f"  ✓ {len(formatted):,} ready", flush=True)
        print(f"    skipped : {skipped:,}", flush=True)
        print(f"    fixed   : {fixed:,} images (resize/pad)", flush=True)

    # barrier: all ranks finish loading before training starts
    if IS_DIST:
        dist.barrier()

    return formatted

# ================================================================
# SECTION 7: MAIN
# ================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if IS_MAIN:
        eff = BATCH_SIZE * GRAD_ACCUM * WORLD_SIZE
        print(f"\n{'='*50}", flush=True)
        print(f"  model        : {MODEL_NAME}",    flush=True)
        print(f"  data         : {PREPARED_PATH}", flush=True)
        print(f"  gpus         : {WORLD_SIZE}",    flush=True)
        print(f"  eff. batch   : {eff}",           flush=True)
        print(f"  epochs       : {NUM_EPOCHS}",    flush=True)
        print(f"  lora_r       : {LORA_R}",        flush=True)
        print(f"  max_img_size : {MAX_IMAGE_SIZE}", flush=True)
        print(f"  max_ratio    : {MAX_RATIO}",      flush=True)
        print(f"{'='*50}\n", flush=True)

    # ── Data ─────────────────────────────────────────────────────
    train_data = load_and_format(PREPARED_PATH)

    # ── Model ────────────────────────────────────────────────────
    if IS_MAIN:
        print(f"  Loading model on cuda:{LOCAL_RANK} ...", flush=True)

    torch.cuda.empty_cache()
    gc.collect()

    model, processor = FastVisionModel.from_pretrained(
        model_name                 = MODEL_NAME,
        max_seq_length             = MAX_SEQ_LEN,
        load_in_4bit               = LOAD_4BIT,
        device_map                 = {"": LOCAL_RANK},
        use_gradient_checkpointing = "unsloth",
    )

    # barrier: all ranks load model before LoRA
    if IS_DIST:
        dist.barrier()

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r                          = LORA_R,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = 0.0,
        bias                       = "none",
        use_rslora                 = False,
        random_state               = SEED,
    )
    model.config.use_cache = False
    FastVisionModel.for_training(model)

    # barrier: all ranks apply LoRA before trainer builds
    if IS_DIST:
        dist.barrier()

    if IS_MAIN:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(
            f"  ✓ LoRA: {trainable:,}/{total:,}"
            f" ({trainable/total*100:.2f}%)",
            flush=True,
        )

    # ── SFT Config ───────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir                    = OUT_DIR,
        num_train_epochs              = NUM_EPOCHS,
        per_device_train_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        learning_rate                 = LR,
        lr_scheduler_type             = "cosine",
        warmup_ratio                  = 0.03,
        weight_decay                  = 0.01,
        max_grad_norm                 = 1.0,
        bf16                          = torch.cuda.is_bf16_supported(),
        fp16                          = not torch.cuda.is_bf16_supported(),
        optim                         = "adamw_8bit",
        seed                          = SEED,
        # ── DDP fixes ────────────────────────────────────────────
        ddp_find_unused_parameters    = False,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        dataloader_pin_memory         = False,
        dataloader_num_workers        = 0,
        # ── Logging ──────────────────────────────────────────────
        logging_steps                 = LOG_STEPS,
        logging_first_step            = True,
        report_to                     = "none",
        # ── Saving ───────────────────────────────────────────────
        save_steps                    = SAVE_STEPS,
        save_total_limit              = 3,
        save_strategy                 = "steps",
        # ── Vision ───────────────────────────────────────────────
        remove_unused_columns         = False,
        dataset_text_field            = "",
        dataset_kwargs                = {"skip_prepare_dataset": True},
        max_seq_length                = MAX_SEQ_LEN,
    )

    # ── Trainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model         = model,
        tokenizer     = processor,
        data_collator = UnslothVisionDataCollator(model, processor),
        train_dataset = train_data,
        args          = sft_config,
        callbacks     = [VerboseLossCallback()],
    )

    if IS_MAIN:
        print(f"  Samples : {len(train_data):,}", flush=True)
        print(f"\n{'='*50}", flush=True)
        print(f"  TRAINING", flush=True)
        print(f"{'='*50}\n", flush=True)

        if torch.cuda.is_available():
            res  = torch.cuda.memory_reserved(LOCAL_RANK)  / 1024**3
            allc = torch.cuda.memory_allocated(LOCAL_RANK) / 1024**3
            tot  = torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1024**3
            print(f"  GPU before training:", flush=True)
            print(f"    total     : {tot:.1f} GB",  flush=True)
            print(f"    allocated : {allc:.1f} GB", flush=True)
            print(f"    free      : {tot-res:.1f} GB", flush=True)

    # ── Train ────────────────────────────────────────────────────
    result = trainer.train(resume_from_checkpoint=RESUME_FROM)

    # ── Save rank 0 only ─────────────────────────────────────────
    if IS_DIST:
        dist.barrier()

    if IS_MAIN:
        save_path = os.path.join(OUT_DIR, "final_model")
        print(f"\n  Saving → {save_path}", flush=True)

        if SAVE_METHOD == "lora":
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
        elif SAVE_METHOD == "merged_16bit":
            model.save_pretrained_merged(
                save_path, processor, save_method="merged_16bit")
        elif SAVE_METHOD == "gguf":
            model.save_pretrained_gguf(
                save_path, processor, quantization_method="q4_k_m")

        summary = {
            "model"         : MODEL_NAME,
            "world_size"    : WORLD_SIZE,
            "epochs"        : NUM_EPOCHS,
            "lora_r"        : LORA_R,
            "training_loss" : result.training_loss,
            "save_path"     : save_path,
            "completed_at"  : datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_reserved(LOCAL_RANK) / 1024**3
            print(f"  Peak VRAM : {peak:.1f} GB", flush=True)

        print(f"  ✅ Done → {save_path}\n", flush=True)

    # ── Cleanup ──────────────────────────────────────────────────
    if IS_DIST and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    main()
