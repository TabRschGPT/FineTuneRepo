#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B Vision LoRA fine-tune with Unsloth + DDP
USAGE:
  Single GPU : python train.py
  Multi  GPU : torchrun --nproc_per_node=4 --master_port=5553 train.py
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
        backend="nccl",
        timeout=datetime.timedelta(hours=2),
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
from datasets import load_dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# ================================================================
# SECTION 3: CONFIG
# ================================================================
PREPARED_PATH  = "vietmed/sft_16k_mix"
DATASET_SPLIT  = "train"
MODEL_NAME     = "unsloth/Qwen3-VL-8B-Instruct"
OUT_DIR        = "./outputs"
MAX_SEQ_LEN    = 2048
LOAD_4BIT      = True
LORA_R         = 16
LORA_ALPHA     = 16
NUM_EPOCHS     = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 3e-4
SEED           = 42
SAVE_STEPS     = 100
LOG_STEPS      = 5
SAVE_METHOD    = "lora"
RESUME_FROM    = None

# ── Image constraints ─────────────────────────────────────────────
# Unsloth smart_resize hard limit is 200
# We use 150 with generous padding to stay well clear
MAX_IMAGE_SIZE  = 1024   # longest side pixel limit
MAX_RATIO       = 150    # ← lowered from 180; Unsloth limit is 200
                         #   150 gives safe margin even after proportional resize
MIN_SHORT_SIDE  = 8      # prevent degenerate 1px images

# ================================================================
# SECTION 4: IMAGE FIX  (fully corrected)
#
# Root cause of crash:
#   A 4910x20 image has ratio = 245.5
#   After proportional resize to longest=1024:
#     w=1024, h=max(1, int(20*(1024/4910))) = max(1,4) = 4
#     ratio = 1024/4 = 256  ← STILL > 200
#   Unsloth's smart_resize then raises:
#     ValueError: absolute aspect ratio must be smaller than 200
#
# Fix order (CRITICAL — do ratio fix BEFORE proportional resize):
#   Step 1: Fix ratio FIRST by padding the short side
#   Step 2: Then resize longest side down to MAX_IMAGE_SIZE
#   Step 3: Final ratio check + clamp (safety net)
#
# Why ratio first?
#   If we resize first, extreme ratios stay extreme (just smaller).
#   If we pad first, the image becomes "reasonable" shaped,
#   then resize brings it to the right pixel count.
# ================================================================
def fix_image(img: Image.Image) -> Image.Image:
    """
    Guarantee output image satisfies:
      - longest side <= MAX_IMAGE_SIZE
      - aspect ratio <= MAX_RATIO  (Unsloth limit is 200)
      - both sides >= MIN_SHORT_SIDE
      - mode == RGB
    """
    # ── Ensure RGB ────────────────────────────────────────────────
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size

    # ── Degenerate image guard ────────────────────────────────────
    w = max(w, 1)
    h = max(h, 1)
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)

    # ── Step 1: Fix extreme aspect ratio FIRST (pad short side) ───
    # Must happen before proportional resize, because resize
    # preserves ratio — a 245:1 image stays 245:1 after resize.
    ratio = max(w, h) / min(w, h)
    if ratio > MAX_RATIO:
        if w >= h:
            # Wide image → pad height to satisfy ratio
            # new_h * MAX_RATIO >= w  →  new_h = ceil(w / MAX_RATIO)
            new_h   = max(MIN_SHORT_SIDE, -(-w // MAX_RATIO))  # ceiling div
            pad_top = (new_h - h) // 2
            canvas  = Image.new("RGB", (w, new_h), (255, 255, 255))
            canvas.paste(img, (0, pad_top))
            img, w, h = canvas, w, new_h
        else:
            # Tall image → pad width
            new_w    = max(MIN_SHORT_SIDE, -(-h // MAX_RATIO))
            pad_left = (new_w - w) // 2
            canvas   = Image.new("RGB", (new_w, h), (255, 255, 255))
            canvas.paste(img, (pad_left, 0))
            img, w, h = canvas, new_w, h

    # ── Step 2: Resize longest side down to MAX_IMAGE_SIZE ────────
    long_side = max(w, h)
    if long_side > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / long_side
        new_w = max(MIN_SHORT_SIDE, int(w * scale))
        new_h = max(MIN_SHORT_SIDE, int(h * scale))
        img   = img.resize((new_w, new_h), Image.LANCZOS)
        w, h  = new_w, new_h

    # ── Step 3: Safety net — final ratio clamp ────────────────────
    # Handles rounding edge cases from integer division above
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > MAX_RATIO:
        if w >= h:
            new_h   = max(MIN_SHORT_SIDE, -(-w // MAX_RATIO))
            pad_top = (new_h - h) // 2
            canvas  = Image.new("RGB", (w, new_h), (255, 255, 255))
            canvas.paste(img, (0, pad_top))
            img = canvas
        else:
            new_w    = max(MIN_SHORT_SIDE, -(-h // MAX_RATIO))
            pad_left = (new_w - w) // 2
            canvas   = Image.new("RGB", (new_w, h), (255, 255, 255))
            canvas.paste(img, (pad_left, 0))
            img = canvas

    # ── Step 4: Guarantee minimum short side ─────────────────────
    w, h = img.size
    if min(w, h) < MIN_SHORT_SIDE:
        if w < h:
            img = img.resize((MIN_SHORT_SIDE, h), Image.LANCZOS)
        else:
            img = img.resize((w, MIN_SHORT_SIDE), Image.LANCZOS)

    return img


def validate_image(img: Image.Image, idx: int) -> bool:
    """
    Verify image will pass Unsloth's smart_resize.
    Returns True if safe, False if should skip.
    """
    w, h   = img.size
    ratio  = max(w, h) / max(min(w, h), 1)
    if ratio >= 200:
        if IS_MAIN:
            print(
                f"  [ERROR] sample {idx}: ratio {ratio:.1f} after fix! "
                f"size={img.size} — skipping",
                flush=True,
            )
        return False
    if min(w, h) < 1:
        if IS_MAIN:
            print(f"  [ERROR] sample {idx}: degenerate size {img.size} — skipping", flush=True)
        return False
    return True


# ================================================================
# SECTION 5: LOSS CALLBACK
# ================================================================
class VerboseLossCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_loss  = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not IS_MAIN: return
        self.start_time = datetime.datetime.now()
        print(f"\n  Started at {self.start_time.strftime('%H:%M:%S')}", flush=True)

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
        print(f"\n  ✓ Epoch {int(state.epoch)} done | loss {self.last_loss}\n", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if not IS_MAIN: return
        duration = str(datetime.datetime.now() - self.start_time)
        print(
            f"\n  ✅ Done | steps={state.global_step}"
            f" | loss={self.last_loss} | {duration}\n",
            flush=True,
        )


# ================================================================
# SECTION 6: DATA FORMATTER
# ================================================================
def load_and_format(path: str, split: str = "train") -> list:
    """
    Load HuggingFace dataset + convert to Unsloth vision format.
    All images are fixed and VALIDATED before storing.
    """
    if IS_MAIN:
        print(f"\n  Loading dataset '{path}' split='{split}' ...", flush=True)

    raw = load_dataset(path)

    if IS_MAIN:
        print(f"  Available splits : {list(raw.keys())}", flush=True)

    # Select split with fallback
    if split in raw:
        dataset = raw[split]
    elif "train" in raw:
        dataset = raw["train"]
        if IS_MAIN:
            print(f"  [WARN] Split '{split}' not found — using 'train'", flush=True)
    else:
        first = list(raw.keys())[0]
        dataset = raw[first]
        if IS_MAIN:
            print(f"  [WARN] Using first available split: '{first}'", flush=True)

    if IS_MAIN:
        print(f"  Split size : {len(dataset):,}", flush=True)
        print(f"  Columns    : {dataset.column_names}", flush=True)

    if len(dataset) == 0:
        raise ValueError(f"Dataset split '{split}' is empty!")

    # Inspect first sample
    if IS_MAIN:
        s0 = dataset[0]
        print(f"\n  First sample preview:", flush=True)
        for k, v in s0.items():
            if k == "image":
                if hasattr(v, "size"):
                    ratio = max(v.size) / max(min(v.size), 1)
                    print(f"    image : PIL {v.size} mode={v.mode} ratio={ratio:.1f}", flush=True)
                else:
                    print(f"    image : {type(v)}", flush=True)
            else:
                print(f"    {k} : {str(v)[:120]}", flush=True)

    formatted  = []
    skipped    = 0
    fixed      = 0
    bad_ratio  = 0  # images that failed validation even after fix

    for i, sample in enumerate(dataset):
        try:
            # ── Parse conversations ────────────────────────────────
            raw_conv = sample["conversations"]
            if isinstance(raw_conv, str):
                conversations = json.loads(raw_conv)
            elif isinstance(raw_conv, list):
                conversations = raw_conv
            else:
                raise ValueError(f"Unexpected conversations type: {type(raw_conv)}")

            # ── Get image ─────────────────────────────────────────
            img = sample["image"]
            if isinstance(img, str):
                img = Image.open(img)
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unexpected image type: {type(img)}")

            # ── Fix image ─────────────────────────────────────────
            original_size = img.size
            img = fix_image(img)
            if img.size != original_size:
                fixed += 1

            # ── Validate image will pass Unsloth's smart_resize ───
            # This is the CRITICAL check — skip bad images rather
            # than crash mid-training at collation time
            if not validate_image(img, i):
                bad_ratio += 1
                skipped   += 1
                continue

            # ── Build messages ────────────────────────────────────
            messages    = []
            image_added = False

            for turn in conversations:
                role    = turn.get("role", turn.get("from", ""))
                content = turn.get("content", turn.get("value", ""))

                # Normalize role names
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant"):
                    role = "assistant"

                if role == "user" and not image_added:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text",  "text":  content},
                        ],
                    })
                    image_added = True
                else:
                    messages.append({
                        "role":    role,
                        "content": content,
                    })

            if len(messages) < 2:
                raise ValueError(f"Too few messages: {len(messages)}")
            if not image_added:
                raise ValueError("No user turn — image not attached")

            formatted.append({"messages": messages})

        except Exception as e:
            if IS_MAIN and skipped < 20:
                print(f"  [WARN] sample {i}: {type(e).__name__}: {e}", flush=True)
            skipped += 1

        if IS_MAIN and (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(dataset)} processed ...", flush=True)

    # ── Summary ───────────────────────────────────────────────────
    if IS_MAIN:
        print(f"\n  ✓ {len(formatted):,} ready", flush=True)
        print(f"    skipped   : {skipped:,}", flush=True)
        print(f"    bad ratio : {bad_ratio:,} (skipped after fix failed)", flush=True)
        print(f"    fixed     : {fixed:,} images (resize/pad)", flush=True)

    if len(formatted) == 0:
        raise ValueError(
            "All samples were skipped! "
            f"Dataset columns: {dataset.column_names}"
        )

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
        print(f"  model        : {MODEL_NAME}",     flush=True)
        print(f"  data         : {PREPARED_PATH}",  flush=True)
        print(f"  split        : {DATASET_SPLIT}",  flush=True)
        print(f"  gpus         : {WORLD_SIZE}",     flush=True)
        print(f"  eff. batch   : {eff}",            flush=True)
        print(f"  epochs       : {NUM_EPOCHS}",     flush=True)
        print(f"  lora_r       : {LORA_R}",         flush=True)
        print(f"  max_img_size : {MAX_IMAGE_SIZE}",  flush=True)
        print(f"  max_ratio    : {MAX_RATIO}",       flush=True)
        print(f"{'='*50}\n", flush=True)

    # ── Data ──────────────────────────────────────────────────────
    train_data = load_and_format(PREPARED_PATH, split=DATASET_SPLIT)

    assert len(train_data) > 0, "train_data is empty!"
    assert len(train_data) >= WORLD_SIZE, (
        f"Dataset ({len(train_data)}) < WORLD_SIZE ({WORLD_SIZE})"
    )

    if IS_MAIN:
        print(f"  ✓ Training samples: {len(train_data):,}", flush=True)

    # ── Model ─────────────────────────────────────────────────────
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

    if IS_DIST:
        dist.barrier()

    if IS_MAIN:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  ✓ LoRA: {trainable:,}/{total:,} ({trainable/total*100:.2f}%)", flush=True)

    # ── SFT Config ────────────────────────────────────────────────
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
        ddp_find_unused_parameters    = False,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        dataloader_pin_memory         = False,
        dataloader_num_workers        = 0,
        logging_steps                 = LOG_STEPS,
        logging_first_step            = True,
        report_to                     = "none",
        save_steps                    = SAVE_STEPS,
        save_total_limit              = 3,
        save_strategy                 = "steps",
        remove_unused_columns         = False,
        dataset_text_field            = "",
        dataset_kwargs                = {"skip_prepare_dataset": True},
        max_seq_length                = MAX_SEQ_LEN,
    )

    # ── Trainer ───────────────────────────────────────────────────
    trainer = SFTTrainer(
        model         = model,
        tokenizer     = processor,
        data_collator = UnslothVisionDataCollator(model, processor),
        train_dataset = train_data,
        args          = sft_config,
        callbacks     = [VerboseLossCallback()],
    )

    if IS_MAIN:
        print(f"\n{'='*50}\n  TRAINING\n{'='*50}\n", flush=True)
        if torch.cuda.is_available():
            res  = torch.cuda.memory_reserved(LOCAL_RANK)  / 1024**3
            allc = torch.cuda.memory_allocated(LOCAL_RANK) / 1024**3
            tot  = torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1024**3
            print(f"  GPU  total : {tot:.1f} GB", flush=True)
            print(f"  GPU  alloc : {allc:.1f} GB", flush=True)
            print(f"  GPU  free  : {tot-res:.1f} GB", flush=True)

    # ── Train ─────────────────────────────────────────────────────
    result = trainer.train(resume_from_checkpoint=RESUME_FROM)

    # ── Save ──────────────────────────────────────────────────────
    if IS_DIST:
        dist.barrier()

    if IS_MAIN:
        save_path = os.path.join(OUT_DIR, "final_model")
        print(f"\n  Saving → {save_path}", flush=True)

        if SAVE_METHOD == "lora":
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
        elif SAVE_METHOD == "merged_16bit":
            model.save_pretrained_merged(save_path, processor, save_method="merged_16bit")
        elif SAVE_METHOD == "gguf":
            model.save_pretrained_gguf(save_path, processor, quantization_method="q4_k_m")

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

    # ── Cleanup ───────────────────────────────────────────────────
    if IS_DIST and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ================================================================
if __name__ == "__main__":
    main()
