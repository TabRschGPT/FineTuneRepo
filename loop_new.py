#!/usr/bin/env python
# ==========================
# ENV
# ==========================
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
HF_CACHE_ROOT = "./hf_cache"
os.environ["HF_HOME"]            = HF_CACHE_ROOT
os.environ["HF_DATASETS_CACHE"]  = f"{HF_CACHE_ROOT}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_ROOT}/transformers"
os.environ["HF_HUB_CACHE"]       = f"{HF_CACHE_ROOT}/hub"

# ==========================
# IMPORTS
# ==========================
import argparse
import gc
import io
import json
import random
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterator, List, Any, Optional

import torch
from PIL import Image as PILImage
from tqdm import tqdm
from datasets import load_dataset, Dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ==========================
# TOKENS
# ==========================
REASONING_START = "<details>"
REASONING_END   = "</details>"
SOLUTION_START  = "<solution>"
SOLUTION_END    = "</solution>"

# ==========================
# IMAGE CONSTRAINTS
# ==========================
MAX_IMAGE_SIZE = 1024
MAX_RATIO      = 150
MIN_SHORT_SIDE = 8

# ==========================
# CONFIG
# ==========================
@dataclass
class AlgoConfig:
    base_model:           str   = "unsloth/Qwen3-VL-8B-Instruct"
    iterations_k:         int   = 3
    step_size_j:          int   = 50
    max_seq_length:       int   = 1024
    learning_rate:        float = 2e-4
    epochs_per_iter:      int   = 1
    batch_size:           int   = 1
    grad_accum:           int   = 16
    warmup_steps:         int   = 5
    gen_max_new_tokens:   int   = 128
    cls_max_new_tokens:   int   = 16
    force_float_solution: bool  = True
    out_dir:              str   = "outputs"


# ==========================
# MEMORY HELPERS
# ==========================
def free_memory():
    """Aggressively free CPU and GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory(tag: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        print(
            f"  [MEM{' ' + tag if tag else ''}] "
            f"allocated={allocated:.2f}GB  reserved={reserved:.2f}GB",
            flush=True,
        )


# ==========================
# IMAGE HELPERS
# ==========================
def fix_image(img: PILImage.Image) -> PILImage.Image:
    """
    Guarantee:
      - longest side <= MAX_IMAGE_SIZE
      - aspect ratio <= MAX_RATIO
      - both sides >= MIN_SHORT_SIDE
      - mode == RGB
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    w = max(w, 1)
    h = max(h, 1)
    if img.size != (w, h):
        img = img.resize((w, h), PILImage.LANCZOS)

    # Step 1: Fix extreme aspect ratio FIRST
    ratio = max(w, h) / min(w, h)
    if ratio > MAX_RATIO:
        if w >= h:
            new_h   = max(MIN_SHORT_SIDE, -(-w // MAX_RATIO))
            pad_top = (new_h - h) // 2
            canvas  = PILImage.new("RGB", (w, new_h), (255, 255, 255))
            canvas.paste(img, (0, pad_top))
            img, w, h = canvas, w, new_h
        else:
            new_w    = max(MIN_SHORT_SIDE, -(-h // MAX_RATIO))
            pad_left = (new_w - w) // 2
            canvas   = PILImage.new("RGB", (new_w, h), (255, 255, 255))
            canvas.paste(img, (pad_left, 0))
            img, w, h = canvas, new_w, h

    # Step 2: Resize longest side down
    long_side = max(w, h)
    if long_side > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / long_side
        new_w = max(MIN_SHORT_SIDE, int(w * scale))
        new_h = max(MIN_SHORT_SIDE, int(h * scale))
        img   = img.resize((new_w, new_h), PILImage.LANCZOS)
        w, h  = new_w, new_h

    # Step 3: Safety net
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > MAX_RATIO:
        if w >= h:
            new_h   = max(MIN_SHORT_SIDE, -(-w // MAX_RATIO))
            pad_top = (new_h - h) // 2
            canvas  = PILImage.new("RGB", (w, new_h), (255, 255, 255))
            canvas.paste(img, (0, pad_top))
            img = canvas
        else:
            new_w    = max(MIN_SHORT_SIDE, -(-h // MAX_RATIO))
            pad_left = (new_w - w) // 2
            canvas   = PILImage.new("RGB", (new_w, h), (255, 255, 255))
            canvas.paste(img, (pad_left, 0))
            img = canvas

    # Step 4: Minimum short side
    w, h = img.size
    if min(w, h) < MIN_SHORT_SIDE:
        if w < h:
            img = img.resize((MIN_SHORT_SIDE, h), PILImage.LANCZOS)
        else:
            img = img.resize((w, MIN_SHORT_SIDE), PILImage.LANCZOS)

    return img


def validate_image(img: PILImage.Image, idx: int) -> bool:
    w, h  = img.size
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio >= 200:
        print(
            f"  [ERROR] sample {idx}: ratio {ratio:.1f} after fix! "
            f"size={img.size} — skipping",
            flush=True,
        )
        return False
    if min(w, h) < 1:
        print(
            f"  [ERROR] sample {idx}: degenerate size {img.size} — skipping",
            flush=True,
        )
        return False
    return True


# ==========================
# DATA LOADING
# ==========================
def load_and_format_domain(
    hf_dataset_name: str,
    split:           str = "train",
    num_samples:     int = None,
    seed:            int = 42,
) -> List[Dict]:
    """
    Load HF dataset with conversations + image format.
    Extracts question (first human turn) and answer (first assistant turn).
    """
    print(f"\n  Loading dataset '{hf_dataset_name}' split='{split}' ...", flush=True)
    raw = load_dataset(hf_dataset_name)

    # Split fallback
    if split in raw:
        dataset = raw[split]
    elif "train" in raw:
        dataset = raw["train"]
        print(f"  [WARN] Split '{split}' not found — using 'train'", flush=True)
    else:
        first   = list(raw.keys())[0]
        dataset = raw[first]
        print(f"  [WARN] Using first available split: '{first}'", flush=True)

    print(f"  Available splits : {list(raw.keys())}",  flush=True)
    print(f"  Full split size  : {len(dataset):,}",    flush=True)
    print(f"  Columns          : {dataset.column_names}", flush=True)

    if len(dataset) == 0:
        raise ValueError(f"Dataset split '{split}' is empty!")

    # Sample selection
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
        print(f"  Selected samples : {len(dataset):,}", flush=True)
    else:
        print(f"  Using all samples: {len(dataset):,}", flush=True)

    # Preview first sample
    s0 = dataset[0]
    print(f"\n  First sample preview:", flush=True)
    for k, v in s0.items():
        if k == "image":
            if hasattr(v, "size"):
                ratio = max(v.size) / max(min(v.size), 1)
                print(
                    f"    image : PIL {v.size} mode={v.mode} ratio={ratio:.1f}",
                    flush=True,
                )
            else:
                print(f"    image : {type(v)}", flush=True)
        else:
            print(f"    {k} : {str(v)[:120]}", flush=True)

    formatted = []
    skipped   = 0
    fixed     = 0
    bad_ratio = 0

    for i, sample in enumerate(dataset):
        try:
            # Parse conversations
            raw_conv = sample["conversations"]
            if isinstance(raw_conv, str):
                conversations = json.loads(raw_conv)
            elif isinstance(raw_conv, list):
                conversations = raw_conv
            else:
                raise ValueError(f"Unexpected conversations type: {type(raw_conv)}")

            # Get image
            img = sample["image"]
            if isinstance(img, str):
                img = PILImage.open(img)
            elif not isinstance(img, PILImage.Image):
                raise ValueError(f"Unexpected image type: {type(img)}")

            # Fix image
            original_size = img.size
            img = fix_image(img)
            if img.size != original_size:
                fixed += 1

            # Validate
            if not validate_image(img, i):
                bad_ratio += 1
                skipped   += 1
                continue

            # Extract question + answer
            question = ""
            answer   = ""
            for turn in conversations:
                role    = turn.get("role", turn.get("from", ""))
                content = turn.get("content", turn.get("value", ""))
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant"):
                    role = "assistant"
                if role == "user" and not question:
                    question = str(content).strip()
                elif role == "assistant" and not answer:
                    answer = str(content).strip()

            if not question:
                raise ValueError("No user turn found in conversations")
            if not answer:
                raise ValueError("No assistant turn found in conversations")

            formatted.append({
                "question": question,
                "context":  "",
                "answer":   answer,
                "file":     "",
                "image":    img,
            })

        except Exception as e:
            if skipped < 20:
                print(f"  [WARN] sample {i}: {type(e).__name__}: {e}", flush=True)
            skipped += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(dataset)} processed ...", flush=True)

    print(f"\n  ✓ {len(formatted):,} ready",                            flush=True)
    print(f"    skipped   : {skipped:,}",                               flush=True)
    print(f"    bad ratio : {bad_ratio:,} (skipped after fix failed)",  flush=True)
    print(f"    fixed     : {fixed:,} images (resize/pad)",             flush=True)

    if len(formatted) == 0:
        raise ValueError(
            "All samples were skipped! "
            f"Dataset columns: {dataset.column_names}"
        )

    return formatted


def domain_iterator_from_list(
    data: List[Dict],
    seed: int,
) -> Iterator[Dict]:
    """Shuffle and yield items from pre-loaded list."""
    rng = random.Random(seed)
    buf = list(data)
    rng.shuffle(buf)
    for item in buf:
        yield item


# ==========================
# PARSING
# ==========================
_REASONING_RE = re.compile(
    re.escape(REASONING_START) + r"\s*(.*?)\s*" + re.escape(REASONING_END),
    re.DOTALL,
)
_SOL_RE = re.compile(
    re.escape(SOLUTION_START) + r"\s*(.*?)\s*" + re.escape(SOLUTION_END),
    re.DOTALL,
)
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def extract_reasoning_block(text: str) -> str:
    m = _REASONING_RE.search(text or "")
    if not m:
        return ""
    rationale = m.group(1).strip()
    rationale = _SOL_RE.sub("", rationale).strip()
    return rationale


def extract_solution_block(text: str) -> Optional[str]:
    m = _SOL_RE.search(text or "")
    if not m:
        return None
    val = m.group(1).strip()
    val = _strip_tags(val)
    return val if val else None


def extract_single_float(text: str) -> Optional[str]:
    m = _FLOAT_RE.search(text or "")
    return m.group(0) if m else None


def parse_model_answer(raw_text: str, force_float: bool) -> str:
    raw_text = (raw_text or "").strip()
    sol = extract_solution_block(raw_text)
    if sol is not None:
        if force_float:
            f = extract_single_float(sol)
            return f if f is not None else sol
        return sol
    if force_float:
        f = extract_single_float(raw_text)
        return f if f is not None else _strip_tags(raw_text)
    return _strip_tags(raw_text)


# ==========================
# PROMPTS
# ==========================
def build_gen_prompt(ex: Dict, force_float: bool = False) -> str:
    q      = str(ex.get("question", "")).strip()
    c      = str(ex.get("context",  "")).strip()
    q_text = f"Context: {c}\n\nQuestion: {q}" if c else q

    if force_float:
        solution_rule = (
            f"and then your final answer between {SOLUTION_START} and "
            f"{SOLUTION_END}. Put a single float inside "
            f"{SOLUTION_START}{SOLUTION_END}."
        )
    else:
        solution_rule = (
            f"and then your final answer between {SOLUTION_START} and "
            f"{SOLUTION_END}. Put only the final answer inside "
            f"{SOLUTION_START}{SOLUTION_END}."
        )

    return (
        f"{q_text}\n"
        f"Also first provide your reasoning or working out on how you "
        f"would go about solving the question "
        f"between {REASONING_START} and {REASONING_END} "
        f"{solution_rule}"
    )


def build_cls_prompt(ex: Dict) -> str:
    q        = str(ex.get("question",    "")).strip()
    c        = str(ex.get("context",     "")).strip()
    gold     = str(ex.get("answer",      "")).strip()
    proposed = str(ex.get("generated_c", "")).strip()

    return (
        "You are a vision language model that checks answers.\n"
        "Task: Decide if the proposed answer is correct or incorrect.\n"
        "Rules:\n"
        "1. Look at the question, context, image, and the ground truth answer.\n"
        "2. Compare the proposed answer with the ground truth answer in meaning "
        "and detail.\n"
        "3. Reply with exactly one word: correct or incorrect.\n"
        "4. Do not add any extra words or punctuation.\n"
        f"Context: {c}\n"
        f"Question: {q}\n"
        f"Ground truth answer: {gold}\n"
        f"Proposed answer: {proposed}\n"
        "Reply with only one word:"
    )


# ==========================
# FORMATTERS
# ==========================
def format_for_gen(ex: Dict, cfg: AlgoConfig) -> Dict:
    user_text = build_gen_prompt(ex, force_float=cfg.force_float_solution)
    gold      = str(ex.get("answer", "")).strip()

    if cfg.force_float_solution:
        f    = extract_single_float(gold)
        gold = f if f is not None else gold

    rationale      = str(ex.get("generated_rationale", "")).strip()
    assistant_text = (
        f"{REASONING_START}{rationale}{REASONING_END}\n"
        f"{SOLUTION_START}{gold}{SOLUTION_END}"
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text",  "text":  user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
    }


def format_for_cls(ex: Dict) -> Dict:
    user_text = build_cls_prompt(ex)
    label     = str(ex.get("answer_correct", "")).strip()

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text",  "text":  user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]
    }


# ==========================
# MODEL MANAGER
# ==========================
class DualModelManager:
    def __init__(self, cfg: AlgoConfig):
        self.cfg = cfg

        # ── Clear memory before loading ──────────────────────────
        free_memory()
        print_gpu_memory("before model load")

        print(f"\nLoading Base Model: {cfg.base_model}", flush=True)
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            cfg.base_model,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            max_seq_length=cfg.max_seq_length,
            dtype=torch.bfloat16,
        )

        print_gpu_memory("after model load")

        self.gen_path = Path(cfg.out_dir) / "generator_adapter"
        self.cls_path = Path(cfg.out_dir) / "classifier_adapter"

        # ── Init adapters one at a time with memory cleanup between ──
        self._init_adapter(str(self.gen_path))
        free_memory()
        print_gpu_memory("after gen adapter init")

        self._init_adapter(str(self.cls_path))
        free_memory()
        print_gpu_memory("after cls adapter init")

    def _init_adapter(self, path: str):
        """
        Create a fresh LoRA adapter and save it to disk.
        If it already exists, skip.
        """
        if os.path.exists(path):
            print(f"  Adapter already exists at {path} — skipping init", flush=True)
            return

        print(f"  Initializing fresh adapter at {path}", flush=True)

        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers     = True,
            finetune_language_layers   = True,
            finetune_attention_modules = True,
            finetune_mlp_modules       = True,
            r                          = 16,
            lora_alpha                 = 16,
            lora_dropout               = 0,
            bias                       = "none",
            random_state               = 3407,
            use_rslora                 = False,
        )

        # Save adapter weights to disk
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  Saved initial adapter to {path}", flush=True)

        # ── Unload PEFT so we start clean for next adapter ───────
        try:
            self.model = self.model.merge_and_unload()
            print(f"  Unloaded adapter after saving", flush=True)
        except Exception as e1:
            print(f"  merge_and_unload failed ({e1}), trying manual delete", flush=True)
            try:
                if hasattr(self.model, "peft_config"):
                    for name in list(self.model.peft_config.keys()):
                        self.model.delete_adapter(name)
            except Exception as e2:
                print(f"  manual delete also failed: {e2}", flush=True)

    def _clean_active_adapters(self):
        """Remove any currently loaded adapters from the model."""
        try:
            self.model = self.model.unload()
            return
        except Exception:
            pass
        if hasattr(self.model, "peft_config"):
            for name in list(self.model.peft_config.keys()):
                try:
                    self.model.delete_adapter(name)
                except Exception:
                    pass

    def load_generator(self, inference: bool = False):
        print("\n  Swapping to GENERATOR adapter", flush=True)
        self._clean_active_adapters()
        free_memory()
        self.model.load_adapter(str(self.gen_path), adapter_name="generator")
        self.model.set_adapter("generator")
        if inference:
            FastVisionModel.for_inference(self.model)
        else:
            FastVisionModel.for_training(self.model)
        print_gpu_memory("generator loaded")

    def load_classifier(self, inference: bool = False):
        print("\n  Swapping to CLASSIFIER adapter", flush=True)
        self._clean_active_adapters()
        free_memory()
        self.model.load_adapter(str(self.cls_path), adapter_name="classifier")
        self.model.set_adapter("classifier")
        if inference:
            FastVisionModel.for_inference(self.model)
        else:
            FastVisionModel.for_training(self.model)
        print_gpu_memory("classifier loaded")


# ==========================
# MG GENERATE → DISK
# ==========================
def mg_generate(
    manager:  DualModelManager,
    batch:    List[Dict],
    out_path: Path,
):
    """
    Run the generator model over each example in batch.
    Write results as JSONL to out_path.
    """
    manager.load_generator(inference=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(tqdm(batch, desc="MG generate")):
            try:
                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": build_gen_prompt(
                                    ex,
                                    force_float=manager.cfg.force_float_solution,
                                ),
                            },
                        ],
                    }
                ]

                input_text = manager.tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                )

                inputs = manager.tokenizer(
                    ex["image"],
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to("cuda")

                with torch.inference_mode():
                    out = manager.model.generate(
                        **inputs,
                        max_new_tokens=manager.cfg.gen_max_new_tokens,
                        use_cache=True,
                    )

                prompt_len = inputs["input_ids"].shape[1]
                decoded    = manager.tokenizer.decode(
                    out[0][prompt_len:],
                    skip_special_tokens=True,
                ).strip()

                # Free input tensors immediately
                del inputs, out
                free_memory()

                # Extract rationale
                rationale = extract_reasoning_block(decoded)
                if not rationale:
                    before_sol = decoded.split(SOLUTION_START)[0].strip()
                    rationale  = (
                        _strip_tags(before_sol) if before_sol else _strip_tags(decoded)
                    )

                # Extract solution
                solution = parse_model_answer(
                    decoded,
                    force_float=manager.cfg.force_float_solution,
                )
                if not solution:
                    last_line = ""
                    for line in reversed(decoded.splitlines()):
                        stripped = _strip_tags(line).strip()
                        if stripped:
                            last_line = stripped
                            break
                    solution = last_line

                f.write(
                    json.dumps(
                        {
                            "question":            ex["question"],
                            "context":             ex["context"],
                            "answer":              ex["answer"],
                            "file":                ex["file"],
                            "generated_raw":       decoded,
                            "generated_rationale": rationale,
                            "generated_c":         solution,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            except torch.cuda.OutOfMemoryError:
                print(
                    f"  [OOM] sample {idx} in mg_generate — skipping",
                    flush=True,
                )
                free_memory()
                continue

            except Exception as e:
                print(
                    f"  [WARN] sample {idx} in mg_generate: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                continue


# ==========================
# MC VALIDATE → DISK
# ==========================
def mc_validate(
    manager:   DualModelManager,
    gen_jsonl: Path,
    all_data:  List[Dict],
    cls_jsonl: Path,
) -> tuple:
    """
    Validate generated answers using classifier model.
    Returns (gen_items, cls_items) for fine-tuning.
    """
    manager.load_classifier(inference=True)

    # Build image lookup by question text
    img_lookup: Dict[str, PILImage.Image] = {
        item["question"]: item["image"] for item in all_data
    }

    gen_items: List[Dict] = []
    cls_items: List[Dict] = []

    with (
        open(gen_jsonl, "r", encoding="utf-8") as fin,
        open(cls_jsonl, "w", encoding="utf-8") as fout,
    ):
        lines = fin.readlines()

        for idx, line in enumerate(tqdm(lines, desc="MC validate")):
            try:
                ex  = json.loads(line)
                img = img_lookup.get(ex["question"])

                if img is None:
                    print(
                        f"  [WARN] No image found for question: "
                        f"{ex['question'][:60]}",
                        flush=True,
                    )
                    continue

                ex["image"] = img

                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": build_cls_prompt(ex)},
                        ],
                    }
                ]

                input_text = manager.tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                )

                inputs = manager.tokenizer(
                    ex["image"],
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to("cuda")

                with torch.inference_mode():
                    out = manager.model.generate(
                        **inputs,
                        max_new_tokens=manager.cfg.cls_max_new_tokens,
                        use_cache=True,
                    )

                prompt_len = inputs["input_ids"].shape[1]
                verdict    = manager.tokenizer.decode(
                    out[0][prompt_len:],
                    skip_special_tokens=True,
                ).strip().lower()

                # Free tensors
                del inputs, out
                free_memory()

                is_correct = (
                    ("correct" in verdict) and ("incorrect" not in verdict)
                )

                # Write classification result (no image in JSON)
                out_ex = {k: v for k, v in ex.items() if k != "image"}
                out_ex["answer_correct"] = "correct" if is_correct else "incorrect"
                fout.write(json.dumps(out_ex, ensure_ascii=False) + "\n")

                # Collect training items
                if ex.get("generated_rationale", "").strip():
                    if is_correct:
                        gen_items.append({
                            "question":            ex["question"],
                            "context":             ex["context"],
                            "answer":              ex["generated_c"],
                            "generated_rationale": ex["generated_rationale"],
                            "image":               ex["image"],
                        })

                    cls_items.append({
                        "question":            ex["question"],
                        "context":             ex["context"],
                        "answer":              ex["generated_c"],
                        "generated_rationale": ex["generated_rationale"],
                        "answer_correct":      "correct" if is_correct else "incorrect",
                        "image":               ex["image"],
                    })

            except torch.cuda.OutOfMemoryError:
                print(
                    f"  [OOM] sample {idx} in mc_validate — skipping",
                    flush=True,
                )
                free_memory()
                continue

            except Exception as e:
                print(
                    f"  [WARN] sample {idx} in mc_validate: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                continue

    return gen_items, cls_items


# ==========================
# BUILD DATASETS
# ==========================
def build_gen_dataset(items: List[Dict], cfg: AlgoConfig) -> Dataset:
    return Dataset.from_list([format_for_gen(ex, cfg) for ex in items])


def build_cls_dataset(items: List[Dict]) -> Dataset:
    return Dataset.from_list([format_for_cls(ex) for ex in items])


# ==========================
# FINETUNE
# ==========================
def finetune(
    manager: DualModelManager,
    items:   List[Dict],
    mode:    str,
):
    """
    Fine-tune either the generator or classifier adapter
    on the given items list.
    """
    if not items:
        print(f"  [finetune] No items for {mode} — skipping.", flush=True)
        return

    print(
        f"\n  [finetune] Training {mode} on {len(items)} examples ...",
        flush=True,
    )
    print_gpu_memory(f"before {mode} finetune")

    if mode == "generator":
        manager.load_generator(inference=False)
        ds        = build_gen_dataset(items, manager.cfg)
        save_path = manager.gen_path
    else:
        manager.load_classifier(inference=False)
        ds        = build_cls_dataset(items)
        save_path = manager.cls_path

    trainer = SFTTrainer(
        model         = manager.model,
        tokenizer     = manager.tokenizer,
        data_collator = UnslothVisionDataCollator(
            manager.model,
            manager.tokenizer,
        ),
        train_dataset = ds,
        args          = SFTConfig(
            per_device_train_batch_size  = manager.cfg.batch_size,
            gradient_accumulation_steps  = manager.cfg.grad_accum,
            warmup_steps                 = manager.cfg.warmup_steps,
            num_train_epochs             = manager.cfg.epochs_per_iter,
            learning_rate                = manager.cfg.learning_rate,
            output_dir                   = os.path.join(
                                               manager.cfg.out_dir,
                                               "outputs_temp",
                                           ),
            optim                        = "adamw_8bit",
            seed                         = 3407,
            remove_unused_columns        = False,
            dataset_kwargs               = {"skip_prepare_dataset": True},
            max_length                   = manager.cfg.max_seq_length,
            logging_steps                = 10,
            save_steps                   = 200,
            report_to                    = "none",
            dataloader_pin_memory        = False,   # saves memory
            fp16                         = False,
            bf16                         = True,
        ),
    )

    trainer.train()

    print(
        f"  Saving updated {mode} adapter to {save_path}",
        flush=True,
    )
    manager.model.save_pretrained(str(save_path))
    manager.tokenizer.save_pretrained(str(save_path))

    # ── Clean up trainer ────────────────────────────────────────
    del trainer
    del ds
    free_memory()
    print_gpu_memory(f"after {mode} finetune")


# ==========================
# MAIN
# ==========================
def main():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--hf_dataset",  required=True,
                   help="HuggingFace dataset e.g. 'vietmed/sft_16k_mix'")
    p.add_argument("--hf_split",    default="train",
                   help="Dataset split")
    p.add_argument("--num_samples", type=int, default=None,
                   help="How many samples to use. None = all")

    # Training
    p.add_argument("--iterations_k",         type=int,   default=3)
    p.add_argument("--step_size_j",          type=int,   default=50)
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--base_model",
                   default="unsloth/Qwen3-VL-8B-Instruct")
    p.add_argument("--max_seq_length",       type=int,   default=1024)
    p.add_argument("--learning_rate",        type=float, default=2e-4)
    p.add_argument("--epochs_per_iter",      type=int,   default=1)
    p.add_argument("--batch_size",           type=int,   default=1)
    p.add_argument("--grad_accum",           type=int,   default=16)
    p.add_argument("--gen_max_new_tokens",   type=int,   default=512)
    p.add_argument("--cls_max_new_tokens",   type=int,   default=16)
    p.add_argument("--force_float_solution", action="store_true")
    p.add_argument("--out_dir",              default="outputs")

    args = p.parse_args()

    cfg = AlgoConfig(
        base_model           = args.base_model,
        iterations_k         = args.iterations_k,
        step_size_j          = args.step_size_j,
        max_seq_length       = args.max_seq_length,
        learning_rate        = args.learning_rate,
        epochs_per_iter      = args.epochs_per_iter,
        batch_size           = args.batch_size,
        grad_accum           = args.grad_accum,
        gen_max_new_tokens   = args.gen_max_new_tokens,
        cls_max_new_tokens   = args.cls_max_new_tokens,
        force_float_solution = args.force_float_solution,
        out_dir              = args.out_dir,
    )

    random.seed(args.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Print config ─────────────────────────────────────────────
    print(f"\n{'='*55}", flush=True)
    print(f"  model           : {cfg.base_model}",    flush=True)
    print(f"  dataset         : {args.hf_dataset}",   flush=True)
    print(f"  split           : {args.hf_split}",     flush=True)
    print(f"  num_samples     : {args.num_samples}",  flush=True)
    print(f"  iterations_k    : {cfg.iterations_k}",  flush=True)
    print(f"  step_size_j     : {cfg.step_size_j}",   flush=True)
    print(f"  max_seq_length  : {cfg.max_seq_length}", flush=True)
    print(f"  gen_max_tokens  : {cfg.gen_max_new_tokens}", flush=True)
    print(f"  max_img_size    : {MAX_IMAGE_SIZE}",    flush=True)
    print(f"  max_ratio       : {MAX_RATIO}",         flush=True)
    print(f"{'='*55}\n", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Load model FIRST to claim GPU memory before images
    # ══════════════════════════════════════════════════════════════
    print("STEP 1: Loading model ...", flush=True)
    manager = DualModelManager(cfg)
    free_memory()

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Load dataset AFTER model is in GPU
    # ══════════════════════════════════════════════════════════════
    print("\nSTEP 2: Loading dataset ...", flush=True)
    all_data = load_and_format_domain(
        hf_dataset_name = args.hf_dataset,
        split           = args.hf_split,
        num_samples     = args.num_samples,
        seed            = args.seed,
    )

    print(f"\n  Total usable samples: {len(all_data):,}", flush=True)
    assert len(all_data) > 0, "all_data is empty!"

    # ── Build iterator ───────────────────────────────────────────
    stream = domain_iterator_from_list(all_data, seed=args.seed)

    # ══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════
    for it in range(cfg.iterations_k):
        print(f"\n{'='*55}", flush=True)
        print(f"  ITERATION {it+1}/{cfg.iterations_k}", flush=True)
        print(f"{'='*55}", flush=True)

        # ── Pull next batch ──────────────────────────────────────
        batch = []
        for _ in range(cfg.step_size_j):
            try:
                batch.append(next(stream))
            except StopIteration:
                print("  Stream exhausted — reshuffling ...", flush=True)
                stream = domain_iterator_from_list(
                    all_data,
                    seed=args.seed + it,
                )
                try:
                    batch.append(next(stream))
                except StopIteration:
                    break

        if not batch:
            print("  No more data — stopping.", flush=True)
            break

        print(f"  Batch size: {len(batch)}", flush=True)

        gen_p = Path(cfg.out_dir) / f"iter_{it}_gen.jsonl"
        cls_p = Path(cfg.out_dir) / f"iter_{it}_cls.jsonl"

        # ── Generate ─────────────────────────────────────────────
        print(f"\n  [iter {it+1}] Running MG generate ...", flush=True)
        mg_generate(manager, batch, gen_p)
        free_memory()

        # ── Validate ─────────────────────────────────────────────
        print(f"\n  [iter {it+1}] Running MC validate ...", flush=True)
        gen_items, cls_items = mc_validate(
            manager   = manager,
            gen_jsonl = gen_p,
            all_data  = all_data,
            cls_jsonl = cls_p,
        )
        free_memory()

        print(f"\n  Generator training samples : {len(gen_items)}", flush=True)
        print(f"  Classifier training samples: {len(cls_items)}", flush=True)

        # ── Fine-tune generator ──────────────────────────────────
        finetune(manager, gen_items, "generator")

        # ── Fine-tune classifier ─────────────────────────────────
        finetune(manager, cls_items, "classifier")

        print(f"\n  [iter {it+1}] Complete ✓", flush=True)
        print_gpu_memory(f"end of iter {it+1}")

    print("\n" + "="*55, flush=True)
    print("  DONE", flush=True)
    print("="*55, flush=True)


if __name__ == "__main__":
    main()
