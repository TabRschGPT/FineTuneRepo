#!/usr/bin/env python
# coding: utf-8
#
# train_mix_stream_no_save.py
#
# What this script does
# - Uses ALL domain JSONL samples (streams from disk)
# - Samples FineVision subsets in streaming mode
# - Keeps the mixture ratio via: domain_ratio = D / (D + G)  =>  G = D*(1-domain_ratio)/domain_ratio
# - Does NOT save FineVision images to disk
# - Does NOT cache images in memory (each sample loads and yields a PIL image)
#
# Requirements
# - pip install datasets pillow torch trl unsloth huggingface_hub
# - huggingface-cli login (if pulling gated or pushing)
#
# Run
# python train_mix_stream_no_save.py \
#   --domain_data /path/to/domain.jsonl \
#   --best_params_json best_params.json \
#   --output_dir outputs/qwen3vl_stream_full \
#   --epochs 2

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import io
import json
import math
import random
import uuid
import gc
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from PIL import Image as PILImage

from datasets import load_dataset, IterableDataset

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


# ==========================
# Global config
# ==========================
MODEL_MAX_SEQ_LENGTH = 1024
MAX_IMAGE_SIDE = 768

FINEVISION_DATASET_ID = "HuggingFaceM4/FineVision"

DEFAULT_FINEVISION_SUBSETS = [
    "CoSyn_400k_table",
    "docvqa",
    "pdfvqa",
    "textvqa",
    "chartqa",
    "infographic_vqa",
    "arxivqa",
    "ai2d_merged",
    "Unichart",
    "LLaVA_Instruct_150K",
]

PROMPT_STYLE = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully and provide a short explanation inside <think>...</think>.

### Instruction:
You are an expert in chemical patents and scientific document understanding.
You must answer questions about chemical patent tables, figures, and document images.
Use the image and the provided context. If the answer is a number, return the number.
Return:
1) Final Answer on the first line.
2) A short explanation wrapped in <think>...</think> on the next line.

### Question:
{question}

### Context:
{context}

### Response:
"""


# ==========================
# Image helpers (NO SAVE)
# ==========================
def resize_if_needed(img: PILImage.Image) -> PILImage.Image:
    w, h = img.size
    m = max(w, h)
    if m <= MAX_IMAGE_SIDE:
        return img
    s = MAX_IMAGE_SIDE / float(m)
    nw = max(1, int(w * s))
    nh = max(1, int(h * s))
    return img.resize((nw, nh))


def to_pil_rgb(img_obj) -> PILImage.Image:
    """
    Convert various HF image formats into PIL, without saving.
    """
    if isinstance(img_obj, PILImage.Image):
        return resize_if_needed(img_obj.convert("RGB"))

    if isinstance(img_obj, str):
        return resize_if_needed(PILImage.open(img_obj).convert("RGB"))

    if isinstance(img_obj, dict) and img_obj.get("bytes"):
        return resize_if_needed(PILImage.open(io.BytesIO(img_obj["bytes"])).convert("RGB"))

    if isinstance(img_obj, dict) and img_obj.get("path"):
        return resize_if_needed(PILImage.open(img_obj["path"]).convert("RGB"))

    if hasattr(img_obj, "to_pil"):
        return resize_if_needed(img_obj.to_pil().convert("RGB"))

    raise ValueError(f"Unsupported image format: {type(img_obj)}")


# ==========================
# Domain streaming
# ==========================
def resolve_domain_image_path(domain_jsonl_path: str, img_path: str) -> str:
    base_dir = str(Path(domain_jsonl_path).parent.resolve())
    if not os.path.isabs(img_path):
        return os.path.join(base_dir, img_path)
    return img_path


def count_domain_samples(domain_jsonl_path: str, include_logo: bool) -> int:
    """
    Counts valid domain samples (to compute G). Reads the jsonl once.
    """
    n = 0
    with open(domain_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if (not include_logo) and (x.get("artefact_type") == "logo"):
                continue
            n += 1
    return n


def domain_iterator(
    domain_jsonl_path: str,
    include_logo: bool,
    seed: int,
) -> Iterator[dict]:
    """
    Streams domain data from JSONL. No shuffling across whole file.
    For a "good enough" shuffle, we do small-buffer shuffle.
    """
    rng = random.Random(seed)
    buf: List[dict] = []
    BUF_SIZE = 256  # small shuffle buffer

    with open(domain_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if (not include_logo) and (x.get("artefact_type") == "logo"):
                continue

            img_path = resolve_domain_image_path(domain_jsonl_path, x["file"])
            try:
                img = to_pil_rgb(img_path)
            except Exception:
                continue

            rec = {
                "source": "domain",
                "question": str(x.get("question", "")),
                "context": str(x.get("context", "")),
                "answer": str(x.get("answer", "")),
                "rationale": str(x.get("rationale", "")).strip(),
                "image": img,  # PIL
            }

            buf.append(rec)
            if len(buf) >= BUF_SIZE:
                rng.shuffle(buf)
                while buf:
                    yield buf.pop()

    rng.shuffle(buf)
    while buf:
        yield buf.pop()


# ==========================
# FineVision streaming (NO SAVE)
# ==========================
def adapt_fv(ex) -> Tuple[str, str, object]:
    q = ex.get("question") or ex.get("query") or "Describe the image."
    a = ex.get("answer")

    if a is None:
        answers = ex.get("answers")
        if isinstance(answers, list) and len(answers) > 0:
            a = answers[0]
    if a is None:
        a = ""

    img = ex.get("image") or ex.get("img")
    if img is None:
        imgs = ex.get("images")
        if isinstance(imgs, list) and len(imgs) > 0:
            img = imgs[0]
    if img is None:
        raise KeyError("No image field in FineVision sample")

    return str(q), str(a), img


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(weights.values()))
    if s <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: float(v) / s for k, v in weights.items()}


def allocate_quotas(total_n: int, weights: Dict[str, float]) -> Dict[str, int]:
    """
    Largest-remainder allocation so quotas sum to total_n.
    """
    if total_n <= 0:
        return {k: 0 for k in weights}

    w = normalize_weights(weights)
    raw = {k: total_n * v for k, v in w.items()}
    base = {k: int(v) for k, v in raw.items()}
    rem = total_n - sum(base.values())

    frac = sorted(((k, raw[k] - base[k]) for k in raw), key=lambda x: x[1], reverse=True)
    for i in range(rem):
        base[frac[i][0]] += 1

    return base


def finevision_subset_iterator(subset: str, quota: int, seed: int) -> Iterator[dict]:
    """
    Streams a single FineVision subset until quota items are produced.
    """
    if quota <= 0:
        return
        yield  # to make it a generator

    ds = load_dataset(
        FINEVISION_DATASET_ID,
        subset,
        split="train",
        streaming=True,
    )

    got = 0
    for ex in ds:
        if got >= quota:
            break
        try:
            q, a, img_obj = adapt_fv(ex)
            img = to_pil_rgb(img_obj)
            yield {
                "source": "general",
                "question": q,
                "context": "",
                "answer": a,
                "rationale": "",
                "image": img,  # PIL
            }
            got += 1
        except Exception:
            continue


def finevision_iterator(
    total_n: int,
    weights: Dict[str, float],
    seed: int,
) -> Iterator[dict]:
    """
    Streams across subsets by quota, and yields PIL images without saving.
    """
    rng = random.Random(seed)
    weights = normalize_weights(weights)
    quotas = allocate_quotas(total_n, weights)

    # To avoid always starting from same subset, shuffle subset order
    subsets = list(quotas.keys())
    rng.shuffle(subsets)

    for subset in subsets:
        quota = int(quotas.get(subset, 0))
        if quota <= 0:
            continue
        for item in finevision_subset_iterator(subset, quota, seed=seed):
            yield item


# ==========================
# Build chat messages (PIL images)
# ==========================
def build_messages(ex: dict) -> dict:
    user_text = PROMPT_STYLE.format(
        question=str(ex.get("question", "")).strip(),
        context=str(ex.get("context", "")).strip(),
    )

    ans = str(ex.get("answer", "")).strip()
    rat = str(ex.get("rationale", "")).strip()
    if rat:
        assistant_text = f"{ans}\n<think>{rat}</think>"
    else:
        assistant_text = f"{ans}\n<think></think>"

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": ex["image"]},  # PIL object
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
    }


# ==========================
# Mixture generator with exact counts
# ==========================
def mixture_generator(
    domain_jsonl_path: str,
    include_logo: bool,
    domain_count: int,
    general_count: int,
    general_weights: Dict[str, float],
    seed: int,
) -> Iterator[dict]:
    """
    Produces exactly domain_count + general_count items.
    Interleaves samples with probability proportional to remaining counts.
    """
    rng = random.Random(seed)

    dom_it = domain_iterator(domain_jsonl_path, include_logo=include_logo, seed=seed)
    gen_it = finevision_iterator(total_n=general_count, weights=general_weights, seed=seed)

    remD = int(domain_count)
    remG = int(general_count)

    while remD > 0 or remG > 0:
        if remD == 0:
            x = next(gen_it)
            remG -= 1
            yield build_messages(x)
            continue

        if remG == 0:
            x = next(dom_it)
            remD -= 1
            yield build_messages(x)
            continue

        # choose domain with probability remD/(remD+remG)
        if rng.random() < (remD / float(remD + remG)):
            x = next(dom_it)
            remD -= 1
            yield build_messages(x)
        else:
            x = next(gen_it)
            remG -= 1
            yield build_messages(x)


# ==========================
# Best params loader
# ==========================
def load_best_params(best_params_json: str) -> Tuple[float, Dict[str, float]]:
    best = json.load(open(best_params_json, "r", encoding="utf-8"))
    if "domain_ratio" not in best:
        raise ValueError("best_params_json must contain key: domain_ratio")

    domain_ratio = float(best["domain_ratio"])

    weights = {}
    for k, v in best.items():
        if k.startswith("w_"):
            weights[k[len("w_"):]] = float(v)

    # ensure all known subsets exist, missing defaults to 0.0
    for s in DEFAULT_FINEVISION_SUBSETS:
        weights.setdefault(s, 0.0)

    return domain_ratio, weights


# ==========================
# Main
# ==========================
def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--domain_data", required=True)
    p.add_argument("--include_logo", action="store_true")

    # best params
    p.add_argument("--best_params_json", required=True)

    # training
    p.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)

    # output
    p.add_argument("--output_dir", required=True)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", default="wandb")  # set "none" to disable

    args = p.parse_args()
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load best params
    domain_ratio, weights = load_best_params(args.best_params_json)

    if not (0.0 < domain_ratio < 1.0):
        raise ValueError("domain_ratio must be between 0 and 1")

    # Count domain to compute general
    D = count_domain_samples(args.domain_data, include_logo=args.include_logo)
    if D <= 0:
        raise ValueError("Domain dataset is empty after filtering.")

    # G = D*(1-r)/r
    G = int(round(D * (1.0 - domain_ratio) / domain_ratio))
    total_samples = D + G

    # Compute training steps for IterableDataset
    # each step consumes (bs * grad_accum) samples effectively per optimizer update
    samples_per_update = max(1, int(args.bs) * int(args.grad_accum))
    updates_per_epoch = int(math.ceil(total_samples / float(samples_per_update)))
    max_steps = int(updates_per_epoch * int(args.epochs))

    print(f"[mix] domain_count={D}")
    print(f"[mix] general_count={G}")
    print(f"[mix] domain_ratio={domain_ratio}")
    print(f"[train] samples_per_update={samples_per_update}")
    print(f"[train] updates_per_epoch≈{updates_per_epoch}")
    print(f"[train] max_steps={max_steps}")

    # Save resolved config
    resolved = {
        "domain_ratio": domain_ratio,
        "domain_count": D,
        "general_count": G,
        "total_samples": total_samples,
        "epochs": args.epochs,
        "bs": args.bs,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_steps": max_steps,
        "weights": weights,
        "include_logo": args.include_logo,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "resolved_stream_mix_config.json"), "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2)

    # Build streaming dataset
    gen_fn = lambda: mixture_generator(
        domain_jsonl_path=args.domain_data,
        include_logo=args.include_logo,
        domain_count=D,
        general_count=G,
        general_weights=weights,
        seed=args.seed,
    )
    train_ds = IterableDataset.from_generator(gen_fn)

    # Load model
    model, tok = FastVisionModel.from_pretrained(
        args.model,
        max_seq_length=MODEL_MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
        use_gradient_checkpointing="unsloth",
    )

    # token padding safety
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # LoRA (defaults are fine, tune if needed)
    model = FastVisionModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    report_to = args.report_to.strip().lower()
    if report_to in ["none", "false", "0", "off", "null"]:
        report_to = "none"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        data_collator=UnslothVisionDataCollator(model, tok),
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_length=MODEL_MAX_SEQ_LENGTH,
            logging_steps=10,
            save_steps=200,
            max_steps=max_steps,  # IMPORTANT for streaming datasets
            report_to=report_to,
        ),
    )

    trainer.train()

    # Save final checkpoint
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # Cleanup
    del trainer, model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Done. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
