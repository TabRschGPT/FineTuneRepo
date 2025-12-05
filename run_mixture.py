#!/usr/bin/env python
# coding: utf-8

"""
Mixed-domain VLM fine-tune script for Unsloth + Qwen3-VL-8B.

Two-stage mixture with parameter grid:

Stage 1:
    - User sets max_domain_samples.
    - Script samples domain once.
    - For each domain ratio p in DOMAIN_RATIO_GRID:
        * Compute general size G = D * (1 - p) / p.
        * Optionally clamp by general_samples as a cap.

Stage 2:
    - For each subset weight config in SUBSET_WEIGHT_GRID:
        * Allocate G across FineVision subsets by subset weights.
        * Randomly sample from each subset using streaming.

For each (p, subset_weights) pair:
    - Build mixed dataset via probabilistic interleaving.
    - Train, run inference, run LLM judge.
    - Track best configuration by average judge score.
"""

import argparse
import io
import json
import os
import random
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from datasets import load_dataset, Image as HFImage
from tqdm import tqdm

import torch
from PIL import Image as PILImage

from dotenv import load_dotenv
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Load env once
load_dotenv()

# Optional judge client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ------------------------------------------------------------
# Parameter grids
# ------------------------------------------------------------

RECOMMENDED_FINEVISION_SUBSETS = [
    "CoSyn_400k_table",
    "docvqa",
    "pdfvqa",
    "textvqa",
    "chartqa",
    "infographic_vqa",
]

# Stage 1: domain ratios p = domain / (domain + general)
DOMAIN_RATIO_GRID: List[float] = [
    0.70,
    0.60,
    0.50,
]

# Stage 2: different subset weighting strategies
SUBSET_WEIGHT_GRID: List[Dict[str, float]] = [
    # Strong table heavy
    {
        "CoSyn_400k_table": 0.50,
        "docvqa": 0.20,
        "pdfvqa": 0.15,
        "textvqa": 0.10,
        "chartqa": 0.03,
        "infographic_vqa": 0.02,
    },
    # Balanced doc + table
    {
        "CoSyn_400k_table": 0.40,
        "docvqa": 0.25,
        "pdfvqa": 0.15,
        "textvqa": 0.10,
        "chartqa": 0.05,
        "infographic_vqa": 0.05,
    },
    # More general heavy
    {
        "CoSyn_400k_table": 0.30,
        "docvqa": 0.20,
        "pdfvqa": 0.20,
        "textvqa": 0.20,
        "chartqa": 0.05,
        "infographic_vqa": 0.05,
    },
]


# ============================================================
# 1. Helpers for images on disk
# ============================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_stream_image_to_disk(img_obj, tmp_dir: str) -> str:
    """
    Save a streamed image to disk as a real PNG file.
    Output: tmp_dir/<uuid>.png

    Supported inputs:
    - datasets.Image (HFImage)
    - PIL.Image.Image
    - dict with "bytes" or "path"
    - string path to a file
    """

    if img_obj is None:
        raise ValueError("FineVision example has no image field. Check adapt_finevision_example().")

    ensure_dir(tmp_dir)

    # Case 0: datasets.Image wrapper
    if isinstance(img_obj, HFImage):
        img_obj = img_obj.to_pil()

    # Case 1: string path
    if isinstance(img_obj, str):
        try:
            pil = PILImage.open(img_obj).convert("RGB")
            fname = f"{uuid.uuid4().hex}.png"
            out_path = os.path.join(tmp_dir, fname)
            pil.save(out_path, format="PNG")
            return out_path
        except Exception as e:
            raise ValueError(f"Image path exists but cannot be opened: {img_obj}. Error: {e}")

    # Case 2: already PIL image
    if isinstance(img_obj, PILImage.Image):
        fname = f"{uuid.uuid4().hex}.png"
        out_path = os.path.join(tmp_dir, fname)
        img_obj.convert("RGB").save(out_path, format="PNG")
        return out_path

    # Case 3: dict with bytes or path
    if isinstance(img_obj, dict):

        # Raw bytes
        if img_obj.get("bytes") is not None:
            raw = img_obj["bytes"]
            fname = f"{uuid.uuid4().hex}.png"
            out_path = os.path.join(tmp_dir, fname)
            try:
                pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
                pil.save(out_path, format="PNG")
                return out_path
            except Exception as e:
                raise ValueError(
                    f"Image bytes exist but cannot be decoded into a valid image. Error: {e}"
                )

        # Embedded path
        if img_obj.get("path") is not None:
            embedded = img_obj["path"]
            try:
                pil = PILImage.open(embedded).convert("RGB")
                fname = f"{uuid.uuid4().hex}.png"
                out_path = os.path.join(tmp_dir, fname)
                pil.save(out_path, format="PNG")
                return out_path
            except Exception as e:
                raise ValueError(f"Image path inside dict cannot be opened: {embedded}. Error: {e}")

    raise TypeError(
        f"Unsupported image type from FineVision: {type(img_obj)} "
        f"Preview: {repr(img_obj)[:200]}"
    )


# ============================================================
# 2. Domain dataset: JSONL loading and sampling
# ============================================================

def load_domain_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def sample_domain_data(
    jsonl_path: str,
    max_domain_samples: int,
    exclude_logo: bool = True,
):
    """
    Load domain JSONL, filter artefact_type == 'logo' if requested,
    then random sample up to max_domain_samples.
    """
    domain = load_domain_jsonl(jsonl_path)

    if exclude_logo:
        domain = [ex for ex in domain if ex.get("artefact_type") != "logo"]

    if not domain:
        raise ValueError("Domain dataset is empty after logo filter.")

    random.shuffle(domain)
    if max_domain_samples is not None and max_domain_samples > 0:
        domain = domain[:max_domain_samples]

    normalized = []
    for ex in domain:
        normalized.append(
            {
                "source": "domain",
                "type": ex.get("type", "Lookup"),
                "question": ex["question"],
                "context": ex.get("context", ""),
                "answer": ex["answer"],
                "rationale": ex.get("rationale", ""),
                "qid": ex.get("qid"),
                "pid": ex.get("pid"),
                "image_path": ex.get("file"),
                "artefact_type": ex.get("artefact_type", ""),
                "annotations": ex.get("annotations", ""),
            }
        )

    return normalized


# ============================================================
# 3. FineVision general data: subset weighted sampling
# ============================================================

def adapt_finevision_example(ex) -> dict:
    """
    Map one FineVision example to the common schema.
    """

    question = ex.get("question") or ex.get("query") or ""
    context = ex.get("context") or ""
    answer = ex.get("answer")
    if answer is None and isinstance(ex.get("answers"), list) and ex["answers"]:
        answer = ex["answers"][0]

    if not question or answer is None:
        texts = ex.get("texts")
        if isinstance(texts, list) and len(texts) >= 2:
            if not question:
                question = texts[0]
            if answer is None:
                answer = texts[1]
        elif isinstance(texts, list) and len(texts) == 1:
            if not question:
                question = "Describe the image in detail."
            if answer is None:
                answer = texts[0]

    if not question:
        raise KeyError(
            f"No usable question or texts in FineVision example. Keys: {list(ex.keys())}"
        )
    if answer is None:
        answer = ""

    img_obj = None
    if "image" in ex:
        img_obj = ex["image"]
    elif "image_path" in ex:
        img_obj = ex["image_path"]
    elif "img" in ex:
        img_obj = ex["img"]
    elif "images" in ex:
        imgs = ex["images"]
        if isinstance(imgs, list) and len(imgs) > 0:
            img_obj = imgs[0]

    if img_obj is None:
        raise KeyError(
            f"No image-like field found in FineVision example. "
            f"Keys: {list(ex.keys())}"
        )

    return {
        "source": "finevision",
        "type": "GeneralVQA",
        "question": question,
        "context": context,
        "answer": answer,
        "rationale": "",
        "qid": ex.get("qid"),
        "pid": ex.get("pid"),
        "image_obj": img_obj,
    }


def sample_finevision_general(
    tmp_image_dir: str,
    total_general_samples: int,
    subset_weights: Dict[str, float],
    subsets: Optional[List[str]] = None,
) -> List[dict]:
    """
    Stage 2 general sampling with subset proportions.

    - Use subset_weights to decide how many samples per subset.
    - Sample from each subset via HF streaming.
    """

    if total_general_samples <= 0:
        return []

    if subsets is None:
        subsets = RECOMMENDED_FINEVISION_SUBSETS

    # Active subsets = intersection of requested subsets and weight keys
    active_subsets = [s for s in subsets if s in subset_weights]
    if not active_subsets:
        raise ValueError("No active subsets found in subset_weights.")

    # Normalise weights
    weight_sum = sum(subset_weights[s] for s in active_subsets)
    if weight_sum <= 0:
        raise ValueError("Sum of subset_weights must be positive.")

    norm_weights = {s: subset_weights[s] / weight_sum for s in active_subsets}

    # Compute exact quotas and floor
    raw_targets: List[Tuple[str, float]] = []
    for s in active_subsets:
        exact = total_general_samples * norm_weights[s]
        raw_targets.append((s, exact))

    per_subset_targets: Dict[str, int] = {}
    total_floor = 0
    frac_parts: List[Tuple[str, float]] = []
    for s, exact in raw_targets:
        base = int(exact)
        per_subset_targets[s] = base
        total_floor += base
        frac_parts.append((s, exact - base))

    remainder = total_general_samples - total_floor
    frac_parts.sort(key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        s, _ = frac_parts[i]
        per_subset_targets[s] += 1

    all_samples: List[dict] = []

    for subset_name in active_subsets:
        subset_quota = per_subset_targets[subset_name]
        if subset_quota <= 0:
            continue
        if len(all_samples) >= total_general_samples:
            break

        print(f"Streaming FineVision subset: {subset_name} (quota: {subset_quota})")
        ds_stream = load_dataset(
            "HuggingFaceM4/FineVision",
            subset_name,
            split="train",
            streaming=True,
        )

        count = 0
        for ex in ds_stream:
            if count >= subset_quota or len(all_samples) >= total_general_samples:
                break

            try:
                adapted = adapt_finevision_example(ex)
            except KeyError:
                continue

            img_obj = adapted.pop("image_obj", None)
            img_path = save_stream_image_to_disk(img_obj, tmp_image_dir)

            adapted["image_path"] = img_path
            all_samples.append(adapted)
            count += 1

        print(f"Collected {count} samples from {subset_name}")

    if not all_samples:
        raise ValueError(
            "No general FineVision samples collected. Check subset names or schema."
        )

    random.shuffle(all_samples)
    if len(all_samples) > total_general_samples:
        all_samples = all_samples[:total_general_samples]

    return all_samples


# ============================================================
# 4. Mix domain + general and create train / val
# ============================================================

def build_messages_from_example(ex: dict) -> dict:
    """
    Convert normalized example to Unsloth chat messages format.
    """

    instruction = (
        "You are a chemistry and scientific vision expert. "
        "You look at tables, figures and other images and "
        "answer questions based on both the image and the text context."
    )

    user_text = (
        f"{instruction}\n\n"
        f"Question: {ex['question']}\n"
        f"Context: {ex.get('context', '')}\n"
    )

    assistant_text = (
        f"<think>{ex.get('rationale', '')}</think> "
        f"Answer: {ex['answer']}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": ex["image_path"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        },
    ]

    return {"messages": messages}


def split_train_val(examples, val_ratio: float, seed: int = 42):
    random.Random(seed).shuffle(examples)
    n = len(examples)
    n_val = int(n * val_ratio)
    val = examples[:n_val]
    train = examples[n_val:]
    return train, val


def interleave_domain_and_general(domain, general, domain_prob: Optional[float] = None, seed: int = 42):
    """
    For each position, choose domain with probability domain_prob,
    otherwise choose general.
    """

    rng = random.Random(seed)

    domain = list(domain)
    general = list(general)
    rng.shuffle(domain)
    rng.shuffle(general)

    total = len(domain) + len(general)
    if total == 0:
        return []

    if domain_prob is None:
        domain_prob = len(domain) / float(total)

    i_dom = 0
    i_gen = 0
    mixed = []

    while i_dom < len(domain) and i_gen < len(general):
        if rng.random() < domain_prob:
            mixed.append(domain[i_dom])
            i_dom += 1
        else:
            mixed.append(general[i_gen])
            i_gen += 1

    if i_dom < len(domain):
        mixed.extend(domain[i_dom:])
    if i_gen < len(general):
        mixed.extend(general[i_gen:])

    return mixed


# ============================================================
# 5. Model load and training
# ============================================================

def load_model(base="unsloth/Qwen3-VL-8B-Instruct"):
    model, tokenizer = FastVisionModel.from_pretrained(
        base,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
    )
    return model, tokenizer


def train(model, tokenizer, train_ds, val_ds, output, epochs):
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            learning_rate=2e-4,
            logging_steps=1,
            num_train_epochs=epochs,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output,
            report_to="wandb",
            logging_dir=os.path.join(output, "logs"),
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    print(f"Saving model to: {output}")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)


# ============================================================
# 6. Inference on validation set
# ============================================================

def build_inference_messages(ex: dict):
    instruction = (
        "You are a chemistry and scientific vision expert. "
        "You look at tables, figures and other images and "
        "answer questions based on both the image and the text context."
    )

    user_text = (
        f"{instruction}\n\n"
        f"Question: {ex['question']}\n"
        f"Context: {ex.get('context', '')}\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": ex["image_path"]},
            ],
        }
    ]
    return messages


def run_inference(model, tokenizer, val_raw, output_path, max_new_tokens=128):
    FastVisionModel.for_inference(model)
    model.eval()

    results = []
    for ex in tqdm(val_raw, desc="Inference"):
        messages = build_inference_messages(ex)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
            )

        gen_ids = generated[0][inputs.shape[-1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        results.append(
            {
                "source": ex.get("source"),
                "qid": ex.get("qid"),
                "pid": ex.get("pid"),
                "question": ex["question"],
                "context": ex.get("context", ""),
                "gold_answer": ex["answer"],
                "pred_answer": text,
                "image_path": ex["image_path"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved inference results to {output_path}")
    return results


# ============================================================
# 7. LLM judge via Braintrust (gpt-5)
# ============================================================

def judge_with_braintrust(results, cocktail_path, sample_limit=None):
    if OpenAI is None:
        print("openai library not installed. Skip judge.")
        return [], None

    base_url = os.environ.get("BRAINTRUST_BASE_URL")
    api_key = os.environ.get("BRAINTRUST_API_KEY")

    if not base_url or not api_key:
        print("Braintrust env vars not set. Skip judge.")
        return [], None

    client = OpenAI(base_url=base_url, api_key=api_key)

    judged = []
    iterable = results
    if sample_limit is not None and sample_limit > 0:
        iterable = results[:sample_limit]

    for item in tqdm(iterable, desc="LLM judge"):
        prompt = (
            "You are a strict but fair judge for visual question answering.\n"
            "You will see a question, text context, the model answer, "
            "and a file path pointing to the image.\n"
            "Assume you can view the image from the given path.\n"
            "Score how correct the model answer is for the question and image, from 0 to 1.\n"
            "Return only JSON with keys: score (0 or 1), explanation.\n\n"
            f"Question: {item['question']}\n"
            f"Context: {item.get('context','')}\n"
            f"Model answer: {item['pred_answer']}\n"
            f"Image file path: {item['image_path']}\n"
        )

        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        content = resp.choices[0].message.content
        try:
            judge_json = json.loads(content)
        except Exception:
            judge_json = {"score": None, "explanation": content}

        item_with_judge = dict(item)
        item_with_judge["judge_score"] = judge_json.get("score")
        item_with_judge["judge_explanation"] = judge_json.get("explanation")
        judged.append(item_with_judge)

    with open(cocktail_path, "w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    scores = [r["judge_score"] for r in judged if isinstance(r["judge_score"], (int, float))]
    avg_score = sum(scores) / len(scores) if scores else None

    print(f"Saved cocktail file with judge scores to {cocktail_path}")
    if avg_score is not None:
        print(f"Average judge score: {avg_score:.3f}")

    return judged, avg_score


# ============================================================
# 8. Recommend highest score and save JSON
# ============================================================

def recommend_highest_score(judged_results, output_path: str):
    valid = [r for r in judged_results if isinstance(r.get("judge_score"), (int, float))]

    if not valid:
        print("No valid judge scores found. Skip recommendation.")
        return None

    best = max(valid, key=lambda x: x["judge_score"])

    print("\n=== Recommended Highest-Score Sample ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"Saved recommended sample to {output_path}")
    return best


# ============================================================
# 9. Automatic 2D grid search over (domain_ratio, subset_weights)
# ============================================================

def run_grid_search(args):
    """
    For each domain_ratio in DOMAIN_RATIO_GRID
    and each subset_weights config in SUBSET_WEIGHT_GRID:

        Stage 1:
            - Compute G from D and domain_ratio.
            - Clamp by general_samples if > 0.

        Stage 2:
            - Sample G general examples from FineVision
              according to subset_weights.

        Then:
            - Interleave domain and general using domain_ratio.
            - Train, infer, judge.
            - Track best configuration by judge score.
    """

    # Sample domain once
    domain_examples = sample_domain_data(
        jsonl_path=args.domain_data,
        max_domain_samples=args.max_domain_samples,
        exclude_logo=True,
    )
    D = len(domain_examples)
    print(f"[grid] Domain examples after filter and sample: {D}")

    if D == 0:
        raise ValueError("No domain examples found after filtering.")

    if not args.run_judge:
        print("Grid search requires --run_judge to compare models.")
        return

    best_score = None
    best_config = None
    best_output = None

    grid_results = []

    for domain_prob in DOMAIN_RATIO_GRID:
        # Stage 1: compute G from ratio
        G_raw = D * (1.0 - domain_prob) / domain_prob
        G_target = int(round(G_raw))

        if args.general_samples is not None and args.general_samples > 0:
            G_target = min(G_target, args.general_samples)

        print(f"\n=== Domain ratio {domain_prob:.2f}, target general G = {G_target} ===")

        for subset_idx, subset_weights in enumerate(SUBSET_WEIGHT_GRID):
            print(f"\n--- Subset weight config {subset_idx} ---")

            run_tag = f"dp{int(domain_prob * 100)}_sw{subset_idx}"
            run_output = f"{args.output}_{run_tag}"
            os.makedirs(run_output, exist_ok=True)

            print(f"[grid] Sampling general data for {run_tag} ...")
            general_examples = sample_finevision_general(
                tmp_image_dir=args.tmp_image_dir,
                total_general_samples=G_target,
                subset_weights=subset_weights,
                subsets=RECOMMENDED_FINEVISION_SUBSETS,
            )
            print(f"[grid] Collected general examples: {len(general_examples)}")

            all_examples = interleave_domain_and_general(
                domain_examples,
                general_examples,
                domain_prob=domain_prob,
                seed=42,
            )
            print(f"[grid] Total mixed examples: {len(all_examples)}")

            train_raw, val_raw = split_train_val(all_examples, val_ratio=args.val_ratio, seed=42)
            print(f"[grid] Train size: {len(train_raw)}, Val size: {len(val_raw)}")

            print("[grid] Converting to messages format...")
            train_msgs = [build_messages_from_example(ex) for ex in tqdm(train_raw)]
            val_msgs = [build_messages_from_example(ex) for ex in tqdm(val_raw)]

            model, tok = load_model(args.model)
            train(model, tok, train_msgs, val_msgs, run_output, args.epochs)

            infer_path = os.path.join(run_output, "val_predictions.jsonl")
            results = run_inference(model, tok, val_raw, infer_path)

            cocktail_path = os.path.join(run_output, "data_cocktail_with_judge.jsonl")
            judged, avg_score = judge_with_braintrust(
                results,
                cocktail_path,
                sample_limit=args.judge_sample_limit,
            )

            recommended_path = os.path.join(run_output, "recommended_sample.json")
            recommend_highest_score(judged, recommended_path)

            config_info = {
                "domain_ratio": domain_prob,
                "subset_index": subset_idx,
                "subset_weights": subset_weights,
                "G_target": G_target,
                "checkpoint_dir": run_output,
                "avg_score": avg_score,
            }
            grid_results.append(config_info)

            print(f"[grid] Config {run_tag} avg_score = {avg_score}")

            if avg_score is None:
                continue

            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_config = config_info
                best_output = run_output

    # Save grid search summary
    summary_path = f"{args.output}_grid_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(grid_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved grid search summary to {summary_path}")

    print("\n===== Grid search finished =====")
    if best_config is None:
        print("Could not determine a best configuration.")
    else:
        print("Best configuration:")
        print(json.dumps(best_config, ensure_ascii=False, indent=2))
        print(f"Best checkpoint folder: {best_output}")


# ============================================================
# 10. CLI and main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--domain_data", required=True, help="Path to domain JSONL (your schema).")
    p.add_argument("--max_domain_samples", type=int, default=9500, help="Max domain samples.")
    p.add_argument(
        "--general_samples",
        type=int,
        default=0,
        help="Optional cap on total general (FineVision) samples. "
             "If 0, use full G = D*(1-p)/p."
    )
    p.add_argument("--tmp_image_dir", default="tmp_images", help="Folder to store streamed images.")
    p.add_argument("--output", default="mixed_finetuned", help="Base model output directory.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--run_judge", action="store_true", help="If set, run LLM judge and grid search.")
    p.add_argument("--judge_sample_limit", type=int, default=100, help="Max samples to send to judge.")

    return p.parse_args()


def main():
    args = parse_args()
    run_grid_search(args)


if __name__ == "__main__":
    main()
