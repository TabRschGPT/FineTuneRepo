#!/usr/bin/env python
# coding: utf-8

"""
Mixed-domain VLM fine-tune script for Unsloth + Qwen3-VL-8B.

Pipeline:
1. Load domain JSONL (your patent tables etc).
2. Filter out artefact_type == "logo".
3. Random sample from domain data.
4. From huggingface-m4/finevision, stream subsets in RECOMMENDED_FINEVISION_SUBSETS.
5. From each subset, sample some general vision QA examples.
6. For streamed general data, dump image objects to a temporary folder
   and replace the image field with the file path (always .png).
7. Normalize both domain and general examples into a common schema.
8. Mix the data and split into train and val.
9. Convert to Unsloth "messages" format and fine tune.
10. Run inference on the validation split.
11. Save prediction results.
12. Use an LLM judge (gpt-5 over Braintrust) that sees only:
    - question
    - context
    - model answer
    - image path
    Gold label is stored only for analysis.
13. Save a cocktail JSONL with judge scores.
14. Recommend the highest score sample and save it as JSON.
"""

import argparse
import io
import json
import os
import random
import uuid
from pathlib import Path

from datasets import load_dataset, Image as HFImage
from tqdm import tqdm

import torch
from PIL import Image as PILImage

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Optional judge client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


RECOMMENDED_FINEVISION_SUBSETS = [
    "CoSyn_400k_table",
    "docvqa",
    "pdfvqa",
    "textvqa",
    "chartqa",
    "infographic_vqa",
]


# ============================================================
# 1. Helpers for images on disk
# ============================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_stream_image_to_disk(img_obj, tmp_dir: str) -> str:
    """
    Save a streamed image to disk as a real PNG file.
    Always produces: tmp_images/<uuid>.png

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

        # Embedded path inside dict
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

    # If nothing matched, throw
    raise TypeError(
        f"Unsupported image type from FineVision: {type(img_obj)} "
        f"Preview: {repr(img_obj)[:200]}"
    )


# ============================================================
# 2. Domain dataset: JSONL loading and sampling
# ============================================================

def load_domain_jsonl(path: str):
    """Load your patent style JSONL into a list of dicts."""
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
    Load domain JSONL, filter artefact_type == "logo" if requested,
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

    # Normalize to common schema
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
# 3. FineVision general data: streaming and sampling
# ============================================================

def adapt_finevision_example(ex) -> dict:
    """
    Map one FineVision example to the common schema.

    FineVision schema often has:
    - "images": list of images
    - "texts": list of text turns or captions
    - rating fields

    We try:
    1. Use explicit "question"/"query"/"answer"/"answers" when present.
    2. If not present, use "texts" to synthesize:
       - if len(texts) >= 2: first text as question, second as answer
       - if len(texts) == 1: generic question, text as answer
    3. Use the first element of "images" as the image.
    """

    # Try standard QA fields
    question = ex.get("question") or ex.get("query") or ""
    context = ex.get("context") or ""
    answer = ex.get("answer")
    if answer is None and isinstance(ex.get("answers"), list) and ex["answers"]:
        answer = ex["answers"][0]

    # Fallback to "texts" if no question or answer
    if not question or answer is None:
        texts = ex.get("texts")
        if isinstance(texts, list) and len(texts) >= 2:
            # Treat as Q A pair
            if not question:
                question = texts[0]
            if answer is None:
                answer = texts[1]
        elif isinstance(texts, list) and len(texts) == 1:
            # One caption only
            if not question:
                question = "Describe the image in detail."
            if answer is None:
                answer = texts[0]

    # If still empty, skip this example
    if not question:
        raise KeyError(
            f"No usable question or texts in FineVision example. Keys: {list(ex.keys())}"
        )
    if answer is None:
        # Allow empty string as answer if needed
        answer = ""

    # Try to find an image-like field
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
    subsets=None,
):
    """
    Stream from huggingface-m4/finevision for given subsets.
    Sample up to total_general_samples in total across subsets.
    For each example, dump image to disk and store path.

    Robust to schema differences:
    - Skips examples that raise KeyError in adapt_finevision_example.
    """

    from math import ceil

    if subsets is None:
        subsets = RECOMMENDED_FINEVISION_SUBSETS

    per_subset = max(1, ceil(total_general_samples / max(1, len(subsets))))
    all_samples = []

    for subset_name in subsets:
        if len(all_samples) >= total_general_samples:
            break

        print(f"Streaming FineVision subset: {subset_name}")
        ds_stream = load_dataset(
            "HuggingFaceM4/FineVision",
            subset_name,
            split="train",
            streaming=True,
        )

        count = 0
        for ex in ds_stream:
            if count >= per_subset or len(all_samples) >= total_general_samples:
                break

            try:
                adapted = adapt_finevision_example(ex)
            except KeyError:
                # Example has no usable image or texts, skip
                continue

            img_obj = adapted.pop("image_obj", None)
            img_path = save_stream_image_to_disk(img_obj, tmp_image_dir)

            adapted["image_path"] = img_path
            all_samples.append(adapted)
            count += 1

        print(f"Collected {count} samples from {subset_name}")

    if not all_samples:
        raise ValueError(
            "No general FineVision samples collected. "
            "Check subset names or schema."
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
    Use image path string directly. Unsloth will load it.
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


def prepare_mixed_dataset(
    domain_jsonl: str,
    max_domain_samples: int,
    total_general_samples: int,
    tmp_image_dir: str,
    val_ratio: float = 0.1,
):
    """
    Full data preparation:

    1. Domain JSONL to filter logos and sample.
    2. FineVision streaming to sample general data.
    3. Mix and split.
    4. Build messages dataset for Unsloth training.
    5. Also keep raw val examples for inference and judge.
    """
    # Domain
    domain_examples = sample_domain_data(
        jsonl_path=domain_jsonl,
        max_domain_samples=max_domain_samples,
        exclude_logo=True,
    )
    print(f"Domain examples after filter and sample: {len(domain_examples)}")

    # General
    general_examples = []
    if total_general_samples > 0:
        general_examples = sample_finevision_general(
            tmp_image_dir=tmp_image_dir,
            total_general_samples=total_general_samples,
            subsets=RECOMMENDED_FINEVISION_SUBSETS,
        )
    print(f"General FineVision examples: {len(general_examples)}")

    # Mix and split
    all_examples = domain_examples + general_examples
    print(f"Total mixed examples: {len(all_examples)}")

    train_raw, val_raw = split_train_val(all_examples, val_ratio=val_ratio, seed=42)
    print(f"Train size: {len(train_raw)}, Val size: {len(val_raw)}")

    # Convert to messages
    print("Converting to messages format for Unsloth...")
    train_msgs = [build_messages_from_example(ex) for ex in tqdm(train_raw)]
    val_msgs = [build_messages_from_example(ex) for ex in tqdm(val_raw)]

    return train_msgs, val_msgs, val_raw


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
    """User only messages for generation."""
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
    """
    Use gpt-5 through Braintrust proxy to judge predictions.

    Needs:
    - BRAINTRUST_BASE_URL
    - BRAINTRUST_API_KEY

    Judge sees:
    - question
    - context
    - model answer
    - image path

    It does not see or trust the gold label. Gold answer is only stored in the record.
    """
    from dotenv import load_dotenv
    load_dotenv()
    if OpenAI is None:
        print("openai library not installed. Skip judge.")
        return []
    
    base_url = os.environ.get("BRAINTRUST_BASE_URL")
    api_key = os.environ.get("BRAINTRUST_API_KEY")

    if not base_url or not api_key:
        print("Braintrust env vars not set. Skip judge.")
        return []

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

    return judged


# ============================================================
# 8. Recommend highest score and save JSON
# ============================================================

def recommend_highest_score(judged_results, output_path: str):
    """
    Given a list of results containing judge_score,
    find the sample with the highest score and save it as a JSON file.
    If there are ties, take the first one.
    """
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
# 9. CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--domain_data", required=True, help="Path to domain JSONL (your schema).")
    p.add_argument("--max_domain_samples", type=int, default=9500, help="Max domain samples.")
    p.add_argument("--general_samples", type=int, default=2000, help="Total FineVision general samples.")
    p.add_argument("--tmp_image_dir", default="tmp_images", help="Folder to store streamed images.")
    p.add_argument("--output", default="mixed_finetuned", help="Model output directory.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--run_judge", action="store_true", help="If set, run LLM judge.")
    p.add_argument("--judge_sample_limit", type=int, default=100, help="Max samples to send to judge.")

    return p.parse_args()


def main():
    args = parse_args()

    # 1. Data
    train_msgs, val_msgs, val_raw = prepare_mixed_dataset(
        domain_jsonl=args.domain_data,
        max_domain_samples=args.max_domain_samples,
        total_general_samples=args.general_samples,
        tmp_image_dir=args.tmp_image_dir,
        val_ratio=args.val_ratio,
    )

    # 2. Model
    model, tok = load_model(args.model)

    # 3. Train
    train(model, tok, train_msgs, val_msgs, args.output, args.epochs)

    # 4. Inference on val set
    infer_path = os.path.join(args.output, "val_predictions.jsonl")
    results = run_inference(model, tok, val_raw, infer_path)

    # 5. Judge and cocktail plus recommendation
    if args.run_judge:
        cocktail_path = os.path.join(args.output, "data_cocktail_with_judge.jsonl")
        judged = judge_with_braintrust(
            results,
            cocktail_path,
            sample_limit=args.judge_sample_limit,
        )

        recommended_path = os.path.join(args.output, "recommended_sample.json")
        recommend_highest_score(judged, recommended_path)


if __name__ == "__main__":
    main()
