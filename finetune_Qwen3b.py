#!/usr/bin/env python
# coding: utf-8

"""
Simple generator fine-tuning script for Unsloth + Qwen3-VL-8B.

Dataset requirement:
- JSONL file with keys: question, context, answer, image_path

Training:
python train_generator.py \
    --data filter.jsonl \
    --output iteration1_model
"""

import argparse
from datasets import load_dataset, Dataset, Image, load_from_disk
from tqdm import tqdm

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


# ============================================================
# 1. Load Model + LoRA
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


# ============================================================
# 2. Build Messages for Generator Training
# ============================================================

def build_messages(example):
    # Use the "image" column that you already cast to Image()
    img = example["image"]    # HF Image or PIL
    # High level instruction for the model
    instruction = (
        "You are a chemistry expert. You look at scientific tables or figures and "
        "answer user questions based on both the image and the text context."
    )

    user_text = (
        f"{instruction}\n\n"
        f"Question: {example['question']}\n"
        f"Context: {example['context']}\n"
    )

    assistant_text = (
        f"<think>{example.get('rationale', '')}</think> "
        f"Answer: {example['answer']}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": img},
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
# def build_messages(ex):
#     img = ex["image"]

#     user_text = (
#         "You are given a question and context. Provide the correct answer.\n"
#         f"Question: {ex['question']}\n"
#         f"Context: {ex['context']}\n"
#     )

#     return {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": user_text},
#                     {"type": "image", "image": img},
#                 ],
#             },
#             {
#                 "role": "assistant",
#                 "content": [{"type": "text", "text": ex["answer"]}],
#             },
#         ]
#     }


# ============================================================
# 3. Prepare Dataset
# ============================================================
def prepare_dataset(path):
    ds = load_dataset("json", data_files=path, split="train")
    # ds = load_from_disk("sampled")
    ds =  ds.rename_column("file", "image")
    # if "image_path" not in ds.column_names:
    #     raise ValueError("Dataset must contain 'image_path'")

    # ds = ds.cast_column("image_path", Image())
    MAX_SAMPLES = 9500

    # ------------------------------------------
    # 1. Remove artefact_type == "logo"
    # ------------------------------------------
    filtered = ds.filter(
        lambda x: x["artefact_type"] != "logo"
    )
    
    print("After filtering logo:", len(filtered))
    
    if len(filtered) < MAX_SAMPLES:
        raise ValueError(
            f"Not enough samples after filter. Only {len(filtered)} remain."
        )
    
    # ------------------------------------------
    # 2. Shuffle + Random sample 9500
    # ------------------------------------------
    sampled = filtered.shuffle(seed=42).select(range(MAX_SAMPLES))
    # print("Sampled:", len(sampled))
    # sampled = load_from_disk("sampled")
    # ------------------------------------------
    # 3. Train-test split (simple, non-stratified)
    # ------------------------------------------
    splits = sampled.train_test_split(
        test_size=0.1,  # 90 percent train, 10 percent val
        seed=42,
    )
    
    train_split = splits["train"]
    val_split   = splits["test"]
    
    print("Train size:", len(train_split))
    print("Val size:", len(val_split))
    print("Converting dataset to messages...")
    converted_train = [build_messages(sample) for sample in tqdm(train_split)]
    converted_val = [build_messages(sample) for sample in tqdm(val_split)]
    return converted_train, converted_val 

# ============================================================
# 4. Train + Save Model
# ============================================================
def train(model, tokenizer, train_ds, val_ds, output, epochs):
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset = val_ds,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # max_steps = 100,
            learning_rate = 2e-4,
            logging_steps = 1,
            num_train_epochs = 2, # Set this instead of max_steps for full training runs
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
    
            seed = 3407,
            output_dir = "outputs",
    
            # ✅ ENABLE TENSORBOARD
            report_to = "wandb",           
            # <--- HERE
    
            # Optional: set logging directory
            logging_dir = "outputs/tensorboard", # <--- LOG DIR
    
            # Required Unsloth VLM flags
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
            ),
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    print(f"Saving model to: {output}")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)


# ============================================================
# 5. CLI
# ============================================================
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to filter.jsonl")
    p.add_argument("--output", default="finetuned_generator", help="Save directory")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct")
    return p.parse_args()


# ============================================================
# 6. Main
# ============================================================
def main():
    args = parse()

    model, tok = load_model(args.model)
    converted_train, converted_val  = prepare_dataset(args.data)
    train(model, tok, converted_train, converted_val, args.output, args.epochs)


if __name__ == "__main__":
    main()

