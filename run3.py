import torch
import os
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any
# FIX 1: Import Image as HFImage to avoid conflict with PIL
from datasets import Dataset, load_dataset, Image as HFImage
from PIL import Image
from tqdm import tqdm

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# --- CONFIGURATION ---
@dataclass
class AlgoConfig:
    base_model: str = "unsloth/Qwen3-VL-8B-Instruct"
    iterations_k: int = 3           # (Algorithm Line 3)
    step_size_j: int = 50           # (Algorithm Line 6) Samples per iteration
    
    # Paths for our two specialists
    gen_adapter_path: str = "checkpoints/generator_adapter"
    cls_adapter_path: str = "checkpoints/classifier_adapter"
    
    # Training Hyperparams
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
    epochs_per_iter: int = 2
    batch_size: int = 2
    grad_accum: int = 4

cfg = AlgoConfig()

# --- 1. MODEL & ADAPTER MANAGEMENT ---

class DualModelManager:
    def __init__(self, config: AlgoConfig):
        self.cfg = config
        # Load Base Model (Shared Body)
        print(f"Loading Base Model: {config.base_model}")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            config.base_model,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        
        # Initialize/Save empty adapters for both tasks if they don't exist
        self._init_adapter(self.cfg.gen_adapter_path)
        self._init_adapter(self.cfg.cls_adapter_path)

    def _init_adapter(self, path):
        """Creates an initial LoRA configuration and saves it."""
        if not os.path.exists(path):
            print(f"Initializing fresh adapter at {path}")
            # Add LoRA adapters
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers=True, 
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16, lora_alpha=16, lora_dropout=0, bias="none",
                random_state=3407, use_rslora=False,
            )
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Unload to reset state
            print(f"Unloading adapter from {path} to reset model state...")
            self.model = self.model.unload()

    def _clean_active_adapters(self):
        """
        Aggressively removes all active adapters to prevent collisions.
        """
        # 1. Unload via Unsloth native method if available to detach weights
        try:
            self.model.unload() 
        except:
            pass

        # 2. Explicitly delete from PEFT config
        if hasattr(self.model, "peft_config"):
            # Create a list of keys to avoid runtime error during dictionary iteration
            adapter_names = list(self.model.peft_config.keys())
            for name in adapter_names:
                try:
                    self.model.delete_adapter(name)
                except Exception as e:
                    # Sometimes 'default' cannot be deleted if it's the base, ignore
                    pass

    def load_generator(self, inference=False):
        """Switches active adapter to Generator (MG)"""
        print("⚡ Swapping to GENERATOR Adapter...")
        
        # 1. Clean up previous adapter
        self._clean_active_adapters()
        
        # 2. Load new adapter with EXPLICIT name to avoid "default" collision
        self.model.load_adapter(self.cfg.gen_adapter_path, adapter_name="generator")
        self.model.set_adapter("generator")
        
        # 3. Set mode
        if inference: FastVisionModel.for_inference(self.model)
        else: FastVisionModel.for_training(self.model)

    def load_classifier(self, inference=False):
        """Switches active adapter to Classifier (MC)"""
        print("⚡ Swapping to CLASSIFIER Adapter...")
        
        # 1. Clean up previous adapter
        self._clean_active_adapters()
        
        # 2. Load new adapter with EXPLICIT name
        self.model.load_adapter(self.cfg.cls_adapter_path, adapter_name="classifier")
        self.model.set_adapter("classifier")
        
        # 3. Set mode
        if inference: FastVisionModel.for_inference(self.model)
        else: FastVisionModel.for_training(self.model)

# --- 2. DATA FORMATTING ---

def format_for_gen(example):
    """Algorithm Line 8: Construct tG"""
    user_text = (
        f"Question: {example['question']}\n"
        f"Context: {example['context']}\n"
        "Answer:"
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image", "image": example['image']}]},
        {"role": "assistant", "content": [{"type": "text", "text": example['answer']}]}
    ]
    return {"messages": messages}

def format_for_cls(example, predicted_answer, label):
    """Algorithm Line 10: Construct tC"""
    user_text = (
        "Check if the proposed answer is correct.\n"
        f"Question: {example['question']}\n"
        f"Context: {example['context']}\n"
        f"Proposed Answer: {predicted_answer}\n"
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image", "image": example['image']}]},
        {"role": "assistant", "content": [{"type": "text", "text": label}]} # "correct" or "incorrect"
    ]
    return {"messages": messages}

# --- 3. ALGORITHM STEPS ---

def step_generate(manager, batch_data):
    """Algorithm Line 9: c <- MG(tG)"""
    manager.load_generator(inference=True)
    results = []
    
    # Simple inference loop (optimize with batching in prod)
    print("--- Generating Candidates (MG) ---")
    for item in tqdm(batch_data):
        # Construct Prompt
        msgs = format_for_gen(item)['messages']
        # Remove the 'assistant' part for inference
        msgs = [msgs[0]] 
        
        input_text = manager.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = manager.tokenizer(item['image'], input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = manager.model.generate(**inputs, max_new_tokens=512, use_cache=True)
        
        prompt_len = inputs["input_ids"].shape[1]
        decoded = manager.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
        
        results.append({**item, "generated_c": decoded})
    return results

def step_validate(manager, batch_data_with_c):
    """Algorithm Line 11: Validate(MC, tC, c)"""
    manager.load_classifier(inference=True)
    validated_G = [] # For Generator (contains only Correct items)
    validated_C = [] # For Classifier (contains Correct AND Incorrect items with labels)
    
    print("--- Validating Candidates (MC) ---")
    for item in tqdm(batch_data_with_c):
        c = item["generated_c"]
        
        # 1. Construct Prompt for MC
        # Note: We ask MC to output "correct" or "incorrect"
        msgs = format_for_cls(item, c, "")['messages']
        msgs = [msgs[0]] # Only User prompt
        
        input_text = manager.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = manager.tokenizer(item['image'], input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = manager.model.generate(**inputs, max_new_tokens=10, use_cache=True)
            
        prompt_len = inputs["input_ids"].shape[1]
        verdict = manager.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).lower().strip()
        
        # 2. Logic: Is it valid?
        # In strict Algorithm 1, we trust MC. 
        # (Optionally: You can mix in Ground Truth here to bootstrap early iterations)
        is_valid = "correct" in verdict
        
        if is_valid:
            # Line 12: Add (tG, c) to TrainG
            # We update the item's 'answer' to be the generated one (self-training)
            item_gen = item.copy()
            item_gen['answer'] = c 
            validated_G.append(item_gen)
            
            # Line 13: Add (tC, c) to TrainC with label "correct"
            item_cls = item.copy()
            item_cls['answer_correct'] = "correct"
            validated_C.append(item_cls)
        else:
            # Implicit: Also teach MC what 'incorrect' looks like
            item_cls = item.copy()
            item_cls['answer_correct'] = "incorrect"
            validated_C.append(item_cls)
            
    return validated_G, validated_C

def step_finetune(manager, dataset, mode="generator"):
    """Algorithm Line 14 & 15"""
    if len(dataset) == 0:
        print(f"Skipping training for {mode} (No data)")
        return

    if mode == "generator":
        manager.load_generator(inference=False)
        save_path = manager.cfg.gen_adapter_path
        formatted_ds = [format_for_gen(x) for x in dataset]
    else:
        manager.load_classifier(inference=False)
        save_path = manager.cfg.cls_adapter_path
        # Need to map the 'answer_correct' back to message format
        formatted_ds = []
        for x in dataset:
            # Use generated 'c' as the proposed answer in prompt
            formatted_ds.append(format_for_cls(x, x['generated_c'], x['answer_correct']))

    trainer = SFTTrainer(
        model=manager.model,
        tokenizer=manager.tokenizer,
        data_collator=UnslothVisionDataCollator(manager.model, manager.tokenizer),
        train_dataset=formatted_ds,
        args=SFTConfig(
            per_device_train_batch_size=manager.cfg.batch_size,
            gradient_accumulation_steps=manager.cfg.grad_accum,
            warmup_steps=5,
            max_steps=10, # Short bursts of fine-tuning
            learning_rate=manager.cfg.learning_rate,
            output_dir="outputs_temp",
            optim="adamw_8bit",
            seed=3407,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=manager.cfg.max_seq_length,
        ),
    )
    
    trainer.train()
    
    # Save the updated adapter weights back to disk
    print(f"Saving updated {mode} adapter...")
    manager.model.save_pretrained(save_path)
    manager.tokenizer.save_pretrained(save_path)

# --- 4. MAIN EXECUTION (ALGORITHM 1) ---

def main_algorithm_1():
    # Initialize Manager
    manager = DualModelManager(cfg)
    
    # Load your raw dataset R
    print("Loading Dataset...")
    raw_dataset = load_dataset("json", data_files="./data_Finetune1.jsonl", split="train")
    
    # --- FIX STARTS HERE ---
    # 1. Your JSONL uses "file", but the script expects "image". Rename it.
    if "file" in raw_dataset.column_names:
        print("Renaming 'file' column to 'image'...")
        raw_dataset = raw_dataset.rename_column("file", "image")
    
    # 2. Ensure paths are absolute (Optional but recommended if images are not in current dir)
    # If your images are in a subfolder relative to where you run the script, this step ensures they are found.
    def update_path(example):
        # Check if it's already an absolute path or a URL
        if not example['image'].startswith("/") and not example['image'].startswith("http"):
            # Adjust this base path if your 'Extracted' folder is elsewhere
            example['image'] = os.path.abspath(example['image']) 
        return example

    # Map the path update ONLY if strictly necessary (usually HFImage handles relative paths if CWD is correct)
    # raw_dataset = raw_dataset.map(update_path) 

    # 3. Use HFImage() for casting (Now that the column is named 'image')
    raw_dataset = raw_dataset.cast_column("image", HFImage())
    # --- FIX ENDS HERE ---
    
    # Algorithm Line 3: for i in 1 to k
    for i in range(1, cfg.iterations_k + 1):
        print(f"\n\n=== Iteration {i}/{cfg.iterations_k} ===")
        
        # Line 4 & 5: Reset training sets
        TrainG = []
        TrainC = []
        
        # Line 6: Step size loop
        start_idx = (i-1) * cfg.step_size_j
        end_idx = start_idx + cfg.step_size_j
        
        # Check bounds to prevent empty selection errors
        if start_idx >= len(raw_dataset):
            print("Maximum dataset size reached.")
            break
            
        current_batch = raw_dataset.select(range(start_idx, min(end_idx, len(raw_dataset))))
        
        if len(current_batch) == 0:
            print("Out of data!")
            break

        # Line 8 & 9: Instantiate tG and Compute c (Using MG)
        candidates = step_generate(manager, current_batch)
        
        # Line 10 & 11: Construct tC and Validate (Using MC)
        # Line 12 & 13: Accumulate to TrainG and TrainC
        valid_G_data, valid_C_data = step_validate(manager, candidates)
        
        print(f"Stats: {len(valid_G_data)} validated samples added to TrainG")
        
        # Line 14: Fine-tune MG
        print("--- Fine-tuning Generator (MG) ---")
        step_finetune(manager, valid_G_data, mode="generator")
        
        # Line 15: Fine-tune MC
        print("--- Fine-tuning Classifier (MC) ---")
        step_finetune(manager, valid_C_data, mode="classifier")
        
    print("\nAlgorithm 1 Complete. Final models stored in checkpoints/.")

if __name__ == "__main__":
    main_algorithm_1()