import os
# MUST be before importing unsloth/transformers
os.environ["HF_HOME"] = "/workspace/work/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/work/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/work/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/workspace/work/hf_cache/datasets"
os.environ["XDG_CACHE_HOME"] = "/workspace/work/hf_cache"
os.environ["TMPDIR"] = "/workspace/work/tmp"

from unsloth import FastLanguageModel

MODEL_NAME = "vietmed/qwen3vl_peft_generator"
OUT_DIR = "/workspace/work/merged_qwen3vl_16bit"

def main():
    os.makedirs("/workspace/work/tmp", exist_ok=True)
    os.makedirs("/workspace/work/hf_cache/hub", exist_ok=True)
    os.makedirs("/workspace/work/hf_cache/transformers", exist_ok=True)
    os.makedirs("/workspace/work/hf_cache/datasets", exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=4096,
        load_in_4bit=True,   # ok for loading, export will be merged_16bit
        dtype=None,
    )

    model.save_pretrained_merged(
        OUT_DIR,
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"✅ merged_16bit saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
