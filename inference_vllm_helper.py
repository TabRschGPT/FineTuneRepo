import os
import json
import re
import base64
import argparse
from pathlib import Path
from io import BytesIO

import requests
from tqdm import tqdm
from PIL import Image

# ============================================================
# Defaults
# ============================================================
DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")

DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

DEFAULT_MAX_CONTEXT_CHARS = 1800
DEFAULT_MAX_QUESTION_CHARS = 600
DEFAULT_MAX_IMAGE_WIDTH = 1024

# ============================================================
# SYSTEM PROMPT (ANSWER + RATIONALE ONLY)
# ============================================================
SYSTEM_PROMPT = (
    "You are a chemical patent table question answering assistant.\n"
    "Use ONLY the provided table image and context.\n\n"
    "Return output in EXACTLY this XML format:\n"
    "<answer>short answer</answer>\n"
    "<rationale>brief explanation grounded in the table</rationale>\n\n"
    "Think step by step, include thoughts, analysis in <rationale> block."
)

# ============================================================
# Helpers
# ============================================================
def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + "..."

def build_user_text(context: str, question: str, max_ctx: int, max_q: int) -> str:
    return (
        f"CONTEXT:\n{clamp_text(context, max_ctx)}\n\n"
        f"QUESTION:\n{clamp_text(question, max_q)}\n"
    )

def parse_answer_rationale(text: str):
    if not text:
        return None, None

    def extract(tag):
        m = re.search(rf"(?is)<{tag}>(.*?)</{tag}>", text)
        return m.group(1).strip() if m else None

    return extract("answer"), extract("rationale")

# ============================================================
# Image handling
# ============================================================
def image_to_data_url(image_path: Path, max_width: int) -> str:
    img = Image.open(image_path).convert("RGB")

    if img.width > max_width:
        new_h = int(img.height * (max_width / img.width))
        img = img.resize((max_width, new_h))

    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ============================================================
# vLLM client
# ============================================================
def get_model_id(vllm_base_url: str) -> str:
    r = requests.get(f"{vllm_base_url}/v1/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]

def clean_model_key(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", os.path.basename(model_id))

def chat_once(
    vllm_base_url: str,
    model_id: str,
    prompt_text: str,
    image_path: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_image_width: int,
) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path, max_image_width)},
                    },
                ],
            },
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": ["</rationale>"],
    }

    r = requests.post(f"{vllm_base_url}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--vllm_base_url", default=DEFAULT_VLLM_BASE_URL)
    args = parser.parse_args()

    model_id = get_model_id(args.vllm_base_url)
    model_key = clean_model_key(model_id)

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Running table QA"):
            item = json.loads(line)
            img = item.get("table_image")

            if not img or not Path(img).exists():
                item[f"answer_{model_key}"] = None
                item[f"rationale_{model_key}"] = "ERROR: missing image"
                fout.write(json.dumps(item) + "\n")
                continue

            raw = chat_once(
                args.vllm_base_url,
                model_id,
                build_user_text(item.get("context", ""), item.get("question", ""), 1800, 600),
                Path(img),
                DEFAULT_MAX_NEW_TOKENS,
                DEFAULT_TEMPERATURE,
                DEFAULT_TOP_P,
                DEFAULT_MAX_IMAGE_WIDTH,
            )

            answer, rationale = parse_answer_rationale(raw)

            item["model_id"] = model_id
            item[f"answer_{model_key}"] = answer
            item[f"rationale_{model_key}"] = rationale
            item[f"raw_{model_key}"] = raw

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
