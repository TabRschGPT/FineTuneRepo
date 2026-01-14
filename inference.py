import os
import json
import re
import base64
from pathlib import Path
from io import BytesIO

import requests
from tqdm import tqdm
from PIL import Image

# ============================================================
# Config
# ============================================================
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")

# IMPORTANT: your vLLM model id is the path returned by /v1/models
# If you do not set it, we will auto-pick the first model from /v1/models
VLLM_MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct"

INPUT_JSONL = "./output_with_model.jsonl"
OUTPUT_JSONL = "output_with_model_new.jsonl"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9

# Prompt + size controls (prevents "prompt length > max_model_len" 400s)
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "1800"))
MAX_QUESTION_CHARS = int(os.environ.get("MAX_QUESTION_CHARS", "600"))

# Resize image to reduce MM tokens. Tune down if still too long.
MAX_IMAGE_WIDTH = int(os.environ.get("MAX_IMAGE_WIDTH", "1024"))

# If a request still overflows, we retry once with stronger truncation + smaller image.
RETRY_ON_PROMPT_TOO_LONG = True
RETRY_CONTEXT_CHARS = int(os.environ.get("RETRY_CONTEXT_CHARS", "900"))
RETRY_IMAGE_WIDTH = int(os.environ.get("RETRY_IMAGE_WIDTH", "768"))

SYSTEM_PROMPT = (
    "You are a helpful assistant for table question answering.\n"
    "You will be given a table image plus a text context and a question.\n"
    "Answer using the table and context.\n"
    "Return output in this exact format:\n"
    "ANSWER: <short answer>\n"
    "RATIONALE: <brief explanation grounded in the table/context>\n"
)

# ============================================================
# Text helpers
# ============================================================
def clamp_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "..."

def build_user_text(context: str, question: str, ctx_chars: int, q_chars: int) -> str:
    context = clamp_text(context, ctx_chars)
    question = clamp_text(question, q_chars)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n"

def parse_answer_rationale(text: str):
    if not text:
        return None, None

    t = text.strip()

    think_blocks = re.findall(r"(?is)<think>(.*?)</think>", t)
    rationale = "\n\n".join(tb.strip() for tb in think_blocks if tb.strip()) if think_blocks else ""

    t_wo_think = re.sub(r"(?is)<think>.*?</think>", " ", t).strip()

    ans_patterns = [
        r"(?is)\bFINAL\s*ANSWER\b\s*[:\-]\s*(.+?)(?:\n|$)",
        r"(?is)\bFINAL\b\s*[:\-]\s*(.+?)(?:\n|$)",
        r"(?is)\bANSWER\b\s*[:\-]\s*(.+?)(?:\n|$)",
        r"(?is)\bANS\b\s*[:\-]\s*(.+?)(?:\n|$)",
        r"(?is)\bAnswer\b\s*[:\-]\s*(.+?)(?:\n|$)",
    ]

    answer = None
    for pat in ans_patterns:
        m = re.search(pat, t_wo_think)
        if m:
            answer = m.group(1).strip()
            break

    rat_patterns = [
        r"(?is)\bRATIONALE\b\s*[:\-]\s*(.+)$",
        r"(?is)\bRationale\b\s*[:\-]\s*(.+)$",
        r"(?is)\bEXPLANATION\b\s*[:\-]\s*(.+)$",
        r"(?is)\bBecause\b\s*(.+)$",
    ]
    for pat in rat_patterns:
        m = re.search(pat, t_wo_think)
        if m:
            explicit_rat = m.group(1).strip()
            if explicit_rat:
                rationale = explicit_rat
            break

    if answer:
        answer = re.sub(r"(?is)\s*(?:RATIONALE|Rationale|EXPLANATION)\b.*$", "", answer).strip()
        answer = answer.strip().strip('"').strip("'")

    if not answer:
        lines = [ln.strip() for ln in t_wo_think.splitlines() if ln.strip()]
        if lines:
            answer = lines[-1]
            answer = re.split(r"\s{2,}|\n", answer)[0].strip()

    if answer == "":
        answer = None
    if rationale == "":
        rationale = None

    return answer, rationale

# ============================================================
# Image helpers (resize -> base64 data URL)
# ============================================================
def _guess_mime(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"

def image_path_to_data_url(image_path: Path, max_width: int) -> str:
    if image_path.stat().st_size == 0:
        raise RuntimeError(f"Empty image file: {image_path}")

    img = Image.open(image_path).convert("RGB")

    if img.width > max_width:
        new_h = int(img.height * (max_width / img.width))
        img = img.resize((max_width, new_h))

    # Encode as PNG to be safe and stable
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ============================================================
# vLLM client
# ============================================================
def get_model_id() -> str:
    if VLLM_MODEL_ID:
        return VLLM_MODEL_ID

    r = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=30)
    r.raise_for_status()
    j = r.json()
    if not j.get("data"):
        raise RuntimeError("No models returned by /v1/models")
    return j["data"][0]["id"]

def vllm_chat_once(model_id: str, prompt_text: str, image_path: Path, image_width: int) -> str:
    url = f"{VLLM_BASE_URL}/v1/chat/completions"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path, image_width)}},
                ],
            },
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    r = requests.post(url, json=payload, timeout=600)

    if r.status_code != 200:
        # keep server message (vLLM gives useful reasons)
        raise RuntimeError(f"HTTP {r.status_code} | response: {r.text[:2000]}")

    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

def vllm_chat(model_id: str, context: str, question: str, image_path: Path) -> str:
    # first attempt
    prompt = build_user_text(context, question, MAX_CONTEXT_CHARS, MAX_QUESTION_CHARS)
    try:
        return vllm_chat_once(model_id, prompt, image_path, MAX_IMAGE_WIDTH)
    except RuntimeError as e:
        msg = str(e)

        # Retry only for "prompt too long" style errors
        if RETRY_ON_PROMPT_TOO_LONG and ("maximum model length" in msg or "decoder prompt" in msg):
            prompt2 = build_user_text(context, question, RETRY_CONTEXT_CHARS, MAX_QUESTION_CHARS)
            return vllm_chat_once(model_id, prompt2, image_path, RETRY_IMAGE_WIDTH)

        raise

# ============================================================
# Main
# ============================================================
def main():
    model_id = get_model_id()
    print("Using model id:", model_id)
    print(f"vLLM base url: {VLLM_BASE_URL}")

    in_path = Path(INPUT_JSONL)
    out_path = Path(OUTPUT_JSONL)

    # Build safe keys for JSON output
    safe_model_key = "qwen3_8b_vl_raw"
    ans_key = f"answer_{safe_model_key}"
    rat_key = f"rationale_{safe_model_key}"
    raw_key = f"raw_{safe_model_key}"

    n_lines = sum(1 for _ in in_path.open("r", encoding="utf-8"))

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=n_lines, desc=f"Running vLLM ({safe_model_key})"):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            context = item.get("context", "")
            question = item.get("question", "")

            img_path = item.get("table_image", None)
            if not img_path:
                item[ans_key] = None
                item[rat_key] = None
                item[raw_key] = "ERROR: missing table_image"
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            p = Path(img_path)
            if not p.exists():
                item[ans_key] = None
                item[rat_key] = None
                item[raw_key] = f"ERROR: image not found: {str(p)}"
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            try:
                raw = vllm_chat(model_id, context, question, p)
                ans, rat = parse_answer_rationale(raw)

                item[ans_key] = ans
                item[rat_key] = rat
                item[raw_key] = raw

            except Exception as e:
                item[ans_key] = None
                item[rat_key] = None
                item[raw_key] = f"ERROR: {type(e).__name__}: {e}"

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Done. Wrote:", out_path.resolve())

if __name__ == "__main__":
    main()
