import os
import json
import re
import base64
from pathlib import Path

import requests
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
INPUT_JSONL = "./data_test.jsonl"
OUTPUT_JSONL = "output_with_model.jsonl"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9

SYSTEM_PROMPT = (
    "You are a helpful assistant for table question answering.\n"
    "You will be given a table image plus a text context and a question.\n"
    "Answer using the table and context.\n"
    "Return output in this exact format:\n"
    "ANSWER: <short answer>\n"
    "RATIONALE: <brief explanation grounded in the table/context>\n"
)

# ----------------------------
# Prompt helpers
# ----------------------------
def build_user_text(context: str, question: str) -> str:
    context = (context or "").strip()
    question = (question or "").strip()
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

# ----------------------------
# vLLM client helpers
# ----------------------------
def get_model_id() -> str:
    r = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=30)
    r.raise_for_status()
    j = r.json()
    if not j.get("data"):
        raise RuntimeError("No models returned by /v1/models")
    return j["data"][0]["id"]

def img_to_data_url(image_path: Path) -> str:
    b = image_path.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")

    # best effort mime
    suf = image_path.suffix.lower()
    if suf in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suf in [".webp"]:
        mime = "image/webp"
    else:
        mime = "image/png"

    return f"data:{mime};base64,{b64}"

def vllm_chat(model_id: str, prompt_text: str, image_path: Path) -> str:
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": img_to_data_url(image_path)}},
                ],
            },
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    r = requests.post(url, json=payload, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1200]}")
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

# ----------------------------
# Main loop
# ----------------------------
def main():
    model_id = get_model_id()
    print("Using model id:", model_id)

    in_path = Path(INPUT_JSONL)
    out_path = Path(OUTPUT_JSONL)

    safe_model_key = Path(model_id).name.replace("/", "_")
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
            prompt_text = build_user_text(context, question)

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
                raw = vllm_chat(model_id, prompt_text, p)
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
