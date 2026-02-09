import os
import json
import re
import argparse
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from tqdm import tqdm

# ===== LangChain =====
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI

# ============================================================
# Load env
# ============================================================
load_dotenv()

# ============================================================
# JUDGE PROMPT (MULTI-AXIS, INDUSTRY STYLE)
# ============================================================
JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of chemical patent table question answering.\n\n"
    "Evaluate the MODEL ANSWER using ONLY the provided TABLE.\n\n"
    "AXES:\n"
    "1. Answer Correctness:\n"
    "   - correct: final answer exactly matches the table\n"
    "   - incorrect: otherwise\n\n"
    "2. Reasoning Quality (0–2):\n"
    "   - 0: incorrect, missing, or nonsensical reasoning\n"
    "   - 1: partially correct but vague or incomplete\n"
    "   - 2: clear, logical, and correctly explains how the answer is obtained from the table\n\n"
    "3. Grounding / Faithfulness (0–2):\n"
    "   - 0: introduces facts not present in the table (hallucination)\n"
    "   - 1: mostly grounded but with minor extrapolation\n"
    "   - 2: fully grounded in the table only\n\n"
    "RULES:\n"
    "- Use ONLY the table.\n"
    "- Do NOT infer or compute.\n"
    "- If Answer Correctness is incorrect, Reasoning Quality MUST be 0.\n"
    "- If the table implies percentage, missing '%' is acceptable.\n\n"
    "OUTPUT MUST BE STRICT XML.\n"
    "Stop immediately after </response>.\n\n"
    "FORMAT:\n"
    "<response>\n"
    "  <correctness>correct|incorrect</correctness>\n"
    "  <reasoning_score>0|1|2</reasoning_score>\n"
    "  <grounding_score>0|1|2</grounding_score>\n"
    "  <final_answer>VALUE_FROM_TABLE</final_answer>\n"
    "  <confidence>0.0-1.0</confidence>\n"
    "</response>\n"
)

# ============================================================
# STRICT XML PARSER
# ============================================================
class JudgeXMLParser(BaseOutputParser):
    def parse(self, text: str):
        if not text:
            raise ValueError("Empty output")

        try:
            root = ET.fromstring(text.strip())
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")

        if root.tag != "response":
            raise ValueError("Root element must be <response>")

        def get(tag):
            el = root.find(tag)
            if el is None or not el.text:
                raise ValueError(f"Missing <{tag}>")
            return el.text.strip()

        return {
            "correctness": get("correctness"),
            "reasoning_score": int(get("reasoning_score")),
            "grounding_score": int(get("grounding_score")),
            "final_answer": get("final_answer"),
            "confidence": float(get("confidence")),
        }

# ============================================================
# XML CLEAN + REPAIR (SAGE STYLE)
# ============================================================
def pre_clean(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    if "<response>" in text:
        text = text[text.find("<response>"):]
    if "</response>" in text:
        text = text[: text.rfind("</response>") + len("</response>")]
    return text.strip()

repair_llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

base_parser = JudgeXMLParser()

def repair_xml(raw: str, error: str) -> str:
    prompt = f"""
The following output MUST be STRICT XML.

ERROR:
{error}

OUTPUT:
{raw}

Return ONLY corrected XML.
"""
    return repair_llm.invoke(prompt).content

def parse_with_fix(raw: str):
    raw = pre_clean(raw)
    try:
        return base_parser.parse(raw)
    except Exception as e:
        try:
            fixed = pre_clean(repair_xml(raw, str(e)))
            return base_parser.parse(fixed)
        except Exception as e2:
            return {
                "correctness": "incorrect",
                "reasoning_score": 0,
                "grounding_score": 0,
                "final_answer": "UNKNOWN",
                "confidence": 0.0,
                "parser_error": str(e2),
            }

# ============================================================
# MODEL OUTPUT SELECTION
# ============================================================
def select_judge_input(item: dict, model_name: str) -> str | None:
    """
    LLAVA → raw_<model>
    Others → answer_<model> + rationale_<model>
    """
    lname = model_name.lower()

    if "llava" in lname:
        return item.get(f"raw_{model_name}")

    ans = item.get(f"answer_{model_name}")
    rat = item.get(f"rationale_{model_name}")

    if ans and rat:
        return f"ANSWER:\n{ans}\n\nRATIONALE:\n{rat}"
    if ans:
        return f"ANSWER:\n{ans}"
    if rat:
        return f"RATIONALE:\n{rat}"
    return None

# ============================================================
# JUDGE ONE OUTPUT
# ============================================================
def judge_once(judge_llm, question: str, table: str, model_answer: str):
    prompt = f"""
QUESTION:
{question}

TABLE:
{table}

MODEL ANSWER:
{model_answer}
"""
    raw = judge_llm.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    ).content

    parsed = parse_with_fix(raw)
    parsed["raw_judge_output"] = raw
    return parsed

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--judge_model", default="gpt-5")
    args = parser.parse_args()

    judge_llm = ChatOpenAI(
        model=args.judge_model,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Judging"):
            item = json.loads(line)

            question = item.get("question")
            table = item.get("context") or item.get("meta", {}).get("flattened_table_raw")

            for key in list(item.keys()):
                if not key.startswith("answer_") and not key.startswith("raw_"):
                    continue

                model_name = key.replace("answer_", "").replace("raw_", "")
                judge_input = select_judge_input(item, model_name)

                if not judge_input:
                    continue

                result = judge_once(
                    judge_llm,
                    question,
                    table,
                    judge_input
                )

                # ---- gated scoring (industry standard) ----
                gated_score = 0
                if result["correctness"] == "correct":
                    gated_score = result["reasoning_score"] + result["grounding_score"]

                item[f"judge_{model_name}"] = {
                    "correctness": result["correctness"],
                    "reasoning_score": result["reasoning_score"],
                    "grounding_score": result["grounding_score"],
                    "final_answer": result["final_answer"],
                    "confidence": result["confidence"],
                    "gated_score": gated_score,
                }

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Multi-axis judging complete.")

if __name__ == "__main__":
    main()
