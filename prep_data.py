# ================================================================
# PREPARE_DATA.PY
# ================================================================

# ================================================================
# SECTION 0: CACHE ENV (must be before ALL HF imports)
# ================================================================
import os
import io
import json
import random
import hashlib
import datetime
from typing import Optional

HF_CACHE_DIR        = "./hf_cache"
DATASET_PERSIST_DIR = os.path.join(HF_CACHE_DIR, "mixed_datasets")

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"]               = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"]     = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"]    = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HF_HUB_CACHE"]         = os.path.join(HF_CACHE_DIR, "hub")

for d in [
    os.path.join(HF_CACHE_DIR, "datasets"),
    os.path.join(HF_CACHE_DIR, "transformers"),
    os.path.join(HF_CACHE_DIR, "hub"),
    DATASET_PERSIST_DIR,
]:
    os.makedirs(d, exist_ok=True)

print(f"[CACHE] HF root        → {os.path.abspath(HF_CACHE_DIR)}")
print(f"[CACHE] Mixed datasets → {os.path.abspath(DATASET_PERSIST_DIR)}")

# ── HF imports ───────────────────────────────────────────────────
import datasets
from PIL import Image
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    Features,
    Value,
)

# ================================================================
# SECTION 1: CONFIG
# ================================================================
CONFIG = {
    # ── Your domain data ─────────────────────────────────────────
    # JSONL file field already contains "Extracted/..." prefix
    # so base path is "." not "./Extracted" to avoid duplication:
    #   file     = "Extracted/patent/image.png"
    #   base_dir = "."
    #   result   = "./Extracted/patent/image.png"  ✓
    "domain_data_path"       : "aggregate_converted.jsonl",
    "domain_image_base_path" : ".",               # ← NOT "./Extracted"
    "fallback_image_size"    : (224, 224),

    # ── Mixing ───────────────────────────────────────────────────
    # 1.0 = 100% your data  |  0.0 = 100% FineVisionMax
    "domain_percentage"      : 0.70,
    "total_samples"          : 200,

    # ── FineVisionMax ─────────────────────────────────────────────
    "finevision_dataset"     : "HuggingFaceM4/FineVisionMax",
    "finevision_split"       : "train",

    # ── Reproducibility ──────────────────────────────────────────
    "seed"                   : 42,

    # ── Cache ────────────────────────────────────────────────────
    "use_dataset_cache"      : True,
}

# ================================================================
# DATASET FEATURES
#
# Use datasets.Image() directly — store PIL objects in sample dicts
# HF automatically encodes PIL → PNG bytes on save_to_disk()
# HF automatically decodes PNG bytes → PIL on load_from_disk()
#
# Exactly like HF ImageFolder docs:
#   dataset["train"][0]["image"] → <PIL.Image RGB 800x600>
# ================================================================
DATASET_FEATURES = Features({
    "image"        : datasets.Image(),  # PIL in → PIL out
    "conversations": Value("string"),   # JSON string of chat turns
    "source"       : Value("string"),   # "domain" or "finevision"
})

# ================================================================
# SECTION 2: CACHE HELPERS
# ================================================================
def get_cache_key() -> str:
    """
    MD5 fingerprint of config + domain file stats.
    Any change → new key → auto rebuild.
    """
    path        = CONFIG["domain_data_path"]
    fingerprint = "missing"
    if os.path.exists(path):
        s           = os.stat(path)
        fingerprint = f"{s.st_size}_{s.st_mtime}"
    raw = (
        f"dp={CONFIG['domain_percentage']}_"
        f"ts={CONFIG['total_samples']}_"
        f"seed={CONFIG['seed']}_"
        f"path={path}_"
        f"fp={fingerprint}"
    )
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def get_paths(cache_key: str) -> dict:
    base = os.path.join(DATASET_PERSIST_DIR, cache_key + "_prepared")
    return {
        "prepared" : base,
        "sentinel" : base + ".ready",
        "meta"     : os.path.join(base, "cache_meta.json"),
    }


def write_sentinel(path: str) -> None:
    """Proof that save_to_disk() completed. train.py polls this."""
    with open(path, "w") as f:
        f.write(datetime.datetime.now().isoformat())
    print(f"  [SENTINEL] Written → {path}")


def list_existing_caches() -> None:
    entries = [
        e for e in os.listdir(DATASET_PERSIST_DIR)
        if not e.endswith(".ready")
        and os.path.isdir(os.path.join(DATASET_PERSIST_DIR, e))
    ]
    if not entries:
        print("  No existing caches found.")
        return
    print(f"\n  Existing caches:")
    for entry in entries:
        meta_path = os.path.join(
            DATASET_PERSIST_DIR, entry, "cache_meta.json"
        )
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                m = json.load(f)
            pct = m.get("domain_percentage", 0)
            print(f"    ┌─ key     : {entry}")
            print(f"    │  mix     : {pct*100:.1f}% domain / "
                  f"{(1-pct)*100:.1f}% FineVisionMax")
            print(f"    │  samples : {m.get('total_samples'):,}")
            print(f"    │  created : {m.get('created_at')}")
            print(f"    └──────────────────────────────────")

# ================================================================
# SECTION 3: IMAGE LOADER
# ================================================================
def load_image(file_path: str, base_dir: str) -> Image.Image:
    """
    Load image from disk → PIL RGB.

    Tries multiple path candidates in order to handle:
      1. file path used as-is           (absolute paths)
      2. base_dir + file_path           (normal case)
      3. just ./file_path               (if base_dir already in file)

    This covers the duplicate prefix problem where:
      file_path = "Extracted/patent/image.png"
      base_dir  = "./Extracted"
      naive join = "./Extracted/Extracted/patent/image.png"  ✗

    With candidate tries:
      candidate 1: "Extracted/patent/image.png"              → may work
      candidate 2: "./Extracted/patent/image.png"  (base=".")→ works ✓
      candidate 3: "./Extracted/patent/image.png"            → works ✓

    Falls back to grey image if nothing found.
    """
    safe = (file_path or "").strip()

    if safe:
        candidates = [
            safe,                           # as-is
            os.path.join(base_dir, safe),   # base + path
            os.path.join(".", safe),        # ./ + path
        ]

        # Deduplicate while preserving order
        seen       = set()
        candidates = [
            c for c in candidates
            if not (c in seen or seen.add(c))
        ]

        for candidate in candidates:
            try:
                if os.path.exists(candidate):
                    img = Image.open(candidate).convert("RGB")
                    return img
            except Exception as e:
                print(f"  [WARN] Cannot open '{candidate}': {e}")

        # Nothing worked
        print(f"  [WARN] Image not found: '{safe}' "
              f"(tried {len(candidates)} paths)")

    # Grey fallback
    w, h = CONFIG["fallback_image_size"]
    return Image.new("RGB", (w, h), color=(200, 200, 200))

# ================================================================
# SECTION 4: FORMATTERS
#
# Both return the same shape:
# {
#   "image"        : <PIL.Image>   ← PIL object, HF handles encoding
#   "conversations": "..."         ← JSON string of chat turns
#   "source"       : "..."         ← "domain" or "finevision"
# }
# ================================================================
def format_domain_sample(sample: dict) -> Optional[dict]:
    """
    Your JSONL record → unified HF dataset sample.

    JSONL fields:
      question      → user turn body          (required)
      answer        → assistant turn          (required)
      context       → user turn prefix        (optional)
      rationale     → assistant turn suffix   (optional)
      artefact_type → question tag            (optional)
      file          → image path              (optional)

    Input example:
      {
        "question"     : "What is the proteinuria % at 0.4 mg/kg?",
        "answer"       : "17%",
        "rationale"    : "From Table 9, identify...",
        "context"      : "Table 9 presents data...",
        "file"         : "Extracted/US20150315128A1/table_13.png",
        "artefact_type": "table"
      }

    Output example:
      {
        "image": <PIL.Image RGB 800x600>,
        "conversations": '[
          {"role":"user",
           "content":"Context:\\nTable 9 presents...\\n\\nQuestion [table]:\\nWhat is..."},
          {"role":"assistant",
           "content":"17%\\n\\nReasoning:\\nFrom Table 9..."}
        ]',
        "source": "domain"
      }
    """
    try:
        # ── User turn ─────────────────────────────────────────────
        context_block = (
            f"Context:\n{sample['context']}\n\n"
            if sample.get("context") else ""
        )
        tag       = sample.get("artefact_type") or "GENERAL"
        user_text = (
            f"{context_block}"
            f"Question [{tag}]:\n"
            f"{sample['question']}"
        )

        # ── Assistant turn ────────────────────────────────────────
        rationale_block = (
            f"\n\nReasoning:\n{sample['rationale']}"
            if sample.get("rationale") else ""
        )
        assistant_text = f"{sample['answer']}{rationale_block}"

        # ── Image ─────────────────────────────────────────────────
        pil_image = load_image(
            sample.get("file") or "",
            CONFIG["domain_image_base_path"],
        )

        return {
            "image"        : pil_image,
            "conversations": json.dumps([
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]),
            "source": "domain",
        }

    except KeyError as e:
        print(f"  [WARN] Domain missing field {e} — skipping")
        return None
    except Exception as e:
        print(f"  [WARN] Domain format error: {e} — skipping")
        return None


def format_finevision_sample(sample: dict) -> Optional[dict]:
    """
    FineVisionMax record → unified HF dataset sample.

    M4 fields:
      images → list of PIL or single PIL (we use images[0])
      texts  → list of {user: "...", assistant: "..."}
               can be multi-turn

    Input example:
      {
        "images": [<PIL.Image>],
        "texts" : [
          {"user": "What is the primary key?",
           "assistant": "The primary key is EquipmentID..."},
          {"user": "What is the relationship?",
           "assistant": "The relationship is Maintained By..."}
        ]
      }

    Output example:
      {
        "image": <PIL.Image RGB>,
        "conversations": '[
          {"role":"user",      "content":"What is the primary key?"},
          {"role":"assistant", "content":"The primary key is EquipmentID..."},
          {"role":"user",      "content":"What is the relationship?"},
          {"role":"assistant", "content":"The relationship is Maintained By..."}
        ]',
        "source": "finevision"
      }
    """
    try:
        # ── Image ─────────────────────────────────────────────────
        images = sample.get("images")

        if isinstance(images, list) and images:
            pil_image = images[0]
        elif isinstance(images, Image.Image):
            pil_image = images
        else:
            pil_image = None

        if isinstance(pil_image, Image.Image):
            pil_image = pil_image.convert("RGB")
        else:
            w, h      = CONFIG["fallback_image_size"]
            pil_image = Image.new("RGB", (w, h), color=(200, 200, 200))

        # ── Conversation turns ────────────────────────────────────
        texts         = sample.get("texts") or []
        conversations = []

        for turn in texts:
            if not isinstance(turn, dict):
                continue
            if turn.get("user"):
                conversations.append({
                    "role"   : "user",
                    "content": turn["user"],
                })
            if turn.get("assistant"):
                conversations.append({
                    "role"   : "assistant",
                    "content": turn["assistant"],
                })

        if not conversations:
            return None

        return {
            "image"        : pil_image,
            "conversations": json.dumps(conversations),
            "source"       : "finevision",
        }

    except Exception as e:
        print(f"  [WARN] FineVisionMax format error: {e} — skipping")
        return None

# ================================================================
# SECTION 5: LOADERS
# ================================================================
def load_domain_samples(n: int) -> list:
    """
    Load n samples from JSONL.
    Oversamples if file has fewer than n records.
    """
    if n == 0:
        print("\n[1/2] Skipping domain (percentage = 0)")
        return []

    print(f"\n[1/2] Loading domain data ...")
    print(f"      File : {CONFIG['domain_data_path']}")
    print(f"      Need : {n:,} samples")

    raw = []
    try:
        with open(CONFIG["domain_data_path"], "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Bad JSON line {ln}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"JSONL not found: {CONFIG['domain_data_path']}"
        )

    print(f"      Found: {len(raw):,} records in file")

    if len(raw) >= n:
        selected = random.sample(raw, n)
    else:
        print(f"  [INFO] Oversampling: {len(raw):,} → {n:,}")
        reps     = (n // len(raw)) + 1
        selected = random.sample(raw * reps, n)

    formatted = [format_domain_sample(s) for s in selected]
    formatted = [s for s in formatted if s is not None]

    skipped = n - len(formatted)
    if skipped:
        print(f"  [INFO] Skipped {skipped} bad samples")

    print(f"  ✓ {len(formatted):,} domain samples ready")
    return formatted


def load_finevision_samples(n: int) -> list:
    """
    Stream n samples from FineVisionMax.
    Never downloads the full dataset — stops early.
    """
    if n == 0:
        print("\n[2/2] Skipping FineVisionMax (percentage = 1.0)")
        return []

    print(f"\n[2/2] Streaming FineVisionMax ...")
    print(f"      Dataset : {CONFIG['finevision_dataset']}")
    print(f"      Need    : {n:,} samples")
    print(f"      Mode    : streaming (no full download)")

    ds = load_dataset(
        CONFIG["finevision_dataset"],
        split    = CONFIG["finevision_split"],
        streaming= True,
    )

    formatted = []
    skipped   = 0
    streamed  = 0

    for raw in ds:
        streamed += 1
        sample    = format_finevision_sample(raw)

        if sample:
            formatted.append(sample)
        else:
            skipped += 1

        if len(formatted) > 0 and len(formatted) % 500 == 0:
            print(f"  streamed {streamed:,} | "
                  f"kept {len(formatted):,}/{n:,} | "
                  f"skipped {skipped:,}")

        if len(formatted) >= n:
            break

    print(f"  ✓ {len(formatted):,} FineVisionMax samples ready")
    print(f"    streamed {streamed:,} total | skipped {skipped:,}")
    return formatted

# ================================================================
# SECTION 6: UNSLOTH MESSAGE FORMATTER
# (used in train.py — NOT called here)
# ================================================================
def to_unsloth_messages(sample: dict) -> dict:
    """
    Call this in train.py AFTER load_from_disk():

        from prepare_data import to_unsloth_messages
        dataset = load_from_disk(path)
        dataset = dataset.map(to_unsloth_messages, num_proc=1)

    When HF loads the dataset from disk it automatically
    decodes stored PNG bytes back to PIL:
        sample["image"] → <PIL.Image RGB 800x600>

    We then attach that PIL to the first user turn for Unsloth.

    Output format:
      Single-turn (domain):
        messages = [
          {role:"user", content:[
            {"type":"image", "image":<PIL>},
            {"type":"text",  "text":"Context:\\n...\\nQuestion [table]:\\n..."}
          ]},
          {role:"assistant", content:"17%\\n\\nReasoning:\\n..."}
        ]

      Multi-turn (finevision):
        messages = [
          {role:"user", content:[          ← image only on first turn
            {"type":"image", "image":<PIL>},
            {"type":"text",  "text":"What is the primary key?"}
          ]},
          {role:"assistant", content:"The primary key is EquipmentID..."},
          {role:"user",      content:"What is the relationship?"},
          {role:"assistant", content:"The relationship is Maintained By..."}
        ]
    """
    pil_image     = sample["image"]               # PIL decoded by HF
    conversations = json.loads(sample["conversations"])

    messages       = []
    image_attached = False

    for turn in conversations:
        role    = turn["role"]
        content = turn["content"]

        if role == "user" and not image_attached:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text" : content},
                ],
            })
            image_attached = True
        else:
            messages.append({
                "role"   : role,
                "content": content,
            })

    return {"messages": messages}

# ================================================================
# SECTION 7: MAIN
# ================================================================
def prepare_dataset(
    domain_percentage : float = CONFIG["domain_percentage"],
    total_samples     : int   = CONFIG["total_samples"],
    seed              : int   = CONFIG["seed"],
) -> str:
    """
    Full pipeline:
      1. Check cache → return instantly if hit
      2. Load + format domain JSONL samples
      3. Stream + format FineVisionMax samples
      4. Merge + shuffle
      5. Dataset.from_list() with datasets.Image() features
      6. Sanity read-back to verify image decodes correctly
      7. save_to_disk() + write sentinel

    Returns path to saved dataset for train.py.
    """
    random.seed(seed)
    n_domain     = round(total_samples * domain_percentage)
    n_finevision = total_samples - n_domain

    cache_key = get_cache_key()
    paths     = get_paths(cache_key)

    print(f"\n{'='*60}")
    print(f"  DATASET PREPARATION")
    print(f"{'='*60}")
    print(f"  Total    : {total_samples:,}")
    print(f"  Domain   : {n_domain:,}  ({domain_percentage*100:.0f}%)")
    print(f"  FineVis  : {n_finevision:,}  ({(1-domain_percentage)*100:.0f}%)")
    print(f"  Seed     : {seed}")
    print(f"  Cache key: {cache_key}")
    print(f"{'='*60}")
    list_existing_caches()

    # ── Cache hit ─────────────────────────────────────────────────
    if CONFIG["use_dataset_cache"] and os.path.exists(paths["prepared"]):
        print(f"\n✅ CACHE HIT")
        print(f"   Path : {paths['prepared']}")
        if os.path.exists(paths["meta"]):
            with open(paths["meta"]) as f:
                m = json.load(f)
            print(f"   Built: {m.get('created_at')}")
            print(f"   Size : {m.get('total_samples'):,} samples")
            print(f"   Mix  : {m.get('domain_percentage')*100:.0f}% domain / "
                  f"{(1 - m.get('domain_percentage'))*100:.0f}% FineVisionMax")
        if not os.path.exists(paths["sentinel"]):
            write_sentinel(paths["sentinel"])
        print(f"\n  → accelerate launch train.py "
              f"--prepared_path {paths['prepared']}")
        return paths["prepared"]

    # ── Build ─────────────────────────────────────────────────────
    print(f"\n  CACHE MISS — building from scratch ...")

    # Step 1: Load both sources
    domain_samples     = load_domain_samples(n_domain)
    finevision_samples = load_finevision_samples(n_finevision)

    # Step 2: Merge + shuffle
    all_samples = domain_samples + finevision_samples
    random.shuffle(all_samples)

    n_dom = sum(1 for s in all_samples if s["source"] == "domain")
    n_fv  = sum(1 for s in all_samples if s["source"] == "finevision")

    print(f"\n  Merged dataset:")
    print(f"    Total        : {len(all_samples):,}")
    print(f"    Domain       : {n_dom:,}")
    print(f"    FineVisionMax: {n_fv:,}")

    # Step 3: Verify first 5 samples have PIL images
    print(f"\n  Verifying samples ...")
    for i, s in enumerate(all_samples[:5]):
        assert isinstance(s["image"], Image.Image), \
            (f"Sample {i} ({s['source']}): "
             f"image must be PIL got {type(s['image'])}")
        assert isinstance(s["conversations"], str), \
            f"Sample {i}: conversations must be str"
        assert isinstance(s["source"], str), \
            f"Sample {i}: source must be str"
    print(f"  ✓ Verified — images are PIL objects")
    print(f"    Sample 0 image : {all_samples[0]['image']}")
    print(f"    Sample 0 source: {all_samples[0]['source']}")
    print(f"    Sample 0 conv  : "
          f"{all_samples[0]['conversations'][:80]}...")

    # Step 4: Create HF Dataset
    # datasets.Image() in Features:
    #   • Accepts PIL objects in from_list()
    #   • Encodes to PNG bytes on save_to_disk()
    #   • Decodes back to PIL on load_from_disk()
    #   • Same behaviour as HF ImageFolder
    print(f"\n  Creating HF Dataset ...")
    dataset = Dataset.from_list(
        all_samples,
        features=DATASET_FEATURES,
    )
    print(f"  ✓ Dataset created")
    print(f"    Features : {dataset.features}")
    print(f"    Size     : {len(dataset):,} samples")

    # Step 5: Sanity read-back
    sample_back = dataset[0]
    print(f"\n  Sanity read-back sample 0:")
    print(f"    image        : {sample_back['image']}")
    print(f"    source       : {sample_back['source']}")
    print(f"    conversations: {sample_back['conversations'][:80]}...")

    assert isinstance(sample_back["image"], Image.Image), \
        "Read-back failed: image is not PIL"
    print(f"  ✓ Read-back OK — image decoded as PIL")

    # Step 6: Save to disk
    if CONFIG["use_dataset_cache"]:
        print(f"\n  Saving → {paths['prepared']} ...")
        dataset.save_to_disk(paths["prepared"])

        meta = {
            "cache_key"         : cache_key,
            "domain_percentage" : domain_percentage,
            "total_samples"     : len(all_samples),
            "n_domain"          : n_dom,
            "n_finevision"      : n_fv,
            "seed"              : seed,
            "domain_data_path"  : CONFIG["domain_data_path"],
            "finevision_dataset": CONFIG["finevision_dataset"],
            "created_at"        : datetime.datetime.now().isoformat(),
        }
        with open(paths["meta"], "w") as f:
            json.dump(meta, f, indent=2)

        # Sentinel written LAST — proves save completed fully
        write_sentinel(paths["sentinel"])

        print(f"\n{'='*60}")
        print(f"  ✅ DONE")
        print(f"  Saved    → {paths['prepared']}")
        print(f"  Sentinel → {paths['sentinel']}")
        print(f"\n  Now run training:")
        print(f"  → accelerate launch train.py "
              f"--prepared_path {paths['prepared']}")
        print(f"{'='*60}")
    else:
        print(f"  [INFO] Cache disabled — not saving to disk")

    return paths["prepared"]


# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    prepare_dataset()
