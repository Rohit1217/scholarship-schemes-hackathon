"""Real end-to-end test: live Databricks LLM + live Sarvam + local FAISS index.

Loads credentials from .env.example (or .env if present).
Overrides FAISS paths to the local dummy index built by build_test_index.py.

Usage:
    conda run -n llm python scripts/e2e_test.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# --- Load .env or .env.example ---
def _load_env_file():
    for name in (".env", ".env.example"):
        p = ROOT / name
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    if k.strip() and k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()
            print(f"  Loaded credentials from {name}")
            return name
    print("  WARNING: no .env or .env.example found")
    return None

_load_env_file()

# Use local dummy index (not UC Volume paths which require Databricks mount)
LOCAL_INDEX = "/tmp/scholarship_test"
os.environ["FAISS_INDEX_PATH"] = f"{LOCAL_INDEX}/faiss.index"
os.environ["FAISS_META_PATH"]  = f"{LOCAL_INDEX}/metadata.parquet"

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SKIP = "\033[93m  SKIP\033[0m"

failures: list[str] = []

def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"{PASS}  {name}")
    else:
        print(f"{FAIL}  {name}" + (f"\n         → {detail}" if detail else ""))
        failures.append(name)

def section(title: str) -> None:
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")

# ---------------------------------------------------------------------------
# 1. Config check
# ---------------------------------------------------------------------------
section("1 / Configuration")

required = {
    "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST", ""),
    "DATABRICKS_TOKEN": os.environ.get("DATABRICKS_TOKEN", ""),
    "LLM_OPENAI_BASE_URL": os.environ.get("LLM_OPENAI_BASE_URL", ""),
    "LLM_MODEL": os.environ.get("LLM_MODEL", ""),
}
optional = {
    "SARVAM_API_KEY": os.environ.get("SARVAM_API_KEY", ""),
}

for k, v in required.items():
    check(f"{k} is set", bool(v), "missing — add to .env")

sarvam_available = bool(optional["SARVAM_API_KEY"])
_sarvam_icon   = "✓" if sarvam_available else "○"
_sarvam_status = "set" if sarvam_available else "not set — translation tests will be skipped"
print(f"  {_sarvam_icon}  SARVAM_API_KEY {_sarvam_status}")
print(f"  ○  FAISS_INDEX_PATH → {os.environ['FAISS_INDEX_PATH']} (local dummy)")

if any(not v for v in required.values()):
    print("\n  Cannot continue — fill in DATABRICKS_HOST, DATABRICKS_TOKEN, "
          "LLM_OPENAI_BASE_URL, LLM_MODEL in .env\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Databricks SDK auth
# ---------------------------------------------------------------------------
section("2 / Databricks SDK authentication")

try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )
    me = w.current_user.me()
    check(f"workspace auth OK (user: {me.user_name})", True)
except Exception as e:
    check("workspace auth", False, str(e))
    print("\n  Cannot continue — fix Databricks credentials.\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. LLM call — real Databricks AI Gateway
# ---------------------------------------------------------------------------
section("3 / LLM — Databricks AI Gateway")

from scholarship.llm_client import (
    SYSTEM_PROMPT, chat_completions, extract_assistant_text, rag_user_message,
)

test_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "Context:\nScheme: NSP Pre-Matric for SC. Income limit: below ₹2,00,000. "
            "Award: ₹150/month.\n\n"
            "Question: I am an 18-year-old Female SC student from Maharashtra with income ₹80,000. "
            "Am I eligible for the NSP Pre-Matric scholarship?"
        ),
    },
]

print("  Calling LLM... ", end="", flush=True)
t0 = time.time()
try:
    response = chat_completions(test_messages, max_tokens=256, temperature=0.2)
    elapsed = time.time() - t0
    llm_text = extract_assistant_text(response)
    print(f"done ({elapsed:.1f}s)")
    check("LLM returns non-empty text", len(llm_text) > 10)
    check("LLM response is a string", isinstance(llm_text, str))
    print(f"\n  LLM response preview:\n")
    for line in llm_text[:400].splitlines():
        print(f"    {line}")
    if len(llm_text) > 400:
        print(f"    … [{len(llm_text)} chars total]")
except Exception as e:
    print("failed")
    check("LLM call succeeds", False, str(e))

# ---------------------------------------------------------------------------
# 4. FAISS retriever — local dummy index
# ---------------------------------------------------------------------------
section("4 / FAISS retriever (local dummy index)")

from scholarship.retriever import get_retriever

retriever = get_retriever()

test_profiles = [
    ("SC female Maharashtra Class 12 income 80000", "NSP_PRE_MAT_SC"),
    ("OBC student disability undergraduate scholarship", "AICTE_SAKSHAM"),
    ("PhD fellowship for SC girls", "UP_SC_ST_GIRLS_PHD"),
    ("Muslim minority girl income 90000 Class 10", "BEGUM_HAZRAT_MAHAL"),
    ("EWS general undergraduate merit scholarship", "EWS_GENERAL_MERIT"),
]

for query, expected_in_top5 in test_profiles:
    t0 = time.time()
    results = retriever.search(query, k=5)
    elapsed = time.time() - t0
    top_ids = results["scheme_id"].tolist()
    in_top5 = expected_in_top5 in top_ids
    check(
        f"'{query[:45]}…'",
        len(results) > 0,
        f"returned {len(results)} rows in {elapsed:.2f}s"
    )
    print(f"         top-3: {top_ids[:3]}  [{elapsed:.2f}s]")

# ---------------------------------------------------------------------------
# 5. Full RAG answer — real LLM + real retriever
# ---------------------------------------------------------------------------
section("5 / Full RAG flow (real retriever + real LLM)")

from app.main import _rag_answer_english, profile_to_query, build_reply_markdown

profile_cases = [
    dict(state="Maharashtra", category="SC",      income=80000,
         gender="Female", age=17, education="Class 12",
         disability=False, minority=False,
         label="SC/Maharashtra/Class12"),
    dict(state="Karnataka",   category="OBC",     income=150000,
         gender="Male",   age=20, education="Undergraduate",
         disability=True,  minority=False,
         label="OBC/Karnataka/UG/Disability"),
    dict(state="Delhi",       category="General", income=300000,
         gender="Female", age=22, education="Postgraduate",
         disability=False, minority=True,
         label="General/Delhi/PG/Minority"),
]

for case in profile_cases:
    label = case.pop("label")
    query_en = profile_to_query(**case)
    print(f"\n  Profile: {label}")
    print(f"  Query  : {query_en[:100]}…")
    print("  RAG+LLM: ", end="", flush=True)
    t0 = time.time()
    try:
        assistant_en, cites = _rag_answer_english(query_en)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        check(f"{label} — LLM answered",  len(assistant_en) > 20)
        check(f"{label} — citations",     len(cites) > 0)
        print(f"  Answer preview: {assistant_en[:200].replace(chr(10), ' ')}")
        print(f"  Citations     : {cites[:120]}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"failed ({elapsed:.1f}s)")
        check(f"{label} — RAG+LLM", False, str(e))

# ---------------------------------------------------------------------------
# 6. Sarvam translation (skipped if key not set)
# ---------------------------------------------------------------------------
section("6 / Sarvam translation")

if not sarvam_available:
    print(f"{SKIP}  All Sarvam tests — SARVAM_API_KEY not set")
else:
    from scholarship.sarvam_client import translate_text, text_to_speech_wav_bytes, wav_bytes_to_numpy_float32

    translate_cases = [
        ("Hello, how are you?", "en-IN", "hi-IN", "English→Hindi"),
        ("You are eligible for the NSP Pre-Matric scholarship.", "en-IN", "ta-IN", "English→Tamil"),
        ("scholarship for SC students Maharashtra", "en-IN", "mr-IN", "English→Marathi"),
    ]
    for text, src, tgt, label in translate_cases:
        print(f"  Translating ({label})… ", end="", flush=True)
        try:
            result = translate_text(text, source_language_code=src, target_language_code=tgt)
            print("done")
            check(f"translate {label}", len(result) > 0 and result != text,
                  f"got: {result[:80]}")
            print(f"    → {result[:80]}")
        except Exception as e:
            print("failed")
            check(f"translate {label}", False, str(e))

    # TTS smoke test
    print("  TTS (English)… ", end="", flush=True)
    try:
        wav = text_to_speech_wav_bytes(
            "You are eligible for two scholarship schemes.",
            target_language_code="en-IN",
        )
        sr, arr = wav_bytes_to_numpy_float32(wav)
        print("done")
        check("TTS returns audio", len(arr) > 0)
        check("TTS sample rate > 0", sr > 0)
        print(f"    → {sr}Hz, {len(arr)} samples ({len(arr)/sr:.1f}s)")
    except Exception as e:
        print("failed")
        check("TTS", False, str(e))

# ---------------------------------------------------------------------------
# 7. build_reply_markdown — bilingual output
# ---------------------------------------------------------------------------
section("7 / build_reply_markdown() bilingual output")

sample_answer = (
    "1. NSP Pre-Matric Scholarship for SC Students — You qualify: SC category, "
    "income ₹80,000 below ₹2 lakh limit. Award: ₹350/month. Deadline: October 31."
)
sample_cites = "- NSP_PRE_MAT_SC · Pre-Matric Scholarship for SC Students"

reply_en = build_reply_markdown(sample_answer, sample_cites, "en")
check("English reply has answer",    sample_answer[:30] in reply_en)
check("English reply has sources",   "Matched schemes" in reply_en)
check("English reply has disclaimer","verify" in reply_en.lower())

if sarvam_available:
    for lang_code, lang_name in [("hi", "Hindi"), ("ta", "Tamil"), ("kn", "Kannada")]:
        print(f"  Building bilingual reply ({lang_name})… ", end="", flush=True)
        try:
            reply = build_reply_markdown(sample_answer, sample_cites, lang_code)
            print("done")
            check(f"bilingual reply for {lang_name} non-empty", len(reply) > 100)
            check(f"bilingual reply for {lang_name} has English section", "English:" in reply)
            first_line = reply.splitlines()[0] if reply.splitlines() else ""
            print(f"    → first line: {first_line[:80]}")
        except Exception as e:
            print("failed")
            check(f"bilingual reply {lang_name}", False, str(e))
else:
    print(f"{SKIP}  Bilingual reply tests — SARVAM_API_KEY not set")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'═' * 58}")
if failures:
    print(f"  \033[91m{len(failures)} test(s) FAILED:\033[0m")
    for f in failures:
        print(f"    • {f}")
    sys.exit(1)
else:
    print(f"  \033[92mAll tests PASSED\033[0m")
    sys.exit(0)
