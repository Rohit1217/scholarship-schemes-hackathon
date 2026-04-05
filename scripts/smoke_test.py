"""End-to-end smoke test: retriever + profile query + mocked LLM + Gradio app build.

Runs entirely locally — no Databricks or Sarvam credentials needed.
The LLM call is intercepted and returns a canned response.

Usage:
    # First build the test index:
    conda run -n llm python scripts/build_test_index.py

    # Then run this smoke test:
    conda run -n llm python scripts/smoke_test.py

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))   # makes `app.main` importable as a package

TEST_INDEX_DIR = "/tmp/scholarship_test"
os.environ["FAISS_INDEX_PATH"] = f"{TEST_INDEX_DIR}/faiss.index"
os.environ["FAISS_META_PATH"] = f"{TEST_INDEX_DIR}/metadata.parquet"

# Disable Sarvam (no API key needed)
os.environ.pop("SARVAM_API_KEY", None)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"{PASS}  {name}")
    else:
        print(f"{FAIL}  {name}" + (f"\n       {detail}" if detail else ""))
        failures.append(name)


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# 1. Pre-flight: index files exist
# ---------------------------------------------------------------------------
section("1 / Pre-flight")

index_path = Path(os.environ["FAISS_INDEX_PATH"])
meta_path = Path(os.environ["FAISS_META_PATH"])
check("faiss.index exists", index_path.exists(),
      f"Run: conda run -n llm python scripts/build_test_index.py")
check("metadata.parquet exists", meta_path.exists())

if not index_path.exists() or not meta_path.exists():
    print("\n  Cannot continue — run build_test_index.py first.\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Module imports
# ---------------------------------------------------------------------------
section("2 / Module imports")

try:
    from scholarship.llm_client import (
        SYSTEM_PROMPT, chat_completions, extract_assistant_text, rag_user_message,
    )
    check("scholarship.llm_client", True)
except Exception as e:
    check("scholarship.llm_client", False, str(e))

try:
    from scholarship.sarvam_client import is_configured, strip_markdown_for_tts
    check("scholarship.sarvam_client", True)
except Exception as e:
    check("scholarship.sarvam_client", False, str(e))

try:
    from scholarship.retriever import FAISSRetriever, get_retriever
    check("scholarship.retriever", True)
except Exception as e:
    check("scholarship.retriever", False, str(e))

# ---------------------------------------------------------------------------
# 3. profile_to_query()
# ---------------------------------------------------------------------------
section("3 / profile_to_query()")

# Import after path is set up
sys.path.insert(0, str(ROOT / "app"))
from main import profile_to_query  # noqa: E402

cases = [
    dict(state="Maharashtra", category="SC", income=80000, gender="Female",
         age=17, education="Class 12", disability=False, minority=False,
         expect=["Maharashtra", "SC", "80,000", "Class 12", "Female"]),
    dict(state="Karnataka", category="OBC", income=150000, gender="Male",
         age=20, education="Undergraduate", disability=True, minority=False,
         expect=["Karnataka", "OBC", "150,000", "Undergraduate", "disability"]),
    dict(state="Delhi", category="General", income=300000, gender="Other",
         age=23, education="Postgraduate", disability=False, minority=True,
         expect=["Delhi", "General", "minority"]),
]

for i, c in enumerate(cases, 1):
    q = profile_to_query(
        c["state"], c["category"], c["income"], c["gender"],
        c["age"], c["education"], c.get("disability", False), c.get("minority", False),
    )
    check(f"case {i}: non-empty string", isinstance(q, str) and len(q) > 20)
    check(f"case {i}: ends with '?'", q.strip().endswith("?"))
    for kw in c["expect"]:
        check(f"case {i}: contains '{kw}'", kw in q, f"query was: {q[:120]}")

# Disability/minority flags
q_with = profile_to_query("Bihar", "ST", 50000, "Female", 15, "Class 8", True, True)
q_without = profile_to_query("Bihar", "ST", 50000, "Female", 15, "Class 8", False, False)
check("disability flag included when True", "disability" in q_with.lower())
check("minority flag included when True", "minority" in q_with.lower())
check("disability flag absent when False", "disability" not in q_without.lower())
check("minority flag absent when False", "minority" not in q_without.lower())

# ---------------------------------------------------------------------------
# 4. FAISSRetriever
# ---------------------------------------------------------------------------
section("4 / FAISSRetriever")

retriever = FAISSRetriever(
    index_path=str(index_path),
    meta_path=str(meta_path),
)

# Test a few queries
test_queries = [
    ("SC student Maharashtra Class 12 income 80000", 5),
    ("OBC girl disability scholarship undergraduate", 5),
    ("minority community Muslim income below 1 lakh Class 10", 5),
    ("EWS general category postgraduate scholarship", 3),
    ("PhD fellowship for SC girls", 3),
]

for query, k in test_queries:
    results = retriever.search(query, k=k)
    check(f"search returns DataFrame: '{query[:40]}…'", hasattr(results, "iterrows"))
    check(f"  → ≤{k} rows returned", len(results) <= k, f"got {len(results)}")
    check(f"  → has required columns",
          all(c in results.columns for c in ("scheme_id", "scheme_name", "text", "score")))
    check(f"  → scores in [-1, 1]",
          (results["score"] >= -1.01).all() and (results["score"] <= 1.01).all(),
          str(results["score"].tolist()))

# Verify top result relevance for a targeted query
sc_results = retriever.search("SC student scholarship Class 10 Maharashtra income 80000", k=5)
top_ids = sc_results["scheme_id"].tolist()
print(f"\n  Top matches for SC/Class10 query: {top_ids[:3]}")
has_sc_scheme = any("SC" in sid or "sc" in sid.lower() for sid in top_ids)
check("  top results contain an SC scheme", has_sc_scheme,
      f"top scheme_ids were: {top_ids}")

# ---------------------------------------------------------------------------
# 5. rag_user_message()
# ---------------------------------------------------------------------------
section("5 / rag_user_message()")

chunks = [
    "Scheme: NSP Pre-Matric for SC. Income limit: below ₹2,00,000.",
    "Scheme: PM YASASVI for OBC. Income limit: below ₹2,50,000.",
]
msg = rag_user_message(chunks, "Which scholarship am I eligible for?")
check("returns non-empty string", isinstance(msg, str) and len(msg) > 0)
check("contains Context block", "Context:" in msg)
check("contains Question block", "Question:" in msg)
check("chunks embedded in message", chunks[0][:30] in msg)

# ---------------------------------------------------------------------------
# 6. extract_assistant_text()
# ---------------------------------------------------------------------------
section("6 / extract_assistant_text()")

fake_response = {
    "choices": [{"message": {"content": "  1. NSP Pre-Matric: you qualify because SC category.  "}}]
}
text = extract_assistant_text(fake_response)
check("strips whitespace", text == "1. NSP Pre-Matric: you qualify because SC category.")

bad_response = {"choices": []}
try:
    extract_assistant_text(bad_response)
    check("raises on malformed response", False)
except ValueError:
    check("raises ValueError on malformed response", True)

# ---------------------------------------------------------------------------
# 7. Full RAG flow (mocked LLM)
# ---------------------------------------------------------------------------
section("7 / Full RAG flow (mocked LLM)")

MOCK_LLM_RESPONSE = {
    "choices": [{
        "message": {
            "content": textwrap.dedent("""\
                1. **Pre-Matric Scholarship for SC Students** — You qualify: SC category, \
income ₹80,000 below ₹2 lakh limit, Class 12 education. \
Award: ₹350/month. Deadline: October 31.

                2. **Post-Matric Scholarship for SC and ST Students** — You qualify: SC category, \
income within ₹2.5 lakh limit. Covers Class 12. \
Award: ₹530–₹1,200/month. Deadline: November 30.
            """).strip()
        }
    }]
}


def mock_chat_completions(messages, **kwargs):
    return MOCK_LLM_RESPONSE


with patch("scholarship.llm_client.chat_completions", side_effect=mock_chat_completions):
    # Re-import app.main's reference to _rag_answer_english
    # (it imports chat_completions at module level, so patch at the source module)
    import scholarship.llm_client as llm_mod
    orig = llm_mod.chat_completions
    llm_mod.chat_completions = mock_chat_completions

    try:
        # Need to import _rag_answer_english from app after patching
        import importlib
        import app.main as app_main
        # Patch the reference in app.main too
        app_main.chat_completions = mock_chat_completions

        query_en = profile_to_query(
            "Maharashtra", "SC", 80000, "Female", 17, "Class 12", False, False
        )
        assistant_en, cites = app_main._rag_answer_english(query_en)

        check("LLM response is non-empty string", isinstance(assistant_en, str) and len(assistant_en) > 0)
        check("Citations block non-empty", isinstance(cites, str) and len(cites) > 0)
        check("Response contains scheme name", "Pre-Matric" in assistant_en or "Scholarship" in assistant_en)

        # build_reply_markdown — English mode (no Sarvam key set)
        reply = app_main.build_reply_markdown(assistant_en, cites, "en")
        check("Reply markdown non-empty", len(reply) > 0)
        check("Reply contains disclaimer", "verify" in reply.lower() or "guidance" in reply.lower())
        check("Reply contains sources block", "Matched schemes" in reply)

        # Non-English mode without Sarvam — should fall back to English silently
        reply_hi = app_main.build_reply_markdown(assistant_en, cites, "hi")
        check("Non-English without Sarvam key returns English fallback", len(reply_hi) > 0)

    finally:
        llm_mod.chat_completions = orig

# ---------------------------------------------------------------------------
# 8. strip_markdown_for_tts()
# ---------------------------------------------------------------------------
section("8 / strip_markdown_for_tts()")

md = "**Bold** and *italic* with `code` and a [link](https://example.com).\n## Header"
plain = strip_markdown_for_tts(md)
check("removes bold markers", "**" not in plain)
check("removes italic markers", not ("*Bold*" in plain or "*italic*" in plain))
check("removes markdown links", "https://example.com" not in plain)
check("result is non-empty", len(plain.strip()) > 0)

# ---------------------------------------------------------------------------
# 9. SYSTEM_PROMPT sanity
# ---------------------------------------------------------------------------
section("9 / SYSTEM_PROMPT")

check("contains 'scholarship'", "scholarship" in SYSTEM_PROMPT.lower())
check("contains 'eligible'", "eligible" in SYSTEM_PROMPT.lower())
check("contains 'numbered list'", "numbered list" in SYSTEM_PROMPT.lower())
check("does not mention 'legal'", "legal" not in SYSTEM_PROMPT.lower())

# ---------------------------------------------------------------------------
# 10. Gradio app builds without error
# ---------------------------------------------------------------------------
section("10 / Gradio app build")

try:
    import app.main as app_main_fresh
    demo = app_main_fresh.build_app()
    check("build_app() returns gr.Blocks", str(type(demo).__name__) == "Blocks")
    # Check key components exist by inspecting the app's blocks
    check("demo has queue method", callable(getattr(demo, "queue", None)))
    check("demo has launch method", callable(getattr(demo, "launch", None)))
    print(f"  App title: {demo.title}")
except Exception as e:
    check("build_app() succeeds", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'═'*55}")
if failures:
    print(f"  \033[91m{len(failures)} test(s) FAILED:\033[0m")
    for f in failures:
        print(f"    • {f}")
    sys.exit(1)
else:
    total = 0
    # count all checks by scanning output above — just report failures=0
    print(f"  \033[92mAll tests PASSED\033[0m")
    sys.exit(0)
