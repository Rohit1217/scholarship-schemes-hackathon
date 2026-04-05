"""
Convert scheme guideline PDFs to structured CSV using Databricks LLM.

Usage:
    python scripts/pdf_to_schemes.py                  # process all PDFs in data/pdfs/
    python scripts/pdf_to_schemes.py data/pdfs/NMMSSGuidelines.pdf  # single file

Outputs:
    data/schemes.csv   — append/merge new rows

Improved schema (superset of original):
  scheme_id, scheme_name, administering_body, level, state,
  scheme_type, eligible_categories,
  min_income_limit, max_income_limit,
  eligible_gender, age_min, age_max,
  eligible_education_levels, eligible_course_type,
  eligible_disability, eligible_minority,
  award_amount, application_deadline,
  description_text, source_url
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Load .env
for _line in (ROOT / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

from scholarship.llm_client import chat_completions, extract_assistant_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

COLS = [
    "scheme_id",            # UPPER_SNAKE_CASE unique ID
    "scheme_name",          # Full official name
    "administering_body",   # Ministry / Department
    "level",                # Central | State
    "state",                # blank for Central; state name for State schemes
    "scheme_type",          # pre_matric | post_matric | merit | fellowship | other
    "eligible_categories",  # SC | ST | OBC | General | EWS | Minority | All (comma-sep)
    "min_income_limit",     # int INR (0 = no minimum)
    "max_income_limit",     # int INR (0 = no upper limit)
    "eligible_gender",      # All | Male | Female
    "age_min",              # int (0 = no minimum)
    "age_max",              # int (0 = no upper limit)
    "eligible_education_levels",  # comma-sep: Class 8/10/12/Undergraduate/Postgraduate/PhD
    "eligible_course_type", # Any | Technical | Medical | General | Professional
    "eligible_disability",  # All | Yes  (Yes = disability is a criterion)
    "eligible_minority",    # All | Yes
    "award_amount",         # human-readable string
    "application_deadline", # human-readable string
    "description_text",     # 2-3 sentence factual description
    "source_url",           # official URL or PDF filename
]

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

MAX_CHARS = 12000  # LLM context limit for a single scheme doc

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from PDF, trimmed to MAX_CHARS."""
    doc = fitz.open(str(pdf_path))
    pages = []
    total = 0
    for page in doc:
        t = page.get_text("text")
        pages.append(t)
        total += len(t)
        if total >= MAX_CHARS:
            break
    doc.close()
    full = "\n".join(pages)
    # Collapse excessive blank lines
    full = re.sub(r"\n{3,}", "\n\n", full)
    return full[:MAX_CHARS]


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a specialist in Indian government scholarship schemes.
Given the text of an official scheme guideline PDF, extract ALL distinct scholarship
schemes described and return a JSON array. Each element must have EXACTLY these fields:

scheme_id, scheme_name, administering_body, level, state,
scheme_type, eligible_categories, min_income_limit, max_income_limit,
eligible_gender, age_min, age_max, eligible_education_levels, eligible_course_type,
eligible_disability, eligible_minority, award_amount, application_deadline,
description_text, source_url

Field rules (follow strictly):
- scheme_id: UPPER_SNAKE_CASE, max 35 chars, unique
- level: "Central" or "State"
- state: "" for Central; Indian state name for State-level
- scheme_type: "pre_matric" | "post_matric" | "merit" | "fellowship" | "other"
- eligible_categories: comma-separated values from SC/ST/OBC/General/EWS/Minority/All
- min_income_limit: integer INR (use 0 if no minimum)
- max_income_limit: integer INR (use 0 if no upper limit stated)
- eligible_gender: "All" | "Male" | "Female"
- age_min, age_max: integers (use 0 if not specified)
- eligible_education_levels: comma-separated from:
    Class 8 / Class 10 / Class 12 / Undergraduate / Postgraduate / PhD
- eligible_course_type: "Any" | "Technical" | "Medical" | "General" | "Professional"
- eligible_disability: "All" or "Yes" (Yes only if disability is eligibility criterion)
- eligible_minority: "All" or "Yes" (Yes only if minority status is criterion)
- award_amount: human-readable string (e.g. "₹12000/year" or "Full tuition fee")
- application_deadline: human-readable (e.g. "October 31 annually" or "")
- description_text: 2-3 factual sentences extracted from the PDF, NO internal quotes
- source_url: "https://scholarships.gov.in" unless a more specific URL is in the text

Output ONLY a valid JSON array. No markdown, no prose, no explanation."""


def extract_schemes_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract scheme data from a single PDF using the LLM."""
    text = extract_pdf_text(pdf_path)
    log.info("Extracted %d chars from %s", len(text), pdf_path.name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"PDF filename: {pdf_path.name}\n\n"
                f"PDF content:\n{text}\n\n"
                "Extract all scholarship schemes as a JSON array."
            ),
        },
    ]

    t0 = time.time()
    resp = chat_completions(messages, max_tokens=3000, temperature=0.05)
    raw = extract_assistant_text(resp)
    log.info("LLM took %.1fs, %d chars returned", time.time() - t0, len(raw))

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.M)
    raw = re.sub(r"^```\s*$", "", raw, flags=re.M).strip()

    schemes = json.loads(raw)
    if isinstance(schemes, dict):
        schemes = [schemes]  # single scheme wrapped in object

    # Normalise all fields
    clean = []
    for s in schemes:
        row = {}
        for col in COLS:
            row[col] = str(s.get(col, "")).strip()
        # Ensure numeric fields are valid integers
        for num_col in ("min_income_limit", "max_income_limit", "age_min", "age_max"):
            try:
                row[num_col] = str(int(float(row[num_col])))
            except (ValueError, TypeError):
                row[num_col] = "0"
        # Default source_url to PDF filename if empty
        if not row["source_url"]:
            row["source_url"] = f"https://scholarships.gov.in (PDF: {pdf_path.name})"
        clean.append(row)
    return clean


# ---------------------------------------------------------------------------
# CSV merge
# ---------------------------------------------------------------------------

OUT_CSV = ROOT / "data" / "schemes.csv"


def load_existing() -> tuple[list[dict], set[str]]:
    if not OUT_CSV.exists():
        return [], set()
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    ids = {r["scheme_id"] for r in rows}
    return rows, ids


def save_all(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved %d rows to %s", len(rows), OUT_CSV)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_pdfs(pdf_paths: list[Path]) -> None:
    existing, existing_ids = load_existing()
    all_rows = list(existing)
    added = 0

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            log.warning("File not found: %s", pdf_path)
            continue

        log.info("\n=== Processing: %s ===", pdf_path.name)
        try:
            schemes = extract_schemes_from_pdf(pdf_path)
        except json.JSONDecodeError as e:
            log.error("JSON parse failed for %s: %s", pdf_path.name, e)
            continue
        except Exception as e:
            log.error("Error processing %s: %s", pdf_path.name, e)
            continue

        for scheme in schemes:
            sid = scheme["scheme_id"]
            if sid in existing_ids:
                log.info("  Skip duplicate: %s", sid)
                continue
            log.info("  + %s: %s", sid, scheme["scheme_name"][:60])
            all_rows.append(scheme)
            existing_ids.add(sid)
            added += 1

        # Save after each PDF so progress isn't lost
        save_all(all_rows)
        time.sleep(1)  # brief pause between LLM calls

    log.info("\nDone. Added %d new schemes. Total: %d", added, len(all_rows))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Specific files passed as arguments
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        # Process all PDFs in data/pdfs/
        pdf_dir = ROOT / "data" / "pdfs"
        if not pdf_dir.exists():
            print(f"No PDFs found. Run:\n  bash scripts/download_scheme_pdfs.sh\nfirst.")
            sys.exit(1)
        paths = sorted(pdf_dir.glob("*.pdf"))
        if not paths:
            print(f"No .pdf files in {pdf_dir}")
            sys.exit(1)

    print(f"Processing {len(paths)} PDF(s)...")
    process_pdfs(paths)
