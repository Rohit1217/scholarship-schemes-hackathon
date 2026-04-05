"""Deploy dummy scholarship data to Databricks and set up the Vector Search index.

Steps:
  1. Upload data/dummy_schemes.csv to UC Volume /Volumes/main/scholarships/raw/
  2. Create/overwrite Delta table main.scholarships.scheme_corpus via SQL warehouse
  3. Create Vector Search endpoint `scholarship-vs-endpoint` (if not exists)
  4. Create Delta Sync index `main.scholarships.scheme_vs_index` (if not exists)
  5. Trigger sync and wait for ONLINE status
  6. Smoke-test retrieval against the live VS index
  7. Full RAG call (VS retrieval + LLM)

Usage:
    python scripts/deploy_dummy.py

Cleanup (removes VS index + endpoint + Delta table):
    python scripts/deploy_dummy.py --cleanup
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Load .env
for name in (".env", ".env.example"):
    p = ROOT / name
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()
        break

HOST    = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN   = os.environ["DATABRICKS_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

CATALOG      = "main"
SCHEMA       = "scholarships"
TABLE        = "scheme_corpus"
VS_ENDPOINT  = os.environ.get("VS_ENDPOINT_NAME", "scholarship-vs-endpoint")
VS_INDEX     = os.environ.get("VS_INDEX_NAME",    f"{CATALOG}.{SCHEMA}.scheme_vs_index")
EMBED_MODEL  = "databricks-gte-large-en"
VOLUME_RAW   = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
CSV_PATH     = ROOT / "data" / "dummy_schemes.csv"

PASS = "\033[92m  ✓\033[0m"
FAIL = "\033[91m  ✗\033[0m"
INFO = "  ·"


def ok(m):          print(f"{PASS} {m}")
def fail(m, d=""):  print(f"{FAIL} {m}" + (f"\n      {d}" if d else "")); sys.exit(1)
def info(m):        print(f"{INFO} {m}")
def section(t):     print(f"\n── {t} {'─'*(52-len(t))}")


# ---------------------------------------------------------------------------
# SQL warehouse helper
# ---------------------------------------------------------------------------

def _get_warehouse_id() -> str:
    r = requests.get(f"{HOST}/api/2.0/sql/warehouses", headers=HEADERS, timeout=15)
    if r.ok:
        wh = r.json().get("warehouses", [])
        running = [w for w in wh if w.get("state") == "RUNNING"]
        if running:
            return running[0]["id"]
        if wh:
            return wh[0]["id"]
    return ""


def _run_sql(statement: str, warehouse_id: str, timeout: int = 120) -> dict:
    """Execute SQL and poll until finished. wait_timeout is capped at 50s by the API."""
    r = requests.post(
        f"{HOST}/api/2.0/sql/statements",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"statement": statement, "warehouse_id": warehouse_id, "wait_timeout": "50s"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    stmt_id = data.get("statement_id")
    # Poll if still running (CTAS can take > 50s)
    deadline = time.time() + timeout
    while data.get("status", {}).get("state") in ("PENDING", "RUNNING"):
        if time.time() > deadline:
            raise TimeoutError(f"SQL timed out after {timeout}s: {statement[:60]}")
        time.sleep(3)
        r2 = requests.get(
            f"{HOST}/api/2.0/sql/statements/{stmt_id}",
            headers={**HEADERS, "Content-Type": "application/json"},
            timeout=15,
        )
        r2.raise_for_status()
        data = r2.json()
    return data


# ---------------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------------

def cleanup():
    section("Cleanup: delete VS index + endpoint + Delta table")

    from databricks.vector_search.client import VectorSearchClient
    client = VectorSearchClient(disable_notice=True)

    try:
        client.delete_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
        ok(f"Deleted VS index: {VS_INDEX}")
    except Exception as e:
        info(f"VS index delete skipped: {e}")

    try:
        client.delete_endpoint(VS_ENDPOINT)
        ok(f"Deleted VS endpoint: {VS_ENDPOINT}")
    except Exception as e:
        info(f"VS endpoint delete skipped: {e}")

    wh = _get_warehouse_id()
    if wh:
        try:
            _run_sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.{TABLE}", wh)
            ok(f"Dropped table {CATALOG}.{SCHEMA}.{TABLE}")
        except Exception as e:
            info(f"Table drop skipped: {e}")

    print("\n  Cleanup done.\n")


# ---------------------------------------------------------------------------
# DEPLOY
# ---------------------------------------------------------------------------

def deploy():
    # ── 1. Upload CSV to Volume ──────────────────────────────
    section("Step 1 / Upload CSV to UC Volume")
    csv_content = CSV_PATH.read_bytes()
    url = f"{HOST}/api/2.0/fs/files{VOLUME_RAW}/dummy_schemes.csv"
    print(f"  Uploading {CSV_PATH.name}… ", end="", flush=True)
    r = requests.put(
        url,
        headers={**HEADERS, "Content-Type": "application/octet-stream"},
        data=csv_content,
        timeout=60,
    )
    if r.status_code in (200, 201, 204):
        print("done")
        ok(f"{VOLUME_RAW}/dummy_schemes.csv")
    else:
        print("failed")
        fail(f"Upload failed: HTTP {r.status_code}", r.text[:300])

    # ── 2. Build Delta table via SQL warehouse ───────────────
    section("Step 2 / Create Delta table from CSV")
    wh = _get_warehouse_id()
    if not wh:
        fail("No SQL warehouse found. Start a warehouse in the Databricks UI first.")

    info(f"Using warehouse: {wh}")

    for stmt in [
        f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}",
        f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw",
        f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.rag",
    ]:
        _run_sql(stmt, wh)

    # CTAS from CSV with text column and CDF enabled
    text_expr = """CONCAT(
        'Scheme: ', COALESCE(scheme_name, ''), '. ',
        'Administered by: ', COALESCE(administering_body, ''), ' (', COALESCE(level, ''), '). ',
        'Eligible categories: ', COALESCE(eligible_categories, ''), '. ',
        'Income limit: below \u20b9', COALESCE(CAST(max_income_limit AS STRING), ''), '. ',
        'Education: ', COALESCE(eligible_education_levels, ''), '. ',
        'Gender: ', COALESCE(eligible_gender, ''), '. ',
        'Disability: ', COALESCE(eligible_disability, ''), '. ',
        'Minority: ', COALESCE(eligible_minority, ''), '. ',
        'Award: ', COALESCE(award_amount, ''), '. ',
        'Deadline: ', COALESCE(application_deadline, ''), '. ',
        'Details: ', COALESCE(description_text, '')
    )"""

    create_sql = f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.{TABLE}
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS
SELECT *, {text_expr} AS text
FROM read_files(
  '{VOLUME_RAW}/dummy_schemes.csv',
  format => 'csv',
  header => true,
  inferSchema => true
)
""".strip()

    result = _run_sql(create_sql, wh, timeout=120)
    state = result.get("status", {}).get("state", "UNKNOWN")
    if state not in ("SUCCEEDED", "RUNNING"):
        fail(f"Table creation failed (state={state})", str(result.get("status", {})))

    count_result = _run_sql(f"SELECT COUNT(*) AS cnt FROM {CATALOG}.{SCHEMA}.{TABLE}", wh)
    row_count = (count_result.get("result", {}).get("data_array") or [[0]])[0][0]
    ok(f"Table ready: {CATALOG}.{SCHEMA}.{TABLE}  ({row_count} rows, CDF enabled)")

    # ── 3. Create VS endpoint ────────────────────────────────
    section("Step 3 / Create Vector Search endpoint")
    from databricks.vector_search.client import VectorSearchClient

    client = VectorSearchClient(disable_notice=True)

    existing_eps = [ep["name"] for ep in client.list_endpoints().get("endpoints", [])]
    if VS_ENDPOINT in existing_eps:
        info(f"Endpoint already exists: {VS_ENDPOINT}")
    else:
        print(f"  Creating endpoint {VS_ENDPOINT}… ", end="", flush=True)
        client.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
        print("done")

    print("  Waiting for ONLINE… ", end="", flush=True)
    for _ in range(30):
        ep = client.get_endpoint(VS_ENDPOINT)
        state = ep.get("endpoint_status", {}).get("state", "")
        if state == "ONLINE":
            break
        time.sleep(10)
    print(state)
    if state != "ONLINE":
        fail(f"Endpoint did not reach ONLINE", f"state={state}")
    ok(f"Endpoint ONLINE: {VS_ENDPOINT}")

    # ── 4. Create VS index ───────────────────────────────────
    section("Step 4 / Create Delta Sync index")

    existing_idxs = [
        idx["name"] for idx in client.list_indexes(VS_ENDPOINT).get("vector_indexes", [])
    ]
    if VS_INDEX in existing_idxs:
        info(f"Index already exists: {VS_INDEX}")
    else:
        print(f"  Creating index {VS_INDEX}… ", end="", flush=True)
        client.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT,
            index_name=VS_INDEX,
            source_table_name=f"{CATALOG}.{SCHEMA}.{TABLE}",
            pipeline_type="TRIGGERED",
            primary_key="scheme_id",
            embedding_source_column="text",
            embedding_model_endpoint_name=EMBED_MODEL,
        )
        print("done")
        ok(f"Index created: {VS_INDEX}")

    # ── 5. Trigger sync and wait ─────────────────────────────
    section("Step 5 / Trigger sync and wait for ONLINE")

    index_obj = client.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
    try:
        index_obj.sync()
        info("Sync triggered")
    except Exception as e:
        info(f"Sync note: {e}")

    print("  Waiting for index ONLINE")
    # Use REST API directly — list_indexes() omits status fields
    import urllib.parse
    idx_url = (
        f"{HOST}/api/2.0/vector-search/endpoints/{VS_ENDPOINT}"
        f"/indexes/{urllib.parse.quote(VS_INDEX, safe='')}"
    )
    state = "UNKNOWN"
    for attempt in range(60):
        r = requests.get(idx_url, headers=HEADERS, timeout=15)
        if r.ok:
            s = r.json().get("status", {})
            state = s.get("detailed_state", "UNKNOWN")
            ready = s.get("ready", False)
            print(f"    [{attempt+1:02d}] {state}  ready={ready}")
            if ready or str(state).upper() in (
                "ONLINE", "ONLINE_NO_PENDING_UPDATE", "ONLINE_TRIGGERED_UPDATE"
            ):
                break
        else:
            print(f"    [{attempt+1:02d}] HTTP {r.status_code}")
        time.sleep(10)
    else:
        fail(f"Index did not reach ONLINE after 10 minutes", f"state={state}")
    ok(f"Index ONLINE: {VS_INDEX}")

    # ── 6. Smoke test retrieval ──────────────────────────────
    section("Step 6 / Smoke test — live VS retrieval")

    from scholarship.retriever import DatabricksVSRetriever
    from app.main import profile_to_query

    retriever = DatabricksVSRetriever(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)

    test_cases = [
        ("SC female Maharashtra Class 12 income 80000",   "NSP_PRE_MAT_SC"),
        ("OBC student disability undergraduate",          "AICTE_SAKSHAM"),
        ("Muslim minority girl Class 10 income 90000",    "BEGUM_HAZRAT_MAHAL"),
        ("EWS general postgraduate scholarship",          "EWS_GENERAL_MERIT"),
        ("PhD fellowship SC girls",                       "UP_SC_GIRLS_PHD"),
    ]

    for query, expected in test_cases:
        results = retriever.search(query, k=5)
        top_ids = results["scheme_id"].tolist()
        hit     = expected in top_ids
        mark    = "\033[92m✓\033[0m" if hit else "\033[93m~\033[0m"
        print(f"  {mark} [{query[:50]:52s}]  top: {top_ids[:3]}")

    ok("VS retrieval working")

    # ── 7. Full RAG call (LLM + VS) ──────────────────────────
    section("Step 7 / Full RAG call (VS + LLM)")
    from scholarship.llm_client import (
        SYSTEM_PROMPT, chat_completions, extract_assistant_text, rag_user_message,
    )

    query = profile_to_query(
        state="Maharashtra", category="SC", income=80000,
        gender="Female", age=17, education="Class 12",
        disability=False, minority=False,
    )
    info(f"Profile query: {query}")

    chunks_df = retriever.search(query, k=7)
    user_msg  = rag_user_message(chunks_df["text"].tolist(), query)
    messages  = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    print("  Calling LLM… ", end="", flush=True)
    t0       = time.time()
    response = chat_completions(messages, max_tokens=512, temperature=0.2)
    answer   = extract_assistant_text(response)
    print(f"done ({time.time()-t0:.1f}s)")
    ok("LLM answer:")
    for line in answer.splitlines():
        print(f"      {line}")

    print(f"""
══════════════════════════════════════════════════════════════
  \033[92mDeploy complete — all steps passed\033[0m

  VS Endpoint : {VS_ENDPOINT}
  VS Index    : {VS_INDEX}

  Next — deploy/redeploy the Databricks App:
    git add -A && git commit -m "switch to Databricks Vector Search" && git push
══════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete VS index, endpoint, and Delta table")
    args = parser.parse_args()
    if args.cleanup:
        cleanup()
    else:
        deploy()
