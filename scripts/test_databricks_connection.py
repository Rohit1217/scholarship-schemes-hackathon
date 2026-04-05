"""Targeted Databricks connectivity test.

Tests in order:
  1. Workspace REST API (SDK auth, current user, cluster list)
  2. Unity Catalog (catalog → schema → volume existence)
  3. Serving endpoints / AI Gateway — discovers the correct LLM URL
  4. Direct HTTP call to the LLM endpoint

Usage:
    conda run -n llm python scripts/test_databricks_connection.py
"""
from __future__ import annotations
import os, sys, socket, time, json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load .env
for name in (".env", ".env.example"):
    p = ROOT / name
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()
        print(f"Loaded credentials from {name}\n")
        break

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
INFO = "  INFO"

def section(t): print(f"\n{'─'*58}\n  {t}\n{'─'*58}")
def ok(msg):    print(f"{PASS}  {msg}")
def fail(msg, detail=""): print(f"{FAIL}  {msg}" + (f"\n         → {detail}" if detail else ""))
def info(msg):  print(f"{INFO}  {msg}")

HOST  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# ─────────────────────────────────────────────────────────
# 1. Basic reachability — DNS + TCP to workspace
# ─────────────────────────────────────────────────────────
section("1 / Workspace reachability")

from urllib.parse import urlparse
ws_host = urlparse(HOST).hostname or ""

try:
    ip = socket.gethostbyname(ws_host)
    ok(f"DNS resolved: {ws_host} → {ip}")
except Exception as e:
    fail(f"DNS failed for {ws_host}", str(e))
    sys.exit(1)

try:
    s = socket.create_connection((ws_host, 443), timeout=5)
    s.close()
    ok(f"TCP port 443 open on {ws_host}")
except Exception as e:
    fail("TCP connection failed", str(e))
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# 2. Workspace REST API — current user + cluster list
# ─────────────────────────────────────────────────────────
section("2 / Workspace REST API")

r = requests.get(f"{HOST}/api/2.0/preview/scim/v2/Me", headers=HEADERS, timeout=15)
if r.ok:
    user = r.json().get("userName", "?")
    ok(f"Authenticated as: {user}")
else:
    fail(f"Auth failed: HTTP {r.status_code}", r.text[:200])
    sys.exit(1)

r = requests.get(f"{HOST}/api/2.0/clusters/list", headers=HEADERS, timeout=15)
if r.ok:
    clusters = r.json().get("clusters", [])
    ok(f"Cluster list: {len(clusters)} cluster(s)")
    for c in clusters[:3]:
        state = c.get("state", "?")
        name  = c.get("cluster_name", "?")
        info(f"  [{state:12s}]  {name}")
else:
    fail(f"Cluster list: HTTP {r.status_code}", r.text[:200])

# ─────────────────────────────────────────────────────────
# 3. Unity Catalog — check main.scholarships exists
# ─────────────────────────────────────────────────────────
section("3 / Unity Catalog")

r = requests.get(
    f"{HOST}/api/2.1/unity-catalog/schemas",
    headers=HEADERS,
    params={"catalog_name": "main"},
    timeout=15,
)
if r.ok:
    schemas = [s["name"] for s in r.json().get("schemas", [])]
    ok(f"Schemas in `main` catalog: {schemas}")
    if "scholarships" in schemas:
        ok("Schema `main.scholarships` exists ✓")
    else:
        fail("Schema `main.scholarships` NOT found",
             "Run CREATE SCHEMA IF NOT EXISTS main.scholarships in a notebook first")
else:
    fail(f"Unity Catalog schemas: HTTP {r.status_code}", r.text[:200])

r = requests.get(
    f"{HOST}/api/2.1/unity-catalog/volumes",
    headers=HEADERS,
    params={"catalog_name": "main", "schema_name": "scholarships"},
    timeout=15,
)
if r.ok:
    vols = [v["name"] for v in r.json().get("volumes", [])]
    ok(f"Volumes in `main.scholarships`: {vols}")
    for vol in ("raw", "rag"):
        if vol in vols:
            ok(f"  Volume `{vol}` exists ✓")
        else:
            fail(f"  Volume `{vol}` NOT found — create it in a notebook")
else:
    info(f"Volumes check: HTTP {r.status_code} (schema may not exist yet)")

# ─────────────────────────────────────────────────────────
# 4. Serving endpoints — find AI Gateway / LLM endpoints
# ─────────────────────────────────────────────────────────
section("4 / Serving endpoints & AI Gateway")

r = requests.get(
    f"{HOST}/api/2.0/serving-endpoints",
    headers=HEADERS,
    timeout=15,
)
if r.ok:
    endpoints = r.json().get("endpoints", [])
    ok(f"Found {len(endpoints)} serving endpoint(s):")
    for ep in endpoints:
        name  = ep.get("name", "?")
        state = ep.get("state", {}).get("ready", "?")
        info(f"  [{state:5}]  {name}")
else:
    fail(f"Serving endpoints: HTTP {r.status_code}", r.text[:200])
    endpoints = []

# Check the configured AI Gateway URL
configured_url = os.environ.get("LLM_OPENAI_BASE_URL", "")
configured_model = os.environ.get("LLM_MODEL", "")
info(f"Configured LLM_OPENAI_BASE_URL : {configured_url}")
info(f"Configured LLM_MODEL           : {configured_model}")

# Resolve DNS for AI Gateway host
if configured_url:
    gw_host = urlparse(configured_url).hostname or ""
    try:
        ip = socket.gethostbyname(gw_host)
        ok(f"AI Gateway DNS resolved: {gw_host} → {ip}")
        gw_dns_ok = True
    except Exception as e:
        fail(f"AI Gateway DNS FAILED for: {gw_host}", str(e))
        gw_dns_ok = False

        # Try to build alternative URLs from workspace host
        # Format 1: same dbc-xxx hostname with .ai-gateway subdomain
        # Format 2: numeric workspace ID (found in workspace URL or org ID)
        r2 = requests.get(f"{HOST}/api/2.0/workspace/get-status",
                          headers=HEADERS,
                          params={"path": "/"},
                          timeout=10)
        if r2.ok:
            # Try to get the org/workspace numeric ID
            r3 = requests.get(f"{HOST}/api/2.0/preview/workspace-info",
                              headers=HEADERS, timeout=10)
            if r3.ok:
                org_id = r3.json().get("workspace_id") or r3.json().get("orgId", "")
                if org_id:
                    alt_url = f"https://{org_id}.ai-gateway.cloud.databricks.com/mlflow/v1"
                    alt_host = f"{org_id}.ai-gateway.cloud.databricks.com"
                    info(f"Trying alternative AI Gateway URL: {alt_url}")
                    try:
                        ip2 = socket.gethostbyname(alt_host)
                        ok(f"Alternative URL DNS resolved: {alt_host} → {ip2}")
                        info(f"UPDATE your LLM_OPENAI_BASE_URL to: {alt_url}")
                    except Exception:
                        pass

# ─────────────────────────────────────────────────────────
# 5. Live LLM call — single token ping
# ─────────────────────────────────────────────────────────
section("5 / LLM endpoint — live call")

from scholarship.llm_client import chat_completions, extract_assistant_text

if not configured_url:
    fail("LLM_OPENAI_BASE_URL not set")
elif not configured_model:
    fail("LLM_MODEL not set")
elif not gw_dns_ok:
    fail("Skipping LLM call — AI Gateway DNS unresolvable from this machine",
         "The LLM will work on Databricks Apps (same network as workspace). "
         "Test it from a Databricks notebook instead (see below).")
    print("""
  ┌─ Test in a Databricks notebook ──────────────────────────────┐
  │  import sys, os                                               │
  │  sys.path.insert(0, "/Workspace/Users/<you>/               │
  │      scholarship_schemes_hackathon/src")                      │
  │                                                               │
  │  os.environ["LLM_OPENAI_BASE_URL"] = "<your-gateway-url>"   │
  │  os.environ["LLM_MODEL"] = "databricks-meta-llama-3-1-70b"  │
  │  # token comes from the notebook context automatically        │
  │                                                               │
  │  from scholarship.llm_client import chat_completions, \\     │
  │      extract_assistant_text                                   │
  │  r = chat_completions(                                        │
  │      [{"role":"user","content":"Say: LLM OK"}],              │
  │      max_tokens=10)                                           │
  │  print(extract_assistant_text(r))                            │
  └───────────────────────────────────────────────────────────────┘
""")
else:
    ping = [{"role": "user", "content": "Reply with exactly two words: LLM OK"}]
    print("  Calling LLM (ping)… ", end="", flush=True)
    t0 = time.time()
    try:
        resp = chat_completions(ping, max_tokens=10, temperature=0)
        elapsed = time.time() - t0
        text = extract_assistant_text(resp)
        print(f"done ({elapsed:.1f}s)")
        ok(f"LLM responded: '{text}'")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"failed ({elapsed:.1f}s)")
        fail("LLM call", str(e))

# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
print(f"\n{'═'*58}")
print("  Done.")
