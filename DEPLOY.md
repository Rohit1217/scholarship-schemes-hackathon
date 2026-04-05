# Deployment Guide

Step-by-step guide for anyone who has cloned the repo and wants to edit, test, and deploy the app on their own Databricks Free Edition workspace.

---

## Prerequisites

- Databricks Free Edition workspace ([sign up](https://www.databricks.com/try-databricks))
- Sarvam AI API key ([sign up](https://www.sarvam.ai/))
- Python 3.10+ with conda or venv
- Git

---

## 1. Clone and set up locally

```bash
git clone https://github.com/Rohit1217/scholarship-schemes-hackathon.git
cd scholarship-schemes-hackathon

# Create conda env
conda create -n llm python=3.10 -y
conda activate llm

# Install dependencies
pip install -r requirements.txt
pip install -e ".[app]"   # installs src/scholarship as a package
```

---

## 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` with your values:

```
DATABRICKS_HOST=https://<your-workspace-id>.cloud.databricks.com
DATABRICKS_TOKEN=<your-personal-access-token>
SARVAM_API_KEY=<your-sarvam-api-key>

LLM_OPENAI_BASE_URL=https://<your-workspace-id>.cloud.databricks.com/serving-endpoints
LLM_MODEL=databricks-llama-4-maverick

VS_ENDPOINT_NAME=scholarship-vs-endpoint
VS_INDEX_NAME=main.scholarships.scheme_vs_index
```

**How to get a Databricks Personal Access Token (PAT):**
> Workspace → top-right profile icon → Settings → Developer → Access tokens → Generate new token

---

## 3. Set up Unity Catalog

In your Databricks workspace, open a notebook or the SQL editor and run:

```sql
CREATE CATALOG IF NOT EXISTS main;
CREATE SCHEMA IF NOT EXISTS main.scholarships;
CREATE VOLUME IF NOT EXISTS main.scholarships.raw;
CREATE VOLUME IF NOT EXISTS main.scholarships.rag;
```

---

## 4. Deploy data + Vector Search index

This uploads dummy scheme data, creates the Delta table, and sets up the Databricks Vector Search index.

```bash
python scripts/deploy_dummy.py
```

This will:
1. Upload `data/dummy_schemes.csv` to `/Volumes/main/scholarships/raw/`
2. Create Delta table `main.scholarships.scheme_corpus` with Change Data Feed enabled
3. Create VS endpoint `scholarship-vs-endpoint`
4. Create Delta Sync index `main.scholarships.scheme_vs_index` (embeds via `databricks-gte-large-en`)
5. Wait for index to reach ONLINE status (~5–10 min first time)
6. Smoke-test retrieval + full RAG call to verify end-to-end

To tear down test data after:
```bash
python scripts/deploy_dummy.py --cleanup
```

---

## 5. Set up the Databricks secret scope

The app reads `SARVAM_API_KEY` from a Databricks secret scope at startup. Run this once:

```bash
python - <<'EOF'
import os, requests
for line in open(".env").readlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

HOST  = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]
KEY   = os.environ["SARVAM_API_KEY"]
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Create scope
requests.post(f"{HOST}/api/2.0/secrets/scopes/create", headers=headers,
    json={"scope": "scholarship", "initial_manage_principal": "users"})

# Store key
r = requests.post(f"{HOST}/api/2.0/secrets/put", headers=headers,
    json={"scope": "scholarship", "key": "sarvam_api_key", "string_value": KEY})
print("Secret stored:", r.status_code)
EOF
```

---

## 6. Deploy to Databricks Apps

### 6a. Create the app and sync the repo (run once)

```bash
python - <<'EOF'
import os, requests, json
for line in open(".env").readlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

HOST  = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]
GITHUB_URL = "https://github.com/Rohit1217/scholarship-schemes-hackathon.git"
YOUR_EMAIL = "<your-databricks-email>"   # your Databricks login email
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Create the app
r = requests.post(f"{HOST}/api/2.0/apps", headers=headers,
    json={"name": "scholarship-finder"}, timeout=30)
print("Create app:", r.status_code, r.json().get("name", r.json()))

# Sync repo into workspace
r = requests.post(f"{HOST}/api/2.0/repos", headers=headers,
    json={"url": GITHUB_URL, "provider": "gitHub",
          "path": f"/Repos/{YOUR_EMAIL}/scholarship-schemes-hackathon"}, timeout=30)
data = r.json()
print("Sync repo:", r.status_code, "id=", data.get("id"), "commit=", data.get("head_commit_id", data))
EOF
```

### 6b. Grant the app service principal permissions

After the app is created, get its service principal application ID from:
> Compute → Apps → scholarship-finder → (note the Service Principal client ID)

Then run:

```bash
python - <<'EOF'
import os, requests, json, time
for line in open(".env").readlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

HOST  = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]
SP_APP_ID = "<service-principal-client-id>"   # from Compute → Apps page
WH_ID     = "<sql-warehouse-id>"              # from SQL → Warehouses
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def run_sql(stmt):
    r = requests.post(f"{HOST}/api/2.0/sql/statements", headers=headers,
        json={"statement": stmt, "warehouse_id": WH_ID, "wait_timeout": "30s"}, timeout=60)
    d = r.json()
    stmt_id = d.get("statement_id")
    state = d.get("status", {}).get("state", "?")
    for _ in range(10):
        if state in ("SUCCEEDED", "FAILED", "CANCELED"): break
        time.sleep(3)
        d = requests.get(f"{HOST}/api/2.0/sql/statements/{stmt_id}", headers=headers).json()
        state = d.get("status", {}).get("state", "?")
    print(f"  {state}  {stmt[:70]}")

# UC privileges
run_sql(f"GRANT USE CATALOG ON CATALOG main TO `{SP_APP_ID}`")
run_sql(f"GRANT USE SCHEMA ON SCHEMA main.scholarships TO `{SP_APP_ID}`")
run_sql(f"GRANT SELECT ON TABLE main.scholarships.scheme_corpus TO `{SP_APP_ID}`")

# Secret scope access
r = requests.post(f"{HOST}/api/2.0/secrets/acls/put", headers=headers,
    json={"scope": "scholarship", "principal": SP_APP_ID, "permission": "READ"})
print("Secret ACL:", r.status_code)

# Vector Search endpoint access — done via UI: AI/Machine Learning → Vector Search → endpoint → Permissions → CAN_USE
print("NOTE: Also grant CAN_USE on VS endpoint 'scholarship-vs-endpoint' to the service principal via the UI.")
EOF
```

### 6c. Deploy (and redeploy after every code change)

```bash
python - <<'EOF'
import os, requests, json, time
for line in open(".env").readlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

HOST  = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]
REPO_ID    = "<repo-id>"       # printed during step 6a (id field)
YOUR_EMAIL = "<your-email>"    # your Databricks login email
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Pull latest code from GitHub into the workspace repo
r = requests.patch(f"{HOST}/api/2.0/repos/{REPO_ID}", headers=headers,
    json={"branch": "main"}, timeout=30)
print("Repo synced to:", r.json().get("head_commit_id"))

# Deploy
r = requests.post(f"{HOST}/api/2.0/apps/scholarship-finder/deployments", headers=headers,
    json={"source_code_path": f"/Workspace/Users/{YOUR_EMAIL}/scholarship-schemes-hackathon",
          "mode": "SNAPSHOT"}, timeout=30)
deploy_id = r.json().get("deployment_id")
print("Deploying:", deploy_id)

for i in range(20):
    time.sleep(15)
    d = requests.get(f"{HOST}/api/2.0/apps/scholarship-finder/deployments/{deploy_id}",
        headers=headers, timeout=30).json()
    state = d.get("status", {}).get("state", "?")
    msg   = d.get("status", {}).get("message", "")
    print(f"  [{i+1}] {state} — {msg}")
    if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
        break
EOF
```

---

## 7. Access the app

After deployment succeeds, the app URL is shown in:
> Databricks workspace → Compute → Apps → scholarship-finder

---

## Workflow for code changes

```
1. Edit code locally
2. git add . && git commit -m "..." && git push
3. Run the redeploy script from step 6c
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 502 Bad Gateway | App startup timeout | Check logs: Compute → Apps → Logs |
| `VS endpoint not ONLINE` | Endpoint still provisioning | Wait 5–10 min; re-run `deploy_dummy.py` |
| `Index not found` | VS index not created | Run `setup_vector_search.ipynb` or `deploy_dummy.py` |
| TTS silent | SARVAM_API_KEY not loaded | Re-run step 5; check SP has secret scope READ |
| `401 Unauthorized` on LLM | SP missing serving endpoint permission | Grant CAN_QUERY on the LLM endpoint |
| `403` on VS search | SP missing VS endpoint permission | Grant CAN_USE on `scholarship-vs-endpoint` via UI |
