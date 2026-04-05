# ScholarshipGenie · छात्रवृत्ति जीनी

AI-powered scholarship eligibility finder for India. Enter your profile (state, category, income, education) and instantly get a ranked list of government schemes you qualify for — with award amounts, deadlines, and apply links — in any of 13 Indian languages.

Built for the **Databricks Hackathon** using Databricks Apps · Vector Search · Delta Lake · Llama 3.1 405B · Sarvam AI · Brevo.

> **Disclaimer:** This tool provides general eligibility guidance only. Always verify directly with the official scheme portal before applying.

---

## The Problem

India has 3,000+ scholarship schemes worth ₹18,000 crore annually — yet **70% of eligible students never apply** because schemes are scattered across 50+ portals, no personalised matching exists, and rural students have no guidance.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATABRICKS PLATFORM                    │
│                                                          │
│  ┌────────────┐   ┌──────────────────┐   ┌───────────┐  │
│  │ Delta Lake │──▶│  Vector Search   │──▶│ Llama 3.1 │  │
│  │ (37+ rows  │   │  Delta Sync      │   │ 405B via  │  │
│  │  CDF on)   │   │  GTE embeddings  │   │ Model     │  │
│  └────────────┘   └──────────────────┘   │ Serving   │  │
│        ▲                                 └─────┬─────┘  │
│        │                                       │         │
│  ┌─────────────┐                       ┌───────▼──────┐  │
│  │ UC Volume   │                       │ Databricks   │  │
│  │ /raw  PDFs  │                       │ Apps (Gradio)│  │
│  │ /appdata    │                       │ OAuth M2M    │  │
│  └─────────────┘                       └───────┬──────┘  │
└───────────────────────────────────────────────┼─────────┘
                                                │
                                    ┌───────────▼──────────┐
                                    │   Student Browser    │
                                    │   (any language)     │
                                    └──────────────────────┘
```

---

## Data Pipeline

```
Stage 1 — Ingestion (Admin)
  Admin uploads PDF → LLM extracts structured JSON → Delta INSERT
  → VS Delta Sync triggered → wait for ONLINE_NO_PENDING_UPDATE
  → Brevo email sent to matched users

Stage 2 — Query (Student)
  Student fills profile → natural language query
  → Vector Search k=7 semantic retrieval
  → RAG context + Llama 3.1 405B → ranked eligible schemes
  → Sarvam AI translation → results in student's language

Stage 3 — Evaluation
  Each response scored: BhashaBench proxy (groundedness, completeness,
  format, language rendering) → metrics logged to MLflow
```

---

## Features

| Feature | Description |
|---------|-------------|
| AI eligibility matching | Llama 3.1 405B checks 8 criteria per scheme |
| Ranked results | Sorted by award value, ★ best pick highlighted |
| 13 Indian languages | Sarvam AI translation + TTS read-aloud |
| Admin PDF ingest | Upload scheme PDF → LLM extracts → live in minutes |
| Email notifications | Brevo alerts when new schemes match saved profiles |
| BhashaBench evaluation | Proxy scorecard logged to MLflow after every search |
| Persistent profiles | Stored in UC Volume — survives redeploys |

---

## Repository Layout

```
scholarship-schemes-hackathon/
├── app/
│   └── main.py                     # Gradio UI — all event handlers
├── src/scholarship/
│   ├── llm_client.py               # Databricks Model Serving wrapper + SYSTEM_PROMPT
│   ├── retriever.py                # DatabricksVSRetriever (Vector Search REST API)
│   ├── admin_service.py            # PDF ingest pipeline (extract → Delta → VS sync)
│   ├── notification_service.py     # Batch notify + direct scheme injection
│   ├── email_client.py             # Brevo transactional email
│   ├── user_store.py               # UC Volume JSON user store (PBKDF2 passwords)
│   ├── profile_matching.py         # profile_to_query, matching_schemes_for_profile
│   ├── evaluation.py               # BhashaBench-style proxy scorecard
│   ├── mlflow_tracking.py          # MLflow experiment logging
│   └── sarvam_client.py            # Sarvam translate / TTS / STT
├── scripts/
│   ├── deploy_dummy.py             # One-shot: Delta table + VS index setup
│   └── notify_users.py             # Standalone batch notify runner
├── app.yaml                        # Databricks Apps entry point + env vars + secrets
├── requirements.txt
└── requirements-app.txt
```

---

## Prerequisites

- Databricks workspace (Free Edition works — [sign up](https://www.databricks.com/try-databricks))
- Unity Catalog enabled (default on new workspaces)
- Sarvam AI API key — [dashboard.sarvam.ai](https://dashboard.sarvam.ai) (free tier)
- Brevo account for email — [brevo.com](https://brevo.com) (300 free/day, verify your sender address)
- Git + Python 3.10+

---

## Step 1 — Clone and install

```bash
git clone https://github.com/Rohit1217/scholarship-schemes-hackathon.git
cd scholarship-schemes-hackathon

conda create -n scholargenie python=3.10 -y
conda activate scholargenie
pip install -r requirements.txt
pip install -e .
```

---

## Step 2 — Create `.env` for local use

```bash
cp .env.example .env
```

Fill in `.env`:

```bash
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<your-pat>         # Settings → Developer → Access tokens

# LLM — copy base_url from Model Serving endpoint "Get code" dialog
LLM_OPENAI_BASE_URL=https://<your-workspace>.cloud.databricks.com/serving-endpoints
LLM_MODEL=databricks-meta-llama-3.1-405b-instruct

# Vector Search
VS_ENDPOINT_NAME=scholarship-vs-endpoint
VS_INDEX_NAME=main.scholarships.scheme_vs_index

# Brevo email notifications
BREVO_API_KEY=<your-brevo-api-key>
EMAIL_FROM_ADDRESS=<your-verified-sender@example.com>
EMAIL_FROM_NAME=ScholarshipGenie

# Sarvam (translation + TTS)
SARVAM_API_KEY=<your-sarvam-key>

# Admin panel
ADMIN_PASSWORD=<choose-a-password>

# Persistent user store (UC Volume path)
USER_DETAILS_STORE_PATH=/Volumes/main/scholarships/appdata/user_profiles.json
```

---

## Step 3 — Unity Catalog setup

Open a notebook in your Databricks workspace and run:

```sql
CREATE SCHEMA IF NOT EXISTS main.scholarships;
CREATE VOLUME IF NOT EXISTS main.scholarships.raw;      -- scheme PDFs + CSV uploads
CREATE VOLUME IF NOT EXISTS main.scholarships.appdata;  -- user profiles JSON
```

Then create the Delta table with Change Data Feed (required for VS Delta Sync):

```sql
CREATE TABLE IF NOT EXISTS main.scholarships.scheme_corpus (
  scheme_id       STRING,
  scheme_name     STRING,
  categories      STRING,
  income_limit    BIGINT,
  education_level STRING,
  gender          STRING,
  state           STRING,
  award_amount    STRING,
  deadline        STRING,
  description     STRING,
  source_url      STRING,
  text            STRING
)
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
```

---

## Step 4 — Load initial scheme data and create the Vector Search index

```bash
export $(grep -v '^#' .env | xargs)
python scripts/deploy_dummy.py
```

This script:
1. Inserts 37 central government schemes into `main.scholarships.scheme_corpus`
2. Creates VS endpoint `scholarship-vs-endpoint` (if it doesn't exist)
3. Creates Delta Sync index `main.scholarships.scheme_vs_index` using `databricks-gte-large-en` embeddings
4. Polls until the index reaches `ONLINE_NO_PENDING_UPDATE` (~5–10 min first time)
5. Runs a smoke-test retrieval query

**Expected output:**
```
Inserted 37 rows into main.scholarships.scheme_corpus
Creating VS endpoint...done
Creating VS index...done
Waiting for index: ONLINE_TRIGGERED_UPDATE ... ONLINE_NO_PENDING_UPDATE ✓
Smoke test — top result: NSP_POST_MAT_SC (score=0.87)
Setup complete.
```

---

## Step 5 — Store secrets

```bash
export $(grep -v '^#' .env | xargs)
HOST=$DATABRICKS_HOST
TOKEN=$DATABRICKS_TOKEN

# Create secret scope (run once)
curl -s -X POST "$HOST/api/2.0/secrets/scopes/create" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"scope":"scholarship","initial_manage_principal":"users"}'

# Store each secret
curl -s -X POST "$HOST/api/2.0/secrets/put" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"scope\":\"scholarship\",\"key\":\"sarvam_api_key\",\"string_value\":\"$SARVAM_API_KEY\"}"

curl -s -X POST "$HOST/api/2.0/secrets/put" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"scope\":\"scholarship\",\"key\":\"brevo_api_key\",\"string_value\":\"$BREVO_API_KEY\"}"

curl -s -X POST "$HOST/api/2.0/secrets/put" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"scope\":\"scholarship\",\"key\":\"databricks_token\",\"string_value\":\"$DATABRICKS_TOKEN\"}"
```

Verify (shows key names only, not values):
```bash
curl -s "$HOST/api/2.0/secrets/list?scope=scholarship" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

---

## Step 6 — Configure `app.yaml`

Edit `app.yaml` with your workspace values:

```yaml
command: ["python", "app/main.py"]

env:
  - name: "LLM_OPENAI_BASE_URL"
    value: "https://<your-workspace>.cloud.databricks.com/serving-endpoints"
  - name: "LLM_MODEL"
    value: "databricks-meta-llama-3.1-405b-instruct"
  - name: "VS_ENDPOINT_NAME"
    value: "scholarship-vs-endpoint"
  - name: "VS_INDEX_NAME"
    value: "main.scholarships.scheme_vs_index"
  - name: "EMAIL_FROM_ADDRESS"
    value: "<your-verified-sender@example.com>"
  - name: "EMAIL_FROM_NAME"
    value: "ScholarshipGenie"
  - name: "ADMIN_PASSWORD"
    value: "<your-admin-password>"
  - name: "USER_DETAILS_STORE_PATH"
    value: "/Volumes/main/scholarships/appdata/user_profiles.json"

secrets:
  - scope: "scholarship"
    key: "sarvam_api_key"
    env_var: "SARVAM_API_KEY"
  - scope: "scholarship"
    key: "databricks_token"
    env_var: "DATABRICKS_TOKEN"
  - scope: "scholarship"
    key: "brevo_api_key"
    env_var: "BREVO_API_KEY"
```

Commit and push:
```bash
git add app.yaml
git commit -m "configure workspace endpoints"
git push
```

---

## Step 7 — Deploy the app

### 7a. Sync repo into Databricks Workspace (run once)

```bash
export $(grep -v '^#' .env | xargs)
HOST=$DATABRICKS_HOST
TOKEN=$DATABRICKS_TOKEN
YOUR_EMAIL=<your-databricks-login-email>

curl -s -X POST "$HOST/api/2.0/repos" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"https://github.com/Rohit1217/scholarship-schemes-hackathon.git\",
    \"provider\": \"gitHub\",
    \"path\": \"/Repos/$YOUR_EMAIL/scholarship-schemes-hackathon\"
  }"
```

Note the `id` in the response — you need it for redeployments.

### 7b. Create and deploy the app (run once)

```bash
# Create the app
curl -s -X POST "$HOST/api/2.0/apps" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "scholarship-finder"}'

# Deploy
curl -s -X POST "$HOST/api/2.0/apps/scholarship-finder/deployments" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"source_code_path\": \"/Workspace/Users/$YOUR_EMAIL/scholarship-schemes-hackathon\"}"
```

Poll until `SUCCEEDED`:
```bash
DEPLOY_ID=<deployment_id from above>
curl -s "$HOST/api/2.0/apps/scholarship-finder/deployments/$DEPLOY_ID" \
  -H "Authorization: Bearer $TOKEN" | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(d['status']['state'], '-', d['status'].get('message',''))
"
```

### 7c. Redeploy after every code change

```bash
REPO_ID=<repo-id from 7a>

# Pull latest code from GitHub into the workspace repo
curl -s -X PATCH "$HOST/api/2.0/repos/$REPO_ID" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"branch": "main"}'

# Deploy
curl -s -X POST "$HOST/api/2.0/apps/scholarship-finder/deployments" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"source_code_path\": \"/Workspace/Users/$YOUR_EMAIL/scholarship-schemes-hackathon\"}"
```

---

## Step 8 — Grant the app service principal permissions

The app runs under a service principal. Find its ID at:
**Compute → Apps → scholarship-finder → Service Principal**

Run in a Databricks SQL notebook:

```sql
GRANT USE CATALOG ON CATALOG main TO `<service-principal-id>`;
GRANT USE SCHEMA ON SCHEMA main.scholarships TO `<service-principal-id>`;
GRANT SELECT ON TABLE main.scholarships.scheme_corpus TO `<service-principal-id>`;
GRANT INSERT ON TABLE main.scholarships.scheme_corpus TO `<service-principal-id>`;
GRANT READ VOLUME ON VOLUME main.scholarships.appdata TO `<service-principal-id>`;
GRANT WRITE VOLUME ON VOLUME main.scholarships.appdata TO `<service-principal-id>`;
GRANT READ VOLUME ON VOLUME main.scholarships.raw TO `<service-principal-id>`;
GRANT WRITE VOLUME ON VOLUME main.scholarships.raw TO `<service-principal-id>`;
```

Grant secret scope access:
```bash
curl -s -X POST "$HOST/api/2.0/secrets/acls/put" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"scope":"scholarship","principal":"<service-principal-id>","permission":"READ"}'
```

Grant Vector Search access via UI:
**AI/Machine Learning → Vector Search → scholarship-vs-endpoint → Permissions → Add SP → CAN_USE**

Then restart the app.

---

## Running Locally

```bash
export $(grep -v '^#' .env | xargs)
python app/main.py
# Open http://localhost:7860
```

---
## Working links
App-link : https://scholarship-finder-7474653566201260.aws.databricksapps.com/?

Video-link : https://drive.google.com/drive/folders/1T_oQ_FPR-m9jTS6s5t6i55RswTBsfWQ9
## Demo Steps

1. **Register** — create an account (ST, Assam, ₹1.5L income, Undergraduate, Female)
2. **Save Profile** → click "Find Scholarships" — results in ~10 sec
3. **Ranked results** — ★ best pick, award amounts, deadlines, apply links
4. **Switch language** to Hindi — results translate; URLs stay in English
5. **Read aloud** — check "Read results aloud" — Sarvam TTS plays the results
6. **Admin panel** — scroll to bottom, enter admin password
7. **Ingest PDF** — upload a new scheme PDF — watch 4-step live progress
8. **Email notification** — registered user receives email if new scheme matches their profile
9. **BhashaBench score** — proxy scorecard appears below results after every search

---

## Environment Variables Reference

| Variable | Where set | Description |
|----------|-----------|-------------|
| `LLM_OPENAI_BASE_URL` | `app.yaml` | Model Serving base URL |
| `LLM_MODEL` | `app.yaml` | Model name (e.g. `databricks-meta-llama-3.1-405b-instruct`) |
| `VS_ENDPOINT_NAME` | `app.yaml` | Vector Search endpoint name |
| `VS_INDEX_NAME` | `app.yaml` | Full VS index name (`catalog.schema.index`) |
| `EMAIL_FROM_ADDRESS` | `app.yaml` | Verified Brevo sender address |
| `EMAIL_FROM_NAME` | `app.yaml` | Sender display name |
| `ADMIN_PASSWORD` | `app.yaml` | Admin panel password |
| `USER_DETAILS_STORE_PATH` | `app.yaml` | UC Volume path for user JSON store |
| `SARVAM_API_KEY` | secret scope | Sarvam AI key for translation + TTS |
| `DATABRICKS_TOKEN` | secret scope | PAT for VS + Model Serving calls |
| `BREVO_API_KEY` | secret scope | Brevo API key for email delivery |
| `ENABLE_MLFLOW_LOGGING` | optional env | Set to `0` to disable MLflow logging |
| `MLFLOW_EXPERIMENT_NAME` | optional env | MLflow experiment path (default: `/Shared/scholarship-finder-app`) |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| 502 on startup | Check app logs: Compute → Apps → Logs. Usually a missing import or secret. |
| `VS endpoint not found` | Run `python scripts/deploy_dummy.py` to create endpoint and index. |
| Index never reaches `ONLINE_NO_PENDING_UPDATE` | First-time indexing takes 5–10 min. App polls up to 15 min. |
| Results always in English | `SARVAM_API_KEY` not loaded — check SP has READ on secret scope. |
| `401` on LLM calls | Grant `CAN_QUERY` on the Model Serving endpoint to the app SP. |
| `403` on VS search | Grant `CAN_USE` on `scholarship-vs-endpoint` to the app SP via UI. |
| Emails not sending | Verify sender address in Brevo. Check `BREVO_API_KEY` is in secret scope. |
| User data lost on redeploy | Ensure `USER_DETAILS_STORE_PATH` is a UC Volume path, not a local path. |
| Admin PDF extracts 0 schemes | PDF may be image-based. LLM retries 3×. Check app logs for raw LLM response. |
