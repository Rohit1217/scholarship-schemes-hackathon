# Scholarship Finder · छात्रवृत्ति खोजक

**Multilingual scholarship eligibility assistant for India** — users fill in a profile
(state, category, income, education, etc.) and the app returns a ranked list of government
scholarship schemes they are likely eligible for, with reasoning, award amounts, and
deadlines — in any of 13 Indian languages.

Built on **Databricks Free Edition** using FAISS RAG, Databricks AI Gateway (LLM), and
Sarvam AI (translation + TTS). Deployed as a **Databricks App** via Gradio.

> **Disclaimer:** This tool provides general eligibility guidance only.
> Always verify directly with the scheme's official portal before applying.

---

## How it works

```
User fills profile form (state, category, income, education …)
  → Profile serialised to English query sentence
  → [if non-English UI selected] Sarvam Mayura translates display → English
  → FAISS semantic search over embedded scholarship scheme corpus
  → Top-7 scheme chunks sent as context to Databricks LLM
  → LLM generates numbered eligibility list with reasoning + award + deadline
  → [if non-English UI selected] Sarvam Mayura translates answer → user's language
  → Bilingual response displayed (user's language + English) + scheme source list
  → Optional: Sarvam Bulbul reads the answer aloud (TTS)
```

**Supported languages:** English · Hindi · Bengali · Telugu · Marathi · Tamil ·
Gujarati · Kannada · Malayalam · Punjabi · Odia · Urdu · Assamese

---

## Table of contents

1. [Technology stack](#technology-stack)
2. [Repository layout](#repository-layout)
3. [Dataset requirements](#dataset-requirements)
   - [CSV schema](#csv-schema)
   - [Sample row](#sample-row)
   - [Sourcing schemes](#where-to-source-scheme-data)
   - [Validation checklist](#dataset-validation-checklist)
4. [Setup: step by step](#setup-step-by-step)
   - [Step 1 — Prerequisites](#step-1--prerequisites)
   - [Step 2 — Clone the repo](#step-2--clone-the-repo)
   - [Step 3 — Databricks auth](#step-3--databricks-authentication)
   - [Step 4 — Unity Catalog setup](#step-4--unity-catalog-setup)
   - [Step 5 — Upload your dataset](#step-5--upload-your-dataset)
   - [Step 6 — Store secrets](#step-6--store-secrets)
   - [Step 7 — Get your LLM endpoint URL](#step-7--get-your-llm-endpoint-url)
   - [Step 8 — Configure app.yaml](#step-8--configure-appyaml)
   - [Step 9 — Run the ingest notebook](#step-9--run-the-ingest-notebook)
   - [Step 10 — Build the FAISS index](#step-10--build-the-faiss-index)
   - [Step 11 — Deploy the app](#step-11--deploy-the-app)
   - [Step 12 — Grant permissions](#step-12--grant-permissions)
5. [Local development](#local-development)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## Technology stack

| Component | Technology |
|-----------|------------|
| LLM | Databricks AI Gateway → Llama (configurable via `LLM_MODEL`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector search | FAISS `IndexFlatIP` (cosine similarity via L2-normalised vectors) |
| Translation | Sarvam Mayura (`POST /translate`) |
| Text-to-speech | Sarvam Bulbul (`POST /text-to-speech`) |
| App framework | Gradio 4.44 on Databricks Apps |
| Data platform | Databricks Free Edition (Unity Catalog, Volumes, Apps) |
| Scheme corpus | Delta table `main.scholarships.scheme_corpus` |

---

## Repository layout

```
scholarship_schemes_hackathon/
├── app/
│   └── main.py                    # Gradio app: profile form → RAG → LLM → results
├── src/
│   └── scholarship/
│       ├── __init__.py
│       ├── llm_client.py          # Databricks AI Gateway wrapper
│       ├── retriever.py           # FAISSRetriever + get_retriever() factory
│       └── sarvam_client.py       # Sarvam translate / STT / TTS
├── notebooks/
│   ├── scholarship_ingest.ipynb   # CSV → Delta table (run first)
│   └── build_rag_index.ipynb      # Delta table → FAISS index (run second)
├── tests/
│   └── test_retriever.py
├── docs/
│   ├── DEVELOPER_GUIDE.md
│   └── WORKSPACE_SETUP.md
├── .env.example                   # Copy to .env for local dev
├── .gitignore
├── app.yaml                       # Databricks Apps entry point + env vars
├── pyproject.toml
├── requirements.txt
└── requirements-app.txt
```

---

## Dataset requirements

The entire RAG pipeline depends on a CSV file (or multiple CSVs) describing
scholarship schemes. **You must prepare this before running any notebook.**

### CSV schema

Each row is one scholarship scheme. All 16 columns are required; use an empty
string (not NULL) for fields that don't apply.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `scheme_id` | string | Unique identifier — no spaces | `NSP_PRE_MAT_SC_2024` |
| `scheme_name` | string | Official full name | `Pre-Matric Scholarship for SC Students` |
| `administering_body` | string | Ministry or department | `Ministry of Social Justice and Empowerment` |
| `level` | string | `Central` or `State` | `Central` |
| `state` | string | State name for state-level schemes; leave blank for central | `Maharashtra` |
| `eligible_categories` | string | Comma-separated social categories | `SC, ST` |
| `min_income_limit` | integer | Min annual family income INR (0 if no lower bound) | `0` |
| `max_income_limit` | integer | Max annual family income INR | `200000` |
| `eligible_gender` | string | `Male`, `Female`, or `All` | `All` |
| `eligible_education_levels` | string | Comma-separated levels from: `Class 8, Class 10, Class 12, Undergraduate, Postgraduate, PhD` | `Class 8, Class 10` |
| `eligible_disability` | string | `Yes` (disability required), `No` (non-disabled only), `Preferred` (bonus for disabled), or `All` | `All` |
| `eligible_minority` | string | Same values as `eligible_disability` | `All` |
| `award_amount` | string | Human-readable award description | `₹1,000/month for day scholars` |
| `application_deadline` | string | Deadline or `Rolling` or `October 31 annually` | `October 31 annually` |
| `description_text` | string | Full scheme description (2–5 sentences). **This is the most important column — more detail = better RAG retrieval.** | `The Pre-Matric Scholarship Scheme for SC students covers Classes 8–10 ...` |
| `source_url` | string | Official scheme URL | `https://scholarships.gov.in/...` |

### Sample row

```csv
scheme_id,scheme_name,administering_body,level,state,eligible_categories,min_income_limit,max_income_limit,eligible_gender,eligible_education_levels,eligible_disability,eligible_minority,award_amount,application_deadline,description_text,source_url
NSP_PRE_MAT_SC_2024,"Pre-Matric Scholarship for SC Students","Ministry of Social Justice and Empowerment",Central,,SC,0,200000,All,"Class 8, Class 10",All,All,"Day scholar: ₹150/month + ₹750 ad hoc; Hosteller: ₹350/month + ₹1000 ad hoc","October 31 annually","The Pre-Matric Scholarship Scheme is for SC students studying in Classes 8 and 10 in government or government-aided schools. The family income must not exceed ₹2 lakh per annum. Awards cover monthly maintenance and ad hoc grants for books and stationery. Applications are submitted on the National Scholarship Portal (NSP).","https://scholarships.gov.in/public/schemeGuidelines/NSP_Guideline.pdf"
```

### Where to source scheme data

The following official sources have machine-readable or easily scrapable data:

| Source | URL | Notes |
|--------|-----|-------|
| National Scholarship Portal | scholarships.gov.in | Most central schemes listed with eligibility criteria |
| MyScheme portal | myscheme.gov.in | Has search filters by category, state, income — good for bulk export |
| State scholarship portals | e.g. mahadbcmahait.gov.in (MH), prerana.karnataka.gov.in (KA) | State-specific schemes |
| PM Yasasvi Scheme | yet.nta.ac.in | OBC, EWS, DNT scholarship details |
| Aicte portals | aicte-india.org | Technical education scholarships |

**Practical approach for a hackathon:**
1. Go to [myscheme.gov.in](https://www.myscheme.gov.in) and filter by
   category (SC/ST/OBC/EWS), scheme type (scholarship), and level (Central + major states).
2. Export or manually copy 50–200 schemes into a CSV.
3. For `description_text`, paste the eligibility section from the scheme's official page —
   the richer this text, the better the RAG retrieval quality.

### Dataset validation checklist

Before running the ingest notebook, verify:

- [ ] The CSV has all 16 columns (exact names, no extra spaces)
- [ ] `scheme_id` values are unique across all rows
- [ ] `level` is exactly `Central` or `State` (capital C/S)
- [ ] `max_income_limit` is a number (no ₹ symbol, no commas)
- [ ] `eligible_categories` uses comma-separated values from: `SC`, `ST`, `OBC`, `General`, `EWS`, `All`
- [ ] `eligible_education_levels` uses values from: `Class 8`, `Class 10`, `Class 12`, `Undergraduate`, `Postgraduate`, `PhD`
- [ ] `description_text` is non-empty for every row (empty descriptions produce bad embeddings)
- [ ] No `\n` newlines inside cell values (use `|` or semicolons instead)
- [ ] File is UTF-8 encoded (important for ₹ symbol and Indian names)

Quick validation command (local):

```bash
python - <<'EOF'
import pandas as pd, sys

df = pd.read_csv("your_schemes.csv")

REQUIRED = [
    "scheme_id","scheme_name","administering_body","level","state",
    "eligible_categories","min_income_limit","max_income_limit",
    "eligible_gender","eligible_education_levels","eligible_disability",
    "eligible_minority","award_amount","application_deadline",
    "description_text","source_url",
]
missing_cols = [c for c in REQUIRED if c not in df.columns]
if missing_cols:
    print("MISSING COLUMNS:", missing_cols); sys.exit(1)

dupes = df[df.duplicated("scheme_id")]
if not dupes.empty:
    print("DUPLICATE scheme_id:", dupes["scheme_id"].tolist()); sys.exit(1)

empty_desc = df[df["description_text"].fillna("").str.strip() == ""]
if not empty_desc.empty:
    print("EMPTY description_text in rows:", empty_desc.index.tolist()); sys.exit(1)

print(f"OK — {len(df)} schemes, all checks passed")
EOF
```

---

## Setup: step by step

### Step 1 — Prerequisites

Install these tools before starting:

```bash
# Python 3.10+
python --version   # should print 3.10.x or higher

# Databricks CLI v0.200+ (the new Go-based CLI)
pip install databricks-cli   # or: brew install databricks on macOS
databricks --version

# Git
git --version
```

You also need:
- A **Databricks Free Edition** account at [databricks.com/try-databricks](https://www.databricks.com/try-databricks)
  — Unity Catalog is enabled by default on new workspaces.
- A **Sarvam AI** API key from [dashboard.sarvam.ai](https://dashboard.sarvam.ai)
  (free tier available — needed for translation and TTS).

---

### Step 2 — Clone the repo

```bash
git clone <your-repo-url>
cd scholarship_schemes_hackathon
```

---

### Step 3 — Databricks authentication

```bash
# Log in and save a profile named "scholarship-free"
databricks auth login \
  --host https://<your-workspace>.cloud.databricks.com \
  --profile scholarship-free

# Verify
databricks auth env --profile scholarship-free

# Set as default for this shell session
export DATABRICKS_CONFIG_PROFILE=scholarship-free
```

Your workspace URL is the one you see in the browser after logging in to Databricks,
e.g. `https://dbc-abc12345-6789.cloud.databricks.com`.

---

### Step 4 — Unity Catalog setup

Open a new **Databricks notebook** (Workspace → New → Notebook, attach to any cluster)
and run:

```sql
-- Create the schema (run once)
CREATE SCHEMA IF NOT EXISTS main.scholarships
  COMMENT 'Scholarship scheme corpus and RAG index';

-- Create the volume for raw CSV uploads
CREATE VOLUME IF NOT EXISTS main.scholarships.raw
  COMMENT 'Raw scholarship CSV files';

-- Create the volume for the FAISS index output
CREATE VOLUME IF NOT EXISTS main.scholarships.rag
  COMMENT 'FAISS index and metadata parquet for RAG';
```

Verify in the UI: **Catalog** → `main` → `scholarships` — you should see `raw` and `rag` volumes.

---

### Step 5 — Upload your dataset

Upload your validated CSV file(s) to the `raw` volume.

**Option A — Databricks UI (easiest):**
1. Catalog → main → scholarships → raw
2. Click **Upload to this volume**
3. Select your CSV file(s)

**Option B — Databricks CLI:**
```bash
databricks fs cp path/to/your_schemes.csv \
  /Volumes/main/scholarships/raw/schemes.csv \
  --profile scholarship-free
```

**Option C — Multiple files:**
If you have separate CSVs per state or category, upload them all — the ingest
notebook will concatenate them automatically:

```bash
for f in data/*.csv; do
  databricks fs cp "$f" "/Volumes/main/scholarships/raw/$(basename $f)" \
    --profile scholarship-free
done
```

Verify the upload:
```bash
databricks fs ls /Volumes/main/scholarships/raw/ --profile scholarship-free
```

---

### Step 6 — Store secrets

The app reads the Sarvam API key from a Databricks secret scope at startup.
This keeps the key out of `app.yaml` and out of the container environment.

```bash
# Create the scope (only needed once)
databricks secrets create-scope scholarship --profile scholarship-free

# Store the Sarvam API key
databricks secrets put-secret scholarship sarvam_api_key --profile scholarship-free
# → You will be prompted to enter the key value (it will not echo)

# Verify (shows key name only, not the value)
databricks secrets list --scope scholarship --profile scholarship-free
```

Expected output:
```
Key name          Last updated
----------------  ---------------
sarvam_api_key    2024-xx-xx ...
```

---

### Step 7 — Get your LLM endpoint URL

The app calls the Databricks AI Gateway for LLM completions.

1. Databricks UI → **AI Gateway** (left sidebar)
2. Find a serving endpoint (e.g. `databricks-meta-llama-3-1-70b-instruct`)
3. Click the endpoint → **Get code** button
4. Copy the `base_url` value — it looks like:
   ```
   https://7474650313055161.ai-gateway.cloud.databricks.com/mlflow/v1
   ```

If you don't see AI Gateway yet:
- Go to **Settings** → **Feature enablement** → enable **AI Gateway**
- Or use a **Model Serving** endpoint instead — the URL format is the same

Note the model name shown in the **Get code** dialog (e.g. `databricks-llama-4-maverick`
or `databricks-meta-llama-3-1-70b-instruct`).

---

### Step 8 — Configure app.yaml

Edit `app.yaml` in the repo root and fill in the two values you found in Step 7:

```yaml
command: ["python", "app/main.py"]

env:
  - name: "LLM_OPENAI_BASE_URL"
    value: "https://<your-workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1"
                                   # ↑ paste your base_url here

  - name: "LLM_MODEL"
    value: "databricks-meta-llama-3-1-70b-instruct"
                                   # ↑ paste your model name here

  - name: "FAISS_INDEX_PATH"
    value: "/Volumes/main/scholarships/rag/faiss.index"

  - name: "FAISS_META_PATH"
    value: "/Volumes/main/scholarships/rag/metadata.parquet"

secrets:
  - scope: "scholarship"
    key: "sarvam_api_key"
    env_var: "SARVAM_API_KEY"
```

Commit and push this change:
```bash
git add app.yaml
git commit -m "configure LLM endpoint"
git push
```

---

### Step 9 — Run the ingest notebook

This notebook reads your CSVs from the `raw` volume, builds the `text` embedding
column, and writes everything to a Delta table.

1. Databricks UI → **Workspace** → navigate to your cloned repo
2. Open `notebooks/scholarship_ingest.ipynb`
3. Attach to any cluster (a single-node cluster is fine)
4. **Run All**

What it does:
- Reads all `*.csv` files from `/Volumes/main/scholarships/raw/`
- Validates that required columns are present
- Builds a rich `text` column for each scheme by concatenating all fields:
  ```
  Scheme: <name>. Administered by: <body> (<level>). Eligible categories: <cats>.
  Income limit: below ₹<max>. Education: <levels>. Gender: <gender>.
  Disability: <disability>. Minority: <minority>. Award: <amount>.
  Deadline: <deadline>. Details: <description_text>
  ```
- Writes `main.scholarships.scheme_corpus` as a Delta table (overwrites if re-run)

**Expected output:**
```
Found 1 CSV file(s): /Volumes/main/scholarships/raw/schemes.csv
Loaded 150 rows from 1 file(s)
✅ Written 150 rows to main.scholarships.scheme_corpus
Row count in Delta table: 150
✅ Ingest complete — ready for build_rag_index.ipynb
```

If you see column warnings, fix the CSV and re-run. Re-runs are safe (overwrites).

---

### Step 10 — Build the FAISS index

This notebook reads the Delta table, embeds every scheme's `text` column with
`sentence-transformers/all-MiniLM-L6-v2`, builds a FAISS `IndexFlatIP`, and saves
the index and a metadata parquet to the `rag` volume.

1. Open `notebooks/build_rag_index.ipynb`
2. **Edit the first cell:** set `REPO_ROOT` to your Workspace Repos path:
   ```python
   REPO_ROOT = "/Workspace/Users/your@email.com/scholarship_schemes_hackathon"
   ```
   To find this path: right-click your repo folder in the Workspace sidebar → **Copy path**.
3. Attach to a cluster and **Run All**

This step downloads the `all-MiniLM-L6-v2` model (~90 MB) on first run. This can take
2–3 minutes. Subsequent runs use the Hugging Face cache and are much faster.

**Expected output:**
```
✅ pip: numpy 1.x, pandas<3, faiss-cpu 1.7.x, pyarrow, sentence-transformers
✅ import faiss OK
Loaded 150 rows from main.scholarships.scheme_corpus
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Encoding 150 texts...
Embeddings shape: (150, 384)
FAISS index: 150 vectors, dim=384
✅ FAISS index saved → /Volumes/main/scholarships/rag/faiss.index
✅ Metadata saved → /Volumes/main/scholarships/rag/metadata.parquet (150 rows)
Top 5 results:  [smoke test output]
✅ RAG index build complete — ready for Databricks Apps deployment
```

Verify the output files exist:
```bash
databricks fs ls /Volumes/main/scholarships/rag/ --profile scholarship-free
# Expected:
#   faiss.index
#   metadata.parquet
```

---

### Step 11 — Deploy the app

1. Databricks UI → **Compute** → **Apps** → **Create App**
2. Select **"Connect to a Git repository"**
3. Enter your repository URL and click **Connect**
4. The entry point is already configured in `app.yaml` — leave it as-is
5. Click **Deploy**

Databricks will:
- Clone your repo into the app container
- Run `pip install -r requirements.txt` (installs Gradio, FAISS, sentence-transformers, etc.)
- Start the app with `python app/main.py`

The first deploy takes ~3–5 minutes (model download + index load). Subsequent deploys
are faster.

Once deployed, Databricks shows you the app URL. Open it in your browser.

---

### Step 12 — Grant permissions

The app runs under a **Databricks service principal**. You can find its ID in:
Apps → your app → **Service Principal**.

Grant it the following permissions:

**AI Gateway endpoint (LLM calls):**
- AI Gateway → your endpoint → **Permissions** → Add → `<service-principal>` → **CAN_QUERY**

**UC Volume (FAISS index read at runtime):**
```sql
GRANT READ VOLUME ON VOLUME main.scholarships.rag TO `<service-principal-id>`;
GRANT USAGE ON SCHEMA main.scholarships TO `<service-principal-id>`;
```

**Secret scope (Sarvam API key):**
```bash
databricks secrets put-acl scholarship <service-principal-id> READ \
  --profile scholarship-free
```

After granting permissions, **restart the app** (Apps → your app → Restart) to ensure
the secrets are loaded fresh.

---

## Local development

For testing the app on your machine before deploying to Databricks:

```bash
# 1. Install all dependencies
pip install -e ".[dev,rag,rag_embed,app]"

# 2. Create your .env file
cp .env.example .env
```

Edit `.env` with your actual values:

```bash
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
DATABRICKS_TOKEN=<your-pat>              # Settings → Developer → Access tokens → Generate new
SARVAM_API_KEY=<your-sarvam-key>
LLM_OPENAI_BASE_URL=https://<workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1
LLM_MODEL=databricks-meta-llama-3-1-70b-instruct
FAISS_INDEX_PATH=/tmp/scholarship_index/faiss.index
FAISS_META_PATH=/tmp/scholarship_index/metadata.parquet
LOG_LEVEL=INFO
```

For `FAISS_INDEX_PATH` and `FAISS_META_PATH`, either:
- **Option A:** download the index files after running the notebooks:
  ```bash
  mkdir -p /tmp/scholarship_index
  databricks fs cp /Volumes/main/scholarships/rag/faiss.index \
    /tmp/scholarship_index/faiss.index
  databricks fs cp /Volumes/main/scholarships/rag/metadata.parquet \
    /tmp/scholarship_index/metadata.parquet
  ```
- **Option B:** re-run `build_rag_index.ipynb` locally (requires a Spark environment)
  — not recommended; use Databricks notebooks instead.

```bash
# 3. Load env vars and start the app
export $(grep -v '^#' .env | xargs)
python app/main.py
```

Open `http://localhost:7860` in your browser.

---

## Testing

```bash
# Install test + RAG dependencies
pip install -e ".[dev,rag,rag_embed]"

# Run all tests
pytest tests/ -v

# Skip tests that need faiss installed
pytest tests/ -v -k "not FAISS"
```

The test suite covers:
- `FAISSRetriever` loads a mock index without error
- `search()` returns correct columns and score range
- `profile_to_query()` produces a non-empty English query string
- `get_retriever()` does not raise at construction time even if index files are missing

---

## Troubleshooting

**Ingest notebook: `No CSV files found at /Volumes/main/scholarships/raw/`**
→ Upload at least one CSV first (Step 5). Check the volume path is exact.

**Ingest notebook: `WARNING: missing columns: [...]`**
→ Your CSV is missing required columns. Compare your header row against the schema table above.
The ingest will still run but missing columns will be blank in the `text` embedding — this
degrades retrieval quality.

**Build index notebook: `ModuleNotFoundError: No module named 'faiss'`**
→ The `pip install` cell failed silently. Re-run the first cell individually and check for errors.
Make sure your cluster has internet access.

**App won't start: `RuntimeError: Set LLM_CHAT_COMPLETIONS_URL or LLM_OPENAI_BASE_URL`**
→ `app.yaml` `LLM_OPENAI_BASE_URL` is still the placeholder value. Edit it with your actual
AI Gateway URL (Step 7–8) and redeploy.

**Results always in English even when another language is selected**
→ `SARVAM_API_KEY` is not set or is invalid. Check:
```bash
databricks secrets list --scope scholarship
```
Then restart the app.

**LLM returns `401 Unauthorized`**
→ The app's service principal is not granted `CAN_QUERY` on the AI Gateway endpoint (Step 12).

**FAISS index not found at runtime**
→ Confirm `build_rag_index.ipynb` ran successfully and the files exist:
```bash
databricks fs ls /Volumes/main/scholarships/rag/
```
Both `faiss.index` and `metadata.parquet` must be present. The app's service principal
also needs `READ VOLUME` on `main.scholarships.rag`.

**Poor retrieval quality (wrong schemes returned)**
→ The `description_text` column is the most important for RAG quality. Go back to your CSV
and add richer eligibility descriptions — 3–5 sentences per scheme covering category,
income, education, state, and any special conditions.

**`gradio_client` crash on startup (api_info error)**
→ Do not remove the monkey-patch block at the top of `app/main.py`. This is a known
bug in `gradio-client==1.3.0` with Gradio 4.44's Chatbot schema. The patch is intentional.

---

## Data pipeline summary

```
your_schemes.csv  (16 columns, UTF-8)
  │
  │  [Step 5] databricks fs cp → /Volumes/main/scholarships/raw/
  │
  ▼
scholarship_ingest.ipynb
  ├── reads *.csv from raw volume
  ├── builds text column (concatenated fields)
  └── writes → main.scholarships.scheme_corpus  (Delta table)
  │
  ▼
build_rag_index.ipynb
  ├── reads scheme_corpus table
  ├── embeds text with all-MiniLM-L6-v2
  ├── L2-normalises → IndexFlatIP (cosine similarity)
  ├── writes → /Volumes/main/scholarships/rag/faiss.index
  └── writes → /Volumes/main/scholarships/rag/metadata.parquet
  │
  ▼
app/main.py  (Gradio on Databricks Apps)
  ├── FAISSRetriever loads index + parquet at first request
  ├── profile_to_query() → English query sentence
  ├── retriever.search(query, k=7) → top scheme chunks
  ├── LLM generates eligibility list
  └── Sarvam translates → user's language + TTS
```
