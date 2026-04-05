# Developer Guide

Full setup, deployment, and development reference for the Scholarship Finder app.

## Prerequisites

- Databricks Free Edition workspace (Unity Catalog enabled)
- Databricks CLI installed and authenticated
- Python 3.10+
- Git

---

## 1. Authenticate with Databricks

```bash
databricks auth login \
  --host https://<your-workspace>.cloud.databricks.com \
  --profile scholarship-free

export DATABRICKS_CONFIG_PROFILE=scholarship-free
```

---

## 2. Create Unity Catalog schema and volumes

Run the following in a Databricks notebook (Workspace → New Notebook → attach to any cluster):

```sql
-- Create schema
CREATE SCHEMA IF NOT EXISTS main.scholarships;

-- Create volumes
CREATE VOLUME IF NOT EXISTS main.scholarships.rag;
CREATE VOLUME IF NOT EXISTS main.scholarships.raw;
```

Or via Databricks CLI:

```bash
databricks unity-catalog schemas create --catalog main --name scholarships
```

---

## 3. Upload scholarship CSVs

Upload your scholarship scheme CSV files to:
```
/Volumes/main/scholarships/raw/
```

Using the Databricks UI: Catalog → main → scholarships → raw → Upload.

Using the CLI:
```bash
databricks fs cp your_schemes.csv \
  dbfs:/Volumes/main/scholarships/raw/your_schemes.csv
```

**Required CSV columns:**

| Column | Description |
|--------|-------------|
| `scheme_id` | Unique identifier |
| `scheme_name` | Full name of the scheme |
| `administering_body` | Ministry / department name |
| `level` | `Central` or `State` |
| `state` | State name (for state-level schemes; blank for central) |
| `eligible_categories` | e.g. `SC, ST` or `All` |
| `min_income_limit` | Minimum annual family income (INR; 0 if no lower bound) |
| `max_income_limit` | Maximum annual family income (INR) |
| `eligible_gender` | `Male`, `Female`, `All` |
| `eligible_education_levels` | e.g. `Class 10, Class 12, Undergraduate` |
| `eligible_disability` | `Yes`, `No`, `Preferred` |
| `eligible_minority` | `Yes`, `No`, `Preferred` |
| `award_amount` | Award description, e.g. `₹12,000/year` |
| `application_deadline` | Deadline description or `Rolling` |
| `description_text` | Full scheme description for RAG context |
| `source_url` | Official scheme URL |

---

## 4. Store secrets

```bash
# Create scope (only needs to be done once)
databricks secrets create-scope scholarship

# Store Sarvam API key
databricks secrets put-secret scholarship sarvam_api_key
# → enter your key when prompted
```

---

## 5. Run notebooks on a Databricks cluster

Run in order, on any cluster with internet access:

1. **`notebooks/scholarship_ingest.ipynb`**
   - Reads CSVs from `/Volumes/main/scholarships/raw/`
   - Builds `text` embedding column
   - Writes to Delta table `main.scholarships.scheme_corpus`

2. **`notebooks/build_rag_index.ipynb`**
   - Reads from `main.scholarships.scheme_corpus`
   - Embeds with `sentence-transformers/all-MiniLM-L6-v2`
   - Saves `faiss.index` + `metadata.parquet` to `/Volumes/main/scholarships/rag/`

> Edit `REPO_ROOT` in the first cell of each notebook to match your Workspace Repos path.

---

## 6. Configure app.yaml

Edit `app.yaml` in the repo root:

```yaml
env:
  - name: "LLM_OPENAI_BASE_URL"
    value: "https://<your-workspace-id>.ai-gateway.cloud.databricks.com/mlflow/v1"
  - name: "LLM_MODEL"
    value: "databricks-meta-llama-3-1-70b-instruct"   # or databricks-llama-4-maverick
```

Get `LLM_OPENAI_BASE_URL` from: Databricks workspace → AI Gateway → your endpoint → **Get code**.

---

## 7. Deploy the app

1. Databricks UI → **Compute** → **Apps** → **Create App**
2. Select **"Connect to a Git repository"** → enter this repo URL
3. Set the entry point to `app/main.py` (already configured in `app.yaml`)
4. Click **Deploy**

---

## 8. Grant service principal permissions

The Databricks Apps service principal needs:

```bash
# AI Gateway endpoint — CAN_QUERY
# UC Volume main.scholarships.rag — READ
# Secret scope: scholarship — READ
```

In the Databricks UI:
- AI Gateway → your endpoint → Permissions → Add service principal → CAN_QUERY
- Catalog Explorer → main → scholarships → rag (volume) → Permissions → READ
- Settings → Secrets → scholarship scope → Manage permissions → READ

---

## Local development

```bash
# Clone the repo
git clone <repo-url>
cd scholarship_schemes_hackathon

# Install in editable mode with all extras
pip install -e ".[dev,rag,rag_embed,app]"

# Set up environment
cp .env.example .env
# Edit .env with your actual values
export $(grep -v '^#' .env | xargs)

# Run the app locally
python app/main.py
```

Open `http://localhost:7860` in your browser.

---

## Running tests

```bash
pip install -e ".[dev,rag,rag_embed]"
pytest tests/ -v
```

Skip tests requiring faiss if not installed:
```bash
pytest tests/ -v -m "not network"
```

---

## Dependency notes

| Pin | Reason |
|-----|--------|
| `gradio~=4.44.0` | Required for Databricks Apps template compatibility |
| `gradio-client==1.3.0` | Must match Gradio 4.44.x wheel metadata; newer client breaks `api_info` JSON schema parsing |
| `huggingface-hub~=0.35.3` | Pin to avoid `HfFolder` drift when `sentence-transformers` upgrades hub |
| `faiss-cpu>=1.7.0,<1.8` | faiss-cpu 1.8+ needs NumPy 2's `numpy._core`; databricks-connect requires numpy<2 |
| `numpy>=1.24,<2` | Required by faiss-cpu 1.7.x + databricks-connect compatibility |

---

## Troubleshooting

**`gradio_client api_info` crash on startup**
→ The monkey-patch in `app/main.py` handles this. Do not remove it.

**`FAISSRetriever` fails to load on Databricks Apps**
→ Verify that `/Volumes/main/scholarships/rag/faiss.index` exists. Re-run `build_rag_index.ipynb`.

**Translation not working (results in English only)**
→ Check `SARVAM_API_KEY` is set. Run `databricks secrets put-secret scholarship sarvam_api_key`.

**LLM call fails with 401**
→ The service principal needs CAN_QUERY on the AI Gateway endpoint. Check `app.yaml` `LLM_OPENAI_BASE_URL`.
