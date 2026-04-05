# Workspace Setup Guide

Admin-level setup for the Scholarship Finder app on Databricks Free Edition.

---

## Unity Catalog setup

### 1. Create the `main.scholarships` schema

In a Databricks notebook or SQL editor:

```sql
-- Create schema (run once)
CREATE SCHEMA IF NOT EXISTS main.scholarships
  COMMENT 'Scholarship scheme corpus and RAG index for the Scholarship Finder app';
```

### 2. Create volumes

```sql
-- Raw CSV uploads
CREATE VOLUME IF NOT EXISTS main.scholarships.raw
  COMMENT 'Raw scholarship scheme CSV files uploaded before ingest';

-- FAISS index output
CREATE VOLUME IF NOT EXISTS main.scholarships.rag
  COMMENT 'FAISS index and metadata parquet for scholarship RAG';
```

### 3. Grant permissions to the Databricks Apps service principal

Replace `<service-principal-id>` with the ID shown in your Databricks Apps settings
(Apps → your app → Service Principal).

```sql
-- Grant schema usage
GRANT USAGE ON SCHEMA main.scholarships TO `<service-principal-id>`;

-- Grant volume read (for FAISS index at runtime)
GRANT READ VOLUME ON VOLUME main.scholarships.rag TO `<service-principal-id>`;

-- Grant table select (if app ever queries the Delta table directly)
GRANT SELECT ON TABLE main.scholarships.scheme_corpus TO `<service-principal-id>`;
```

Or via Databricks CLI:

```bash
databricks unity-catalog permissions update \
  --securable-type schema \
  --full-name main.scholarships \
  --json '{"changes": [{"principal": "<service-principal-id>", "add": ["USE SCHEMA"]}]}'
```

---

## Secret scope setup

### 1. Create secret scope

```bash
databricks secrets create-scope scholarship
```

### 2. Store secrets

```bash
# Sarvam AI API key (for translation, STT, TTS)
databricks secrets put-secret scholarship sarvam_api_key
# → enter your key when prompted
```

### 3. Grant secret scope READ to service principal

```bash
databricks secrets put-acl scholarship <service-principal-id> READ
```

Verify:
```bash
databricks secrets list-acls scholarship
```

---

## AI Gateway endpoint permissions

1. Databricks UI → **AI Gateway** → select your endpoint (e.g. `databricks-meta-llama-3-1-70b-instruct`)
2. **Permissions** tab → **Add**
3. Search for your app's service principal → grant **CAN_QUERY**

---

## Git repository setup (Databricks Repos)

1. Databricks UI → **Workspace** → **Repos** → **Add Repo**
2. Enter your Git repository URL
3. Clone into: `/Workspace/Users/<your-email>/scholarship_schemes_hackathon`
4. This path is what you set as `REPO_ROOT` in the notebooks

---

## Key rotation

To rotate the Sarvam API key:

```bash
databricks secrets put-secret scholarship sarvam_api_key
# → enter new key

# Restart the Databricks App to pick up the new secret:
# Apps → your app → Restart
```

---

## Verify the setup

Run this checklist before deploying the app:

```bash
# 1. Confirm schema exists
databricks unity-catalog schemas get --full-name main.scholarships

# 2. Confirm volumes exist
databricks unity-catalog volumes list --catalog main --schema scholarships

# 3. Confirm secret exists
databricks secrets list --scope scholarship

# 4. Check files in RAG volume (after running notebooks)
databricks fs ls /Volumes/main/scholarships/rag/
# Expected: faiss.index  metadata.parquet
```

---

## Resource summary

| Resource | Path / Name |
|----------|-------------|
| Unity Catalog schema | `main.scholarships` |
| Raw CSV volume | `/Volumes/main/scholarships/raw/` |
| RAG index volume | `/Volumes/main/scholarships/rag/` |
| Delta table | `main.scholarships.scheme_corpus` |
| FAISS index | `/Volumes/main/scholarships/rag/faiss.index` |
| Metadata parquet | `/Volumes/main/scholarships/rag/metadata.parquet` |
| Secret scope | `scholarship` |
| Secret key | `scholarship/sarvam_api_key` |
