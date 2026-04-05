# Scholarship Finder — Pipeline Explained

## Overview

The app answers one question: **"Which Indian government scholarships am I eligible for?"**

A user fills in a profile form (state, category, income, education, etc.). The app turns
that profile into an English query, retrieves the most relevant scholarship scheme documents
from a FAISS vector index, sends them to an LLM, and returns a numbered eligibility list —
optionally translated into 13 Indian languages with text-to-speech.

---

## End-to-End Pipeline

```
CSV files (scheme data)
  │
  ▼  [notebook: scholarship_ingest.ipynb]
Delta table: main.scholarships.scheme_corpus
  │
  ▼  [notebook: build_rag_index.ipynb]
FAISS index + metadata.parquet   (stored in UC Volume)
  │
  ▼  [app: app/main.py — Gradio on Databricks Apps]
User fills profile form
  │
  ▼
profile_to_query()  →  English query string
  │
  ▼
FAISSRetriever.search(query, k=7)  →  top-7 scheme chunks
  │
  ▼
Databricks AI Gateway (Llama LLM)  →  eligibility list in English
  │
  ▼
Sarvam Mayura translate (if non-English)  →  bilingual response
  │
  ▼
Gradio Chatbot  →  user sees results
  │
  ▼ (optional)
Sarvam Bulbul TTS  →  audio playback
```

---

## Stage 1: Data Ingest (`notebooks/scholarship_ingest.ipynb`)

**Input:** One or more CSV files uploaded to UC Volume `/Volumes/main/scholarships/raw/`

**What it does:**

Each CSV row describes one scholarship scheme with 16 required columns:
`scheme_id`, `scheme_name`, `administering_body`, `level` (Central/State), `state`,
`eligible_categories`, `min_income_limit`, `max_income_limit`, `eligible_gender`,
`eligible_education_levels`, `eligible_disability`, `eligible_minority`, `award_amount`,
`application_deadline`, `description_text`, `source_url`.

The notebook:
1. Reads all `*.csv` files from the raw volume, validates column presence.
2. Builds a rich `text` column per row by concatenating all fields into a single sentence:
   ```
   Scheme: <name>. Administered by: <body> (<level>). Eligible categories: <cats>.
   Income limit: below ₹<max>. Education: <levels>. Gender: <gender>.
   Disability: <disability>. Minority: <minority>. Award: <amount>.
   Deadline: <deadline>. Details: <description_text>
   ```
   This single `text` string is what gets embedded — richer text = better retrieval.
3. Writes the Pandas DataFrame as a Spark Delta table:
   `main.scholarships.scheme_corpus` (overwrite-safe).

**Output:** Delta table with all scheme rows plus the `text` embedding column.

---

## Stage 2: Build FAISS Index (`notebooks/build_rag_index.ipynb`)

**Input:** Delta table `main.scholarships.scheme_corpus`

**What it does:**

1. Reads the `scheme_id`, `scheme_name`, and `text` columns from the Delta table into Pandas.
2. Loads the embedding model `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors).
3. Encodes every scheme's `text` column with L2-normalization (`normalize_embeddings=True`).
   L2-normalizing means an inner-product search (`IndexFlatIP`) is mathematically equivalent
   to cosine similarity.
4. Adds all embedding vectors to a `faiss.IndexFlatIP` index.
5. Saves two files to UC Volume `/Volumes/main/scholarships/rag/`:
   - `faiss.index` — the FAISS binary index (all vectors)
   - `metadata.parquet` — lightweight lookup table (`scheme_id`, `scheme_name`, `text`)

**Output:** Two files on disk, one query away from serving results.

---

## Stage 3: Retrieval — `FAISSRetriever` (`src/scholarship/retriever.py`)

`FAISSRetriever` is a lazy-loading class. Nothing is loaded until the first `search()` call.

**On first `search()` call:**
- Loads `faiss.index` with `faiss.read_index()`
- Loads `metadata.parquet` with `pd.read_parquet()`
- Loads the `all-MiniLM-L6-v2` model from HuggingFace (or local cache)

**On every `search(query, k=7)` call:**
1. Encodes the query string into a 384-dim float32 vector (L2-normalized).
2. Calls `index.search(emb, k)` → returns distances and row indices of top-k matches.
3. Looks up each index in the metadata parquet to get `scheme_id`, `scheme_name`, `text`.
4. Returns a `pd.DataFrame` with columns: `scheme_id`, `scheme_name`, `text`, `score`, `rank`.

**Path resolution:** At startup, `get_retriever()` checks `FAISS_INDEX_PATH` and
`FAISS_META_PATH` env vars. If the paths point to a UC Volume not mounted locally,
the Databricks SDK is used to download both files to `/tmp/scholarship_index/` and
they are cached there for subsequent requests.

---

## Stage 4: Profile → Query (`app/main.py: profile_to_query()`)

The Gradio form collects: state, category (SC/ST/OBC/General/EWS), annual income,
gender, age, education level, disability flag, minority flag.

`profile_to_query()` serializes these into a single English sentence:

```
"I am a 22-year-old Female student from Maharashtra.
My category is SC. My annual family income is ₹1,20,000.
I am currently studying at Undergraduate level.
Which scholarship schemes am I eligible for?"
```

This sentence is passed directly to the FAISS retriever as the search query. The query
is always constructed in English regardless of the UI language selected, because the
scheme corpus is indexed in English.

---

## Stage 5: LLM Generation (`src/scholarship/llm_client.py`)

**System prompt:**
```
You are a scholarship eligibility assistant for India.
Given a user's profile and retrieved scholarship scheme information,
list only the schemes the user appears eligible for.
For each scheme, explain in one line why they match, state the award amount
if available, and note the application deadline if available.
Format as a numbered list. Be factual and concise.
Do not invent eligibility criteria not present in the context.
```

**User message** (`rag_user_message()`):
```
Context:
<chunk 1 text>

<chunk 2 text>
...

Question: <profile query>
```

The 7 retrieved scheme chunks are injected as context. The LLM (Databricks AI Gateway →
Llama, configurable via `LLM_MODEL`) generates a numbered eligibility list grounded in
the retrieved context. Temperature is 0.2 for factual, low-variance responses.

**Auth:** The client tries, in order: `DATABRICKS_TOKEN` env var → `LLM_API_KEY` →
`OPENAI_API_KEY` → Databricks SDK OAuth M2M (for service principal on Databricks Apps).

---

## Stage 6: Translation & TTS (`src/scholarship/sarvam_client.py`)

If the user selected a non-English language, the English LLM response is translated
using **Sarvam Mayura** (`POST /translate`).

Long responses are chunked at paragraph boundaries (500-char limit per API call) to
stay within Sarvam's per-request limit. Each chunk is translated independently and
the results are rejoined with newlines.

The final response shown in the chat is **bilingual**: translated language on top,
English below, followed by the matched scheme names (citations) and a disclaimer.

**TTS (optional):** If the user enables "Read results aloud", the translated portion
of the response is passed to **Sarvam Bulbul** (`POST /text-to-speech`). Markdown
is stripped first (`strip_markdown_for_tts()`). The returned WAV bytes are decoded
to a `(sample_rate, float32 numpy array)` tuple for Gradio's Audio component.

**Supported languages:** English, Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati,
Kannada, Malayalam, Punjabi, Odia, Urdu, Assamese (13 total).

---

## App Framework (`app/main.py`)

The UI is built with **Gradio 4.44** on top of `gr.Blocks`:

- **Panel 1 (form):** Language selector, state dropdown (all 28 states + 8 UTs),
  category radio, income number, gender, age, education level, disability and minority
  checkboxes, and a "Find Scholarships" button.
- **Panel 2 (results):** A `gr.Chatbot` showing the bilingual response, an optional
  audio player for TTS, and a "Search Again" button that flips back to Panel 1.

The two panels are toggled via `gr.update(visible=...)` — no page navigation needed.

**Gradio-client monkey-patch:** A known bug in `gradio-client==1.3.0` causes
`get_api_info()` to crash on Chatbot schemas where `additionalProperties` is `True`
(a bool, not a dict). The patch at the top of `main.py` guards the two recursive
functions in `gradio_client.utils` so they return `"Any"` instead of crashing when
encountering a non-dict schema node.

**Deployment (Databricks Apps):** `app.yaml` specifies `python app/main.py` as the
entrypoint and injects env vars (`LLM_OPENAI_BASE_URL`, `LLM_MODEL`, `FAISS_INDEX_PATH`,
`FAISS_META_PATH`) plus the Sarvam API key from a Databricks secret scope. The app's
service principal needs `CAN_QUERY` on the AI Gateway endpoint, `READ VOLUME` on
`main.scholarships.rag`, and `READ` on the `scholarship` secret scope.

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, HuggingFace) |
| Vector index | FAISS `IndexFlatIP` (cosine similarity via L2-normalized vectors) |
| LLM | Databricks AI Gateway → Llama (configurable via `LLM_MODEL`) |
| Translation | Sarvam Mayura (`POST /translate`) |
| Text-to-speech | Sarvam Bulbul v3 (`POST /text-to-speech`) |
| Data platform | Databricks Free Edition — Unity Catalog, Volumes, Delta tables, Apps |
| UI framework | Gradio 4.44 |
| Language | Python 3.10+ |

---

## Key Design Decisions

**Why FAISS IndexFlatIP with L2-normalized vectors?**
Cosine similarity is the right metric for semantic search over short texts. L2-normalizing
at embed time and using inner-product search is mathematically identical to cosine similarity
but FAISS's `IndexFlatIP` is simpler and faster than `IndexFlatL2` with explicit normalization.

**Why is the query always in English?**
The scheme corpus was indexed in English. Embedding a Hindi query against English vectors
would degrade retrieval. The profile fields (state, category, income) are structured data
with unambiguous English representations, so `profile_to_query()` always builds in English.

**Why top-7 chunks?**
7 schemes fit comfortably in an LLM context window while giving enough diversity that the
model can filter down to truly eligible ones. Sending fewer chunks risks missing relevant
schemes; sending many risks hitting token limits and diluting the prompt.

**Why chunk translation at 500 chars?**
Sarvam's `/translate` endpoint has a per-request character limit. 500 chars is a safe
bound that keeps each chunk within a single paragraph, preserving natural sentence
boundaries and producing coherent translations.

**Why lazy-load the retriever?**
FAISS index load + model load takes several seconds. Doing it at module import time would
delay Gradio startup. Lazy loading means the app starts immediately and the one-time cost
is paid on the first user request.
