"""Gradio entrypoint: scholarship eligibility RAG + Databricks LLM + Sarvam multilingual ."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Repo root on Databricks Repos / local clone
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr
import numpy as np

# ---------- Monkey-patch gradio_client bug (1.3.0 + Gradio 4.44.x) ----------
# get_api_info() crashes on Chatbot schemas where additionalProperties is True
# (a bool).  The internal recursive calls use the module-level name, so we must
# replace the actual function objects in the module namespace.
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_inner = _gc_utils._json_schema_to_python_type
_orig_get_type = _gc_utils.get_type


def _safe_inner(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_inner(schema, defs)


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


# Patch module-level names so internal recursive calls also go through guards.
_gc_utils._json_schema_to_python_type = _safe_inner
_gc_utils.get_type = _safe_get_type
# ---------- End monkey-patch ------------------------------------------------

from scholarship.llm_client import (
    SYSTEM_PROMPT,
    chat_completions,
    extract_assistant_text,
    rag_user_message,
)
from scholarship.retriever import DatabricksVSRetriever, Retriever, get_retriever
from scholarship.sarvam_client import (
    is_configured as sarvam_configured,
    numpy_audio_to_wav_bytes,
    speech_to_text_file,
    strip_markdown_for_tts,
    text_to_speech_wav_bytes,
    transcript_from_stt_response,
    translate_text,
    wav_bytes_to_numpy_float32,
)
from scholarship.user_store import UserStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language config (same 13 languages as nyaya-dhwani)
# ---------------------------------------------------------------------------

SARVAM_LANGUAGES: list[tuple[str, str]] = [
    ("en", "English"),
    ("hi", "Hindi · हिन्दी"),
    ("bn", "Bengali"),
    ("te", "Telugu"),
    ("mr", "Marathi"),
    ("ta", "Tamil"),
    ("gu", "Gujarati"),
    ("kn", "Kannada"),
    ("ml", "Malayalam"),
    ("pa", "Punjabi"),
    ("or", "Odia"),
    ("ur", "Urdu"),
    ("as", "Assamese"),
]

UI_TO_BCP47: dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "or": "od-IN",
    "ur": "hi-IN",
    "as": "bn-IN",
}

DISCLAIMER_EN = (
    "This tool provides general eligibility guidance only. "
    "Always verify directly with the scheme's official portal before applying."
)

# ---------------------------------------------------------------------------
# Indian states and UTs
# ---------------------------------------------------------------------------

INDIAN_STATES_UTS: list[str] = [
    # 28 States
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    # 8 Union Territories
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]


def bcp47_target(lang: str) -> str:
    return UI_TO_BCP47.get(lang, "en-IN")


# ---------------------------------------------------------------------------
# RAG runtime
# ---------------------------------------------------------------------------

class RAGRuntime:
    """Lazy-load retriever (Databricks Vector Search)."""

    def __init__(self) -> None:
        self._retriever: Retriever | None = None

    def load(self) -> None:
        if self._retriever is not None:
            return
        self._retriever = get_retriever()
        logger.info("Retriever loaded: %s", type(self._retriever).__name__)

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            raise RuntimeError("RAGRuntime not loaded")
        return self._retriever


_runtime: RAGRuntime | None = None
_user_store: UserStore | None = None


def get_runtime() -> RAGRuntime:
    global _runtime
    if _runtime is None:
        _runtime = RAGRuntime()
    return _runtime


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store


# ---------------------------------------------------------------------------
# Profile → query
# ---------------------------------------------------------------------------

def profile_to_query(
    state: str,
    category: str,
    income: float,
    gender: str,
    age: float,
    education: str,
    disability: bool,
    minority: bool,
) -> str:
    income_lakh = income / 100_000
    income_lakh_str = (
        f"{int(income_lakh)} lakh" if income_lakh == int(income_lakh)
        else f"{income_lakh:.2f} lakh"
    )
    parts = [
        f"I am a {int(age)}-year-old {gender} student from {state}.",
        f"My category is {category}.",
        f"My annual family income is \u20b9{income_lakh_str} (i.e. \u20b9{int(income):,}).",
        f"I am currently studying at {education} level.",
    ]
    if disability:
        parts.append("I have a disability.")
    if minority:
        parts.append("I belong to a minority community.")
    parts.append("Which scholarship schemes am I eligible for?")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# RAG + LLM
# ---------------------------------------------------------------------------

def _format_citations(chunks_df) -> str:
    lines: list[str] = []
    for _, row in chunks_df.iterrows():
        scheme_name = row.get("scheme_name") or ""
        scheme_id = row.get("scheme_id") or ""
        bits = [str(x).strip() for x in (scheme_name, scheme_id) if x and str(x).strip()]
        if bits:
            lines.append("- " + " · ".join(bits[:2]))
    return "\n".join(lines) if lines else "(no metadata)"


_INELIGIBLE_PATTERNS = [
    "not applicable",
    "not eligible",
    "does not match",
    "do not match",
    "doesn't match",
    "don't match",
    "other schemes",
    "no other scheme",
    "is not open",
    "are not open",
    "cannot apply",
    "you don't qualify",
    "you do not qualify",
    "unfortunately",
]


def _strip_ineligible_lines(text: str) -> str:
    """Remove any lines/sentences the model snuck in about ineligible schemes."""
    import re
    lines = text.splitlines()
    kept = []
    for line in lines:
        low = line.lower()
        if any(pat in low for pat in _INELIGIBLE_PATTERNS):
            continue
        kept.append(line)
    # Drop trailing blank lines
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _rag_answer_english(query_en: str) -> tuple[str, str]:
    """LLM answer in English + citations block."""
    rt = get_runtime()
    rt.load()
    q = query_en.strip()
    chunks_df = rt.retriever.search(q, k=7)
    texts = chunks_df["text"].tolist() if "text" in chunks_df.columns else []
    user_content = rag_user_message([str(t) for t in texts], q)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = chat_completions(messages, max_tokens=2048, temperature=0.2)
    assistant_en = _strip_ineligible_lines(extract_assistant_text(raw))
    cites = _format_citations(chunks_df)
    return assistant_en, cites


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

_TRANSLATE_CHUNK_LIMIT = 500


def _chunked_translate(text: str, *, source: str, target: str) -> str:
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > _TRANSLATE_CHUNK_LIMIT and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)

    translated_parts = []
    for chunk in chunks:
        if not chunk.strip():
            translated_parts.append(chunk)
            continue
        try:
            result = translate_text(chunk, source_language_code=source, target_language_code=target)
            translated_parts.append(result)
        except Exception as e:
            logger.warning("Mayura chunk translate failed, keeping original: %s", e)
            translated_parts.append(chunk)
    return "\n".join(translated_parts)


def _maybe_translate(text: str, *, source: str, target: str) -> str:
    if source == target:
        return text
    if not sarvam_configured():
        return text
    if len(text) > _TRANSLATE_CHUNK_LIMIT:
        return _chunked_translate(text, source=source, target=target)
    try:
        return translate_text(text, source_language_code=source, target_language_code=target)
    except Exception as e:
        logger.warning("Mayura translate failed, using original: %s", e)
        return text


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def build_reply_markdown(assistant_en: str, lang: str) -> str:
    """Build response — English only, or bilingual if a non-English language is selected."""
    if lang == "en" or not sarvam_configured():
        return f"{assistant_en}\n\n---\n*{DISCLAIMER_EN}*"

    tgt = bcp47_target(lang)
    body_translated = _maybe_translate(assistant_en, source="en-IN", target=tgt)
    disc_translated = _maybe_translate(DISCLAIMER_EN, source="en-IN", target=tgt)

    lang_label = dict(SARVAM_LANGUAGES).get(lang, lang)
    return (
        f"**{lang_label}:**\n\n{body_translated}\n\n"
        f"---\n**English:**\n\n{assistant_en}\n\n"
        f"---\n*{disc_translated}*"
    )


def maybe_tts(text_markdown: str, lang: str, enabled: bool) -> tuple[int, np.ndarray] | None:
    if not enabled or not sarvam_configured():
        return None
    import re
    narrative = text_markdown.split("\n---\n", 1)[0]
    narrative = re.sub(r"^\*\*[^*]+:\*\*\s*", "", narrative.strip())
    plain = strip_markdown_for_tts(narrative)
    if not plain.strip():
        return None
    tgt = bcp47_target(lang)
    try:
        wav = text_to_speech_wav_bytes(plain, target_language_code=tgt)
        sr, arr = wav_bytes_to_numpy_float32(wav)
        return (sr, arr)
    except Exception as e:
        logger.warning("TTS failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# User account helpers
# ---------------------------------------------------------------------------

def _build_profile_payload(
    state: str,
    category: str,
    income: float | None,
    gender: str,
    age: float | None,
    education: str,
    disability: bool,
    minority: bool,
) -> dict[str, object]:
    return {
        "state": state,
        "category": category,
        "income": None if income is None else int(income),
        "gender": gender,
        "age": None if age is None else int(age),
        "education": education,
        "disability": bool(disability),
        "minority": bool(minority),
    }


def _profile_field_values(profile: dict[str, object] | None) -> tuple[object, ...]:
    profile = profile or {}
    return (
        profile.get("state") or None,
        profile.get("category") or None,
        profile.get("income"),
        profile.get("gender") or None,
        profile.get("age"),
        profile.get("education") or None,
        bool(profile.get("disability", False)),
        bool(profile.get("minority", False)),
    )


def _user_summary_markdown(user: dict[str, object] | None) -> str:
    if not user:
        return ""

    login_id = str(user.get("login_id") or "")
    full_name = str(user.get("full_name") or login_id)
    email = str(user.get("email") or "Not provided")
    phone = str(user.get("phone") or "Not provided")
    profile = user.get("profile") or {}
    has_saved_profile = any(
        profile.get(key)
        for key in ("state", "category", "income", "gender", "age", "education")
    ) or bool(profile.get("disability")) or bool(profile.get("minority"))

    return (
        f"**Welcome, {full_name}**\n\n"
        f"Login ID: `{login_id}`\n\n"
        f"Email: {email}\n\n"
        f"Phone: {phone}\n\n"
        f"Saved scholarship profile: {'Yes' if has_saved_profile else 'No'}"
    )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    custom_css = """
    .gradio-container {
        background:
            radial-gradient(circle at top right, rgba(255, 204, 102, 0.18), transparent 25%),
            linear-gradient(180deg, #eff5ff 0%, #f8fbff 52%, #eef8f1 100%) !important;
    }
    footer { font-size: 0.85rem; color: #1A3C6E; }
    h1 { color: #0D2B5E; font-family: Georgia, serif; }
    .panel-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #d4e2ff;
        border-radius: 18px;
        box-shadow: 0 16px 42px rgba(20, 61, 122, 0.08);
        padding: 10px;
    }
    .account-card {
        background: #f1f7ff;
        border: 1px solid #c6daff;
        border-radius: 14px;
        padding: 6px 14px;
    }
    .status-card {
        background: #fff8e7;
        border: 1px solid #f0d18a;
        border-radius: 14px;
        padding: 8px 14px;
    }
    @media (prefers-color-scheme: dark) {
        .gradio-container {
            background:
                radial-gradient(circle at top right, rgba(255, 179, 71, 0.16), transparent 25%),
                linear-gradient(180deg, #10203a 0%, #162947 52%, #17332a 100%) !important;
        }
        h1 { color: #d8e4f8; }
        footer { color: #7a9cc8; }
        .panel-card {
            background: rgba(18, 33, 57, 0.9);
            border-color: #28456d;
        }
        .account-card {
            background: #152b4d;
            border-color: #30537f;
        }
        .status-card {
            background: #3a2e16;
            border-color: #7b6331;
        }
    }
    .dark .gradio-container {
        background:
            radial-gradient(circle at top right, rgba(255, 179, 71, 0.16), transparent 25%),
            linear-gradient(180deg, #10203a 0%, #162947 52%, #17332a 100%) !important;
    }
    .dark h1 { color: #d8e4f8; }
    .dark footer { color: #7a9cc8; }
    .dark .panel-card {
        background: rgba(18, 33, 57, 0.9);
        border-color: #28456d;
    }
    .dark .account-card {
        background: #152b4d;
        border-color: #30537f;
    }
    .dark .status-card {
        background: #3a2e16;
        border-color: #7b6331;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
        css=custom_css,
        title="Scholarship Finder India",
    ) as demo:
        gr.Markdown(
            "# Scholarship Finder · छात्रवृत्ति खोजक\n"
            "*Find Indian government scholarship schemes you are eligible for*"
        )

        lang_state = gr.State("en")
        current_user_state = gr.State(None)

        with gr.Column(visible=True, elem_classes=["panel-card"]) as auth_col:
            gr.Markdown(
                "### Sign In or Create Account\n"
                "Save your details once and continue with the same scholarship profile later."
            )
            auth_status = gr.Markdown(value="", visible=False, elem_classes=["status-card"])

            with gr.Tabs():
                with gr.Tab("Sign In"):
                    login_id_tb = gr.Textbox(
                        label="Login ID",
                        placeholder="Enter your email or username",
                    )
                    login_password_tb = gr.Textbox(
                        label="Password",
                        type="password",
                        placeholder="Enter your password",
                    )
                    login_btn = gr.Button("Sign In", variant="primary")

                with gr.Tab("Create Account"):
                    with gr.Row():
                        reg_full_name_tb = gr.Textbox(
                            label="Full Name",
                            placeholder="Enter your name",
                        )
                        reg_login_id_tb = gr.Textbox(
                            label="Login ID",
                            placeholder="Choose a unique email or username",
                        )

                    with gr.Row():
                        reg_email_tb = gr.Textbox(
                            label="Email",
                            placeholder="Optional email address",
                        )
                        reg_phone_tb = gr.Textbox(
                            label="Phone Number",
                            placeholder="Optional contact number",
                        )

                    with gr.Row():
                        reg_password_tb = gr.Textbox(
                            label="Password",
                            type="password",
                            placeholder="Create a password",
                        )
                        reg_confirm_tb = gr.Textbox(
                            label="Confirm Password",
                            type="password",
                            placeholder="Re-enter the password",
                        )

                    register_btn = gr.Button("Create Account", variant="primary")

            gr.Markdown(
                "<small>Your Databricks scholarship logic stays unchanged. "
                "This layer only adds login and saved user/profile details.</small>"
            )

        # ------------------------------------------------------------------
        # Panel 1 — Profile form
        # ------------------------------------------------------------------
        with gr.Column(visible=False, elem_classes=["panel-card"]) as form_col:
            with gr.Row():
                account_md = gr.Markdown(elem_classes=["account-card"])
                logout_btn = gr.Button("Log Out")

            profile_status = gr.Markdown(value="", visible=False, elem_classes=["status-card"])
            gr.Markdown("### Your Profile · आपकी प्रोफ़ाइल")

            lang_radio = gr.Radio(
                choices=[(c[1], c[0]) for c in SARVAM_LANGUAGES],
                value="en",
                label="Select your language / अपनी भाषा चुनें",
                info="Results will be shown in your selected language alongside English.",
            )

            with gr.Row():
                state_dd = gr.Dropdown(
                    choices=INDIAN_STATES_UTS,
                    label="State / Union Territory",
                    value=None,
                )
                category_radio = gr.Radio(
                    choices=["SC", "ST", "OBC", "General", "EWS"],
                    label="Category",
                    value=None,
                )

            with gr.Row():
                income_num = gr.Number(
                    label="Annual Family Income (INR)",
                    minimum=0,
                    precision=0,
                    value=None,
                )
                gender_radio = gr.Radio(
                    choices=["Male", "Female", "Other"],
                    label="Gender",
                    value=None,
                )

            with gr.Row():
                age_num = gr.Number(
                    label="Age",
                    minimum=5,
                    maximum=60,
                    precision=0,
                    value=None,
                )
                edu_radio = gr.Radio(
                    choices=[
                        "Class 8",
                        "Class 10",
                        "Class 12",
                        "Undergraduate",
                        "Postgraduate",
                        "PhD",
                    ],
                    label="Education Level",
                    value=None,
                )

            with gr.Row():
                disability_cb = gr.Checkbox(label="Person with Disability", value=False)
                minority_cb = gr.Checkbox(label="Minority Community", value=False)

            with gr.Row():
                save_profile_btn = gr.Button("Save My Details")
                find_btn = gr.Button("Find Scholarships · छात्रवृत्ति खोजें", variant="primary")

            gr.Markdown(
                "<small>Fill in all fields for best results · "
                "Powered by Databricks (Vector Search RAG + LLM) · Sarvam AI (translation, TTS)</small>"
            )

        # ------------------------------------------------------------------
        # Panel 2 — Results
        # ------------------------------------------------------------------
        with gr.Column(visible=False, elem_classes=["panel-card"]) as results_col:
            with gr.Row():
                result_account_md = gr.Markdown(elem_classes=["account-card"])
                logout_results_btn = gr.Button("Log Out")

            gr.Markdown("### Eligible Schemes · पात्र योजनाएँ")
            current_lang = gr.Markdown("*Language: English*")
            result_banner = gr.Markdown(value="", visible=False, elem_classes=["status-card"])

            loading_md = gr.Markdown(
                "Searching for matching schemes - this may take 10-20 seconds...",
                visible=False,
            )

            chatbot = gr.Chatbot(
                label="Scholarship Eligibility Results",
                height=480,
                bubble_full_width=False,
                visible=False,
            )

            with gr.Row(visible=False) as tts_row:
                tts_cb = gr.Checkbox(label="Read results aloud", value=False)
                tts_out = gr.Audio(
                    label="Listen to results",
                    type="numpy",
                    interactive=False,
                )

            again_btn = gr.Button("Edit Profile / Search Again")

        # ------------------------------------------------------------------
        # Event handlers
        # ------------------------------------------------------------------

        labels = dict(SARVAM_LANGUAGES)
        user_store = get_user_store()

        def _signed_in_view(user, notice: str):
            lang_code = str(user.get("preferred_language") or "en")
            profile_values = _profile_field_values(user.get("profile"))
            return (
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=_user_summary_markdown(user)),
                gr.update(value=notice, visible=bool(notice)),
                user,
                lang_code,
                gr.update(value=lang_code),
                gr.update(value=profile_values[0]),
                gr.update(value=profile_values[1]),
                gr.update(value=profile_values[2]),
                gr.update(value=profile_values[3]),
                gr.update(value=profile_values[4]),
                gr.update(value=profile_values[5]),
                gr.update(value=profile_values[6]),
                gr.update(value=profile_values[7]),
            )

        def _signed_out_view(notice: str):
            return (
                gr.update(value=notice, visible=bool(notice)),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value="", visible=False),
                None,
                "en",
                gr.update(value="en"),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=False),
                gr.update(value=False),
            )

        def on_register(full_name, login_id, email, phone, password, confirm_password, lang_code):
            if password != confirm_password:
                return _signed_out_view("**Sign-up failed:** Passwords do not match.")
            try:
                user = user_store.register_user(
                    login_id,
                    password,
                    full_name=full_name,
                    email=email,
                    phone=phone,
                    preferred_language=lang_code,
                )
            except Exception as exc:
                return _signed_out_view(f"**Sign-up failed:** {exc}")
            return _signed_in_view(
                user,
                "*Account created. Your scholarship profile can now be saved and reused.*",
            )

        def on_login(login_id, password):
            try:
                user = user_store.authenticate_user(login_id, password)
            except Exception as exc:
                return _signed_out_view(f"**Sign-in failed:** {exc}")
            if not user:
                return _signed_out_view(
                    "**Sign-in failed:** Check your Login ID and password and try again."
                )
            return _signed_in_view(
                user,
                "*Signed in successfully. Any saved scholarship details have been loaded.*",
            )

        def on_logout():
            return (
                gr.update(value="*You have been logged out.*", visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value="", visible=False),
                None,
                "en",
                gr.update(value="en"),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=False),
                gr.update(value=False),
                gr.update(visible=False),
                gr.update(visible=False, value=[]),
                gr.update(visible=False),
                None,
                gr.update(value="", visible=False),
                gr.update(value=""),
            )

        def on_save_profile(
            current_user,
            lang_code,
            state,
            category,
            income,
            gender,
            age,
            education,
            disability,
            minority,
        ):
            if not current_user or not current_user.get("login_id"):
                return (
                    current_user,
                    gr.update(),
                    gr.update(value="**Please sign in first.**", visible=True),
                )

            profile = _build_profile_payload(
                state,
                category,
                income,
                gender,
                age,
                education,
                disability,
                minority,
            )

            try:
                updated_user = user_store.save_profile(
                    current_user["login_id"],
                    profile,
                    preferred_language=lang_code,
                )
            except Exception as exc:
                logger.exception("on_save_profile")
                return (
                    current_user,
                    gr.update(),
                    gr.update(value=f"**Could not save your details:** {exc}", visible=True),
                )

            return (
                updated_user,
                gr.update(value=_user_summary_markdown(updated_user)),
                gr.update(value="*Your profile details have been saved.*", visible=True),
            )

        def on_find(
            current_user,
            lang_code,
            state,
            category,
            income,
            gender,
            age,
            education,
            disability,
            minority,
        ):
            """Generator: yields loading state immediately, then final results."""
            if not current_user or not current_user.get("login_id"):
                yield (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False, value=[]),
                    gr.update(visible=False),
                    None,
                    gr.update(),
                    gr.update(value=""),
                    gr.update(value="", visible=False),
                    current_user,
                    gr.update(value="**Please sign in first.**", visible=True),
                    gr.update(),
                )
                return

            missing = []
            if not state:
                missing.append("State")
            if not category:
                missing.append("Category")
            if income is None:
                missing.append("Annual Family Income")
            if not gender:
                missing.append("Gender")
            if age is None:
                missing.append("Age")
            if not education:
                missing.append("Education Level")
            if missing:
                summary = _user_summary_markdown(current_user)
                yield (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False, value=[]),
                    gr.update(visible=False),
                    None,
                    gr.update(),
                    gr.update(value=summary),
                    gr.update(value="", visible=False),
                    current_user,
                    gr.update(
                        value="**Please complete these fields first:** " + ", ".join(missing),
                        visible=True,
                    ),
                    gr.update(value=summary),
                )
                return

            # ── Yield 1: show loading spinner immediately ──────────────────
            summary = _user_summary_markdown(current_user)
            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False, value=[]),
                gr.update(visible=False),
                None,
                gr.update(value=f"*Language: {labels.get(lang_code, lang_code)}*"),
                gr.update(value=summary),
                gr.update(value="", visible=False),
                current_user,
                gr.update(value="", visible=False),
                gr.update(value=summary),
            )

            # ── Do the actual work ─────────────────────────────────────────
            try:
                query_en = profile_to_query(
                    state, category, income, gender, age, education, disability, minority
                )
                assistant_en, cites = _rag_answer_english(query_en)
                reply_md = build_reply_markdown(assistant_en, lang_code)
                history = [[None, reply_md]]
            except Exception as e:
                logger.exception("on_find")
                history = [[None, f"**Error:** {e}"]]

            # ── Yield 2: hide loading, show results ────────────────────────
            yield (
                gr.update(visible=False),                       # keep form hidden
                gr.update(visible=True),                        # keep results panel
                gr.update(visible=False),                       # hide loading message
                gr.update(visible=True, value=history),         # show chatbot with results
                gr.update(visible=True),                        # show tts row
                None,                                           # tts_out
                gr.update(value=f"*Language: {labels.get(lang_code, lang_code)}*"),
                gr.update(value=summary),
                gr.update(value="", visible=False),
                current_user,
                gr.update(value="", visible=False),
                gr.update(value=summary),
            )

        register_btn.click(
            on_register,
            inputs=[
                reg_full_name_tb,
                reg_login_id_tb,
                reg_email_tb,
                reg_phone_tb,
                reg_password_tb,
                reg_confirm_tb,
                lang_state,
            ],
            outputs=[
                auth_status,
                auth_col,
                form_col,
                results_col,
                account_md,
                profile_status,
                current_user_state,
                lang_state,
                lang_radio,
                state_dd,
                category_radio,
                income_num,
                gender_radio,
                age_num,
                edu_radio,
                disability_cb,
                minority_cb,
            ],
        )

        login_btn.click(
            on_login,
            inputs=[login_id_tb, login_password_tb],
            outputs=[
                auth_status,
                auth_col,
                form_col,
                results_col,
                account_md,
                profile_status,
                current_user_state,
                lang_state,
                lang_radio,
                state_dd,
                category_radio,
                income_num,
                gender_radio,
                age_num,
                edu_radio,
                disability_cb,
                minority_cb,
            ],
        )

        logout_btn.click(
            on_logout,
            outputs=[
                auth_status,
                auth_col,
                form_col,
                results_col,
                account_md,
                profile_status,
                current_user_state,
                lang_state,
                lang_radio,
                state_dd,
                category_radio,
                income_num,
                gender_radio,
                age_num,
                edu_radio,
                disability_cb,
                minority_cb,
                loading_md,
                chatbot,
                tts_row,
                tts_out,
                result_banner,
                result_account_md,
            ],
        )

        logout_results_btn.click(
            on_logout,
            outputs=[
                auth_status,
                auth_col,
                form_col,
                results_col,
                account_md,
                profile_status,
                current_user_state,
                lang_state,
                lang_radio,
                state_dd,
                category_radio,
                income_num,
                gender_radio,
                age_num,
                edu_radio,
                disability_cb,
                minority_cb,
                loading_md,
                chatbot,
                tts_row,
                tts_out,
                result_banner,
                result_account_md,
            ],
        )

        save_profile_btn.click(
            on_save_profile,
            inputs=[
                current_user_state,
                lang_radio,
                state_dd,
                category_radio,
                income_num,
                gender_radio,
                age_num,
                edu_radio,
                disability_cb,
                minority_cb,
            ],
            outputs=[current_user_state, account_md, profile_status],
        )

        find_btn.click(
            on_find,
            inputs=[
                current_user_state,
                lang_radio,
                state_dd,
                category_radio,
                income_num,
                gender_radio,
                age_num,
                edu_radio,
                disability_cb,
                minority_cb,
            ],
            outputs=[
                form_col,
                results_col,
                loading_md,
                chatbot,
                tts_row,
                tts_out,
                current_lang,
                result_account_md,
                result_banner,
                current_user_state,
                profile_status,
                account_md,
            ],
        )

        def on_tts_toggle(history, lang_code, enabled):
            if not history:
                return None
            last_reply = history[-1][1] if history[-1][1] else ""
            return maybe_tts(last_reply, lang_code, enabled)

        tts_cb.change(
            on_tts_toggle,
            inputs=[chatbot, lang_state, tts_cb],
            outputs=[tts_out],
        )

        # Update lang_state when language radio changes
        lang_radio.change(
            lambda x: x,
            inputs=[lang_radio],
            outputs=[lang_state],
        )

        def on_again():
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value=[]),
                gr.update(visible=False),
                None,
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
            )

        again_btn.click(
            on_again,
            outputs=[
                form_col,
                results_col,
                loading_md,
                chatbot,
                tts_row,
                tts_out,
                result_banner,
                profile_status,
            ],
        )

        gr.Markdown(
            "<small>Powered by Databricks (Vector Search RAG + Llama) · "
            "Sarvam AI (translation, text-to-speech) · "
            "Data: Government of India scholarship schemes</small>"
        )

    return demo


def _load_secrets_from_scope() -> None:
    """Load secrets from Databricks secret scope into env vars (for Databricks Apps).

    The Apps UI secret resources don't always wire through reliably.
    Fall back to reading from the workspace secret scope via the SDK,
    the same way notebooks do with dbutils.secrets.get().
    """
    mapping = {
        "SARVAM_API_KEY":    ("scholarship", "sarvam_api_key"),
        "DATABRICKS_TOKEN":  ("scholarship", "databricks_token"),
    }
    for env_var, (scope, key) in mapping.items():
        if os.environ.get(env_var, "").strip():
            continue  # already set (e.g. locally or via Apps resource)
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            val = w.secrets.get_secret(scope=scope, key=key)
            if val and val.value:
                import base64
                try:
                    decoded = base64.b64decode(val.value).decode("utf-8")
                except Exception:
                    decoded = val.value
                os.environ[env_var] = decoded
                logger.info("Loaded %s from secret scope %s/%s", env_var, scope, key)
        except Exception as exc:
            logger.warning("Could not load %s from secret scope: %s", env_var, exc)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    _load_secrets_from_scope()
    demo = build_app()
    demo.queue()
    # Databricks Apps injects GRADIO_SERVER_PORT and GRADIO_ROOT_PATH.
    # server_name must be 0.0.0.0 so the Apps gateway can reach the process.
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )


if __name__ == "__main__":
    main()
