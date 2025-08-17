import os
import time
import json
import re
import unicodedata
import streamlit as st
from openai import OpenAI

# -----------------------------
# LlamaParse (EXACT base logic; sidebar removed, defaults kept)
# -----------------------------
from llama_cloud_services import LlamaParse, EU_BASE_URL

st.set_page_config(page_title="PDF → Markdown (LlamaParse Agentic Plus)", layout="wide")
st.title("EMBA Due Diligence Checklist Generator")

# --- Secrets / config ---
API_KEY = st.secrets.get("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_CLOUD_API_KEY")
if not API_KEY:
    st.error("Missing LLAMA_CLOUD_API_KEY in .streamlit/secrets.toml")
    st.stop()

base_url = st.secrets.get("LLAMA_CLOUD_BASE_URL")
if base_url and base_url.strip().lower() == "eu":
    base_url = EU_BASE_URL  # convenience alias for EU region

# Defaults (formerly in sidebar)
use_html_tables = True
hide_headers = False
hide_footers = False
show_page_breaks = True

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if not uploaded:
    st.caption("Upload a .pdf to begin.")
    st.stop()

file_bytes = uploaded.read()
extra_info = {"file_name": uploaded.name}

# Agentic Plus = document-wide agent mode
parser = LlamaParse(
    api_key=API_KEY,
    base_url=base_url,
    parse_mode="parse_document_with_agent",  # Agentic Plus / highest-fidelity
    result_type="markdown",                  # return Markdown
    output_tables_as_HTML=use_html_tables,
    hide_headers=hide_headers,
    hide_footers=hide_footers,
    verbose=False,
)

st.info("Parsing with Agentic Plus… (document-wide agent for complex layouts)")
with st.spinner("Contacting LlamaParse…"):
    result = parser.parse(file_bytes, extra_info=extra_info)

# Combine page-level markdown (the API still returns pages even in doc mode)
parts = []
for i, page in enumerate(result.pages, start=1):
    md = page.md or ""
    if show_page_breaks and i > 1:
        parts.append("\n---\n")
    parts.append(md)
full_markdown = "".join(parts).strip()

st.success(f"Parsed **{len(result.pages)}** page(s) from **{uploaded.name}**")


#st.divider()
#st.subheader("Extracted Markdown")
#st.markdown(full_markdown, unsafe_allow_html=use_html_tables)
# -----------------------------
# End of LlamaParse section
# -----------------------------

st.divider()
st.header("Deep Research → Due Diligence Checklist")

# OpenAI config
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=3600)

# EXACT checklist template (verbatim)
DD_TEMPLATE = """# DUE DILIGENCE CHECKLIST

**Section** | **Recommended materials to be included, but not limited to:**

### Executive Summary 
- Description of opportunity and key description of business structure
- Key due diligence findings •
- Assumptions & Evaluation of key risk •

---

### Overview of Project 
- Introduction & Overview •

---

### Key Due Diligence and Assumption Validation 
- Market Opportunity •
- Business Model •

---

# DUE DILIGENCE CHECKLIST

**Section**

- Competitive Analysis •
- Execution Feasibility (Business) •

---

### Investment Rationale 
- Investment Thesis •
- Key Enablers and Assumption
- Risks and Uncertainties

---

NUS BUSINESS SCHOOL

© Professor Virginia Cha and Jeremy Goh 2022. All Rights Reserved.
"""

# -----------------------------
# Robust text & URL sanitizers
# -----------------------------
# characters to treat as zero-width/bidi controls
_ZERO_WIDTH_BIDI = r"\u200B\u200C\u200D\u2060\ufeff\u200E\u200F\u061C\u202A-\u202E\u2066-\u2069"
_ZERO_WIDTH_BIDI_CLASS = f"[{_ZERO_WIDTH_BIDI}]"

# exotic spaces that should become a normal space
_UNICODE_SPACES = [
    "\u00A0", "\u1680", "\u180E", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004",
    "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200A", "\u202F", "\u205F",
    "\u3000"
]

# map fancy dashes to ASCII hyphen (for URLs only)
_DASHES_FOR_URL = dict.fromkeys(map(ord, "–—‒−-"), "-")  # includes non-breaking hyphen

def _normalize_unicode_text(s: str) -> str:
    """Flatten fancy Unicode, preserve word boundaries, normalize spaces, fix dash spacing."""
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)

    # normalize exotic spaces to regular spaces
    for sp in _UNICODE_SPACES:
        s = s.replace(sp, " ")

    # if zero-width/bidi controls appear *between* word chars, turn them into a real space
    s = re.sub(rf"(?<=\w){_ZERO_WIDTH_BIDI_CLASS}+(?=\w)", " ", s)

    # remove remaining zero-width/bidi controls elsewhere
    s = re.sub(_ZERO_WIDTH_BIDI_CLASS, "", s)

    # remove soft hyphen (used for hyphenation in copy/paste)
    s = s.replace("\u00AD", "")

    # ensure spaces around em/en/ASCII hyphen when used between word-ish tokens
    s = re.sub(r"(?<=[\w\)\]])([–—-])(?=[\w\(\[])", r" \1 ", s)

    # collapse repeated spaces and tidy spaces before punctuation
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" +([,.;:!?])", r"\1", s)

    return s

def _clean_url(url: str) -> str:
    """Canonicalize URL: remove text-fragment, strip whitespace, normalize dashes, trim trailing punctuation."""
    if not url:
        return url
    u = unicodedata.normalize("NFKC", url)
    # remove Chrome text fragments (allow random spaces)
    u = re.sub(r"#\s*:\s*~\s*:\s*text\s*=.*$", "", u, flags=re.IGNORECASE)
    # drop all whitespace characters and soft hyphens
    u = re.sub(r"\s+", "", u).replace("\u00AD", "")
    # normalize dash variants
    u = u.translate(_DASHES_FOR_URL)
    # strip trailing punctuation accidentally captured
    u = re.sub(r"[),.;:!?]+$", "", u)
    return u

def _sanitize_md_links(md: str) -> str:
    """Rewrite [label](url) with cleaned url."""
    def repl(m):
        label, url = m.group(1), m.group(2)
        return f"[{label}]({_clean_url(url)})"
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, md)

def _sanitize_bare_urls(md: str) -> str:
    """Clean bare URLs not inside markdown links."""
    def repl(m):
        return _clean_url(m.group(0))
    return re.sub(r"https?://[^\s)\]\}>]+", repl, md)

def _final_sanitize(md_text: str) -> str:
    """Full pipeline for model output before rendering."""
    t = _normalize_unicode_text(md_text)
    t = _sanitize_md_links(t)
    t = _sanitize_bare_urls(t)
    return t

# -----------------------------

# Single action button
if st.button("Create Due Diligence Report", type="primary"):
    # Prompt updated per request (no slides/bullets mention; no repetition; formulas rendered as code/plain text)
    prompt = f"""
You are a senior research analyst. You will receive a venture pitch or business document in Markdown.

TASK:
1) Conduct deep research using axuthoritative, up-to-date sources. Gather facts, numbers, trends, regulations, competitors, market sizing, unit economics, execution feasibility factors, and key risks.
2) Produce a due diligence checklist report that fills the template EXACTLY as provided below. Do not change headings, punctuation, symbols, or section order. Keep the literal text that is part of the template; add your researched content as detailed content under each listed line.
3) Do NOT repeat information between sections. Each section must provide new, non-duplicative insights (you may cross-reference without copying text).
4) Do NOT add any formulas or colors. Reply in only markdown and nothing else.
5) ONLY REPLY IN MARKDOWN, NO COLORS, NO HTML

OUTPUT:
- Return only the completed template in Markdown. Do not add extra sections or commentary before or after.
- Preserve this exact template shape and wording:

{DD_TEMPLATE}

REFERENCE DOCUMENT (from the user upload):
---
{full_markdown}
"""

    # Internal defaults (no UI controls)
    include_code_interpreter = True
    max_tool_calls = 120

    tools = [{"type": "web_search_preview"}]
    if include_code_interpreter:
        tools.append({"type": "code_interpreter", "container": {"type": "auto"}})

    # Submit background job
    try:
        job = client.responses.create(
            model="o4-mini-deep-research",
            input=prompt,
            background=True,
            tools=tools,
            max_tool_calls=int(max_tool_calls),
        )
    except Exception as e:
        st.error(f"Failed to submit deep research job: {e}")
        st.stop()

    job_id = getattr(job, "id", None)
    if not job_id:
        st.error("Did not receive a job id from the API.")
        st.stop()

    # Progress bar only (with percentage), 2× slower than before
    prog = st.progress(0)
    pct_text = st.empty()
    started = time.time()
    pct = 5
    prog.progress(pct)
    pct_text.markdown(f"**{pct}%**")

    def fetch_job(jid):
        try:
            return client.responses.get(jid)
        except Exception:
            return client.responses.retrieve(jid)

    while True:
        time.sleep(3)  # polling interval
        try:
            job = fetch_job(job_id)
        except Exception:
            continue

        status = getattr(job, "status", "unknown")

        if status in ("queued", "in_progress", "unknown"):
            elapsed = int(time.time() - started)
            pct = min(95, max(pct, 10 + (elapsed // 4)))
            prog.progress(pct)
            pct_text.markdown(f"**{pct}%**")
            continue
        elif status == "completed":
            prog.progress(100)
            pct_text.markdown("**100%**")
            break
        elif status in ("failed", "cancelled", "errored"):
            prog.progress(100)
            pct_text.markdown("**100%**")
            st.error(f"Job status: {status}")
            try:
                st.code(json.dumps(job.dict(), indent=2) if hasattr(job, "dict") else str(job))
            except Exception:
                pass
            st.stop()
        else:
            pct = min(98, pct + 1)
            prog.progress(pct)
            pct_text.markdown(f"**{pct}%**")

    # Retrieve final text
    output_text = getattr(job, "output_text", None)
    if not output_text:
        try:
            chunks = []
            for it in getattr(job, "output", []) or []:
                t = getattr(it, "type", None) or (isinstance(it, dict) and it.get("type"))
                if t == "message":
                    content = getattr(it, "content", []) or it.get("content", [])
                    for c in content:
                        if (getattr(c, "type", None) or c.get("type")) == "output_text":
                            chunks.append(getattr(c, "text", None) or c.get("text", ""))
            output_text = "\n".join(chunks).strip()
        except Exception:
            output_text = None

    if not output_text:
        st.error("No text returned from Deep Research.")
        st.stop()

    # --- Normalize weird Unicode & sanitize links/URLs before rendering ---
    sanitized_output = _final_sanitize(output_text)

    st.divider()
    st.subheader("Due Diligence Checklist (Completed)")
    st.markdown(sanitized_output, unsafe_allow_html=False)

    # (Removed the final download button as requested)
