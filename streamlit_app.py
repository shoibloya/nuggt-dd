import os
import time
import json
import streamlit as st
from openai import OpenAI

# -----------------------------
# LlamaParse (EXACT base logic; sidebar removed, defaults kept)
# -----------------------------
from llama_cloud_services import LlamaParse, EU_BASE_URL

st.set_page_config(page_title="PDF → Markdown (LlamaParse Agentic Plus)", layout="wide")
st.title("PDF → Markdown (Agentic Plus)")

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
st.download_button(
    "Download Markdown",
    data=full_markdown.encode("utf-8"),
    file_name=f"{os.path.splitext(uploaded.name)[0]}.md",
    mime="text/markdown",
)

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

# Single action button
if st.button("Create Due Diligence Report", type="primary"):
    # Prompt updated per request (no slides/bullets mention; no repetition; formulas rendered as code/plain text)
    prompt = f"""
You are a senior research analyst. You will receive a venture pitch or business document in Markdown.

TASK:
1) Conduct deep research using axuthoritative, up-to-date sources. Gather facts, numbers, trends, regulations, competitors, market sizing, unit economics, execution feasibility factors, and key risks.
2) Produce a due diligence checklist report that fills the template EXACTLY as provided below. Do not change headings, punctuation, symbols, or section order. Keep the literal text that is part of the template; add your researched content as detailed content under each listed line.
3) Do NOT repeat information between sections. Each section must provide new, non-duplicative insights (you may cross-reference without copying text).
4) Do NOT use LaTeX. Render any formulas or expressions as plain text or Markdown code (e.g., `ROI = (Gain - Cost) / Cost`). 

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
            # keep showing progress only
            continue

        status = getattr(job, "status", "unknown")

        if status in ("queued", "in_progress", "unknown"):
            # 2× slower progression: prior heuristic ~10 + elapsed//2; now 10 + elapsed//4
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
            # keep progress minimal
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

    st.divider()
    st.subheader("Due Diligence Checklist (Completed)")
    st.markdown(output_text, unsafe_allow_html=False)

    st.download_button(
        "Download Due Diligence Checklist",
        data=output_text.encode("utf-8"),
        file_name=f"{os.path.splitext(uploaded.name)[0]}_due_diligence_checklist.md",
        mime="text/markdown",
    )

