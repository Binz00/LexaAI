#!/usr/bin/env python3
"""
LexaAI — Step 11: Streamlit Web Application
===============================================
The main user interface for LexaAI. Provides a clean, professional
interface for querying Sri Lankan law across 6 legal domains.

Features:
    - Text input for legal situation description
    - Output mode selector (5 modes)
    - Mandatory legal disclaimer on every output
    - Enhanced criminal law disclaimer
    - Source citations with expandable statute text
    - GAN confidence score display
    - Domain-colored section cards

Usage:
    streamlit run src/11_app.py
"""
import sys
import importlib
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.utils.constants import (
    OUTPUT_MODES, OUTPUT_MODE_LABELS,
    MANDATORY_DISCLAIMER, CRIMINAL_LAW_DISCLAIMER,
)

_renderer_mod = importlib.import_module("src.10_output_renderer")
render_output = _renderer_mod.render_output


# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="LexaAI — Sri Lankan Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .disclaimer-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404; /* Dark brown for readability */
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-weight: 500;
    }
    .criminal-disclaimer {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24; /* Dark red for readability */
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-weight: 500;
    }
    .section-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .domain-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
    .stTextArea textarea { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

DOMAIN_COLORS = {
    "family_law": "#FF6B6B",
    "contract_law": "#4ECDC4",
    "business_law": "#45B7D1",
    "traffic_law": "#FFA07A",
    "criminal_law": "#9B59B6",
}


# ── Load Pipeline (cached) ───────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached across sessions)."""
    try:
        _rag_mod = importlib.import_module("src.09_rag_pipeline")
        LexAIRAGPipeline = _rag_mod.LexAIRAGPipeline
        return LexAIRAGPipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    output_mode = st.selectbox(
        "Output Mode",
        options=OUTPUT_MODES,
        format_func=lambda x: OUTPUT_MODE_LABELS.get(x, x),
        index=0,
    )

    st.markdown("---")

    st.markdown("### 📚 Legal Domains")
    st.markdown("""
    - 👨‍👩‍👧 **Family Law** — Marriage, Inheritance
    - 📜 **Contract Law** — Fraud Prevention
    - 🏪 **Business Law** — Sale of Goods
    - 🚗 **Traffic Law** — Motor Traffic Act
    - ⚖️ **Criminal Law** — Penal Code
    """)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "LexaAI uses Legal-BERT fine-tuned on 6 Sri Lankan statutes "
        "with RAG retrieval and a GAN safety gate."
    )
    st.caption("LexAI v4.0 · Faculty of Computing · 2025/2026")


# ── Main Content ──────────────────────────────────────────────────
st.markdown('<h1 class="main-header">⚖️ LexaAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sri Lankan Legal Information Assistant</p>',
            unsafe_allow_html=True)

# Mandatory disclaimer
st.markdown(
    f'<div class="disclaimer-box">{MANDATORY_DISCLAIMER}</div>',
    unsafe_allow_html=True,
)

# Query input
query = st.text_area(
    "Describe your legal situation",
    placeholder=(
        "Example: My employer dismissed me without any notice or warning. "
        "I had been working there for 5 years. What are my rights?"
    ),
    height=120,
    key="legal_query",
)

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ── Process Query ─────────────────────────────────────────────────
if submit and query.strip():
    pipeline = load_pipeline()

    if pipeline is None:
        st.error("⚠️ Pipeline not available. Ensure all models are trained.")
    else:
        with st.spinner("Analyzing your legal situation..."):
            result = pipeline.query(query.strip(), output_mode)

        # Criminal law disclaimer
        if result.get("has_criminal_law"):
            st.markdown(
                f'<div class="criminal-disclaimer">{CRIMINAL_LAW_DISCLAIMER}</div>',
                unsafe_allow_html=True,
            )

        # Safety check
        safety = result.get("safety", {})
        if not safety.get("safe", True):
            st.warning(
                f"⚠️ Low confidence output (score: {safety.get('score', 0):.3f}). "
                "Results may be unreliable."
            )

        # Render output
        st.markdown("---")
        rendered = render_output(result, output_mode)
        st.markdown(rendered)

        # Confidence and safety metrics
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            conf = result.get("confidence", 0)
            css_class = "confidence-high" if conf > 0.5 else "confidence-low"
            st.metric("QA Confidence", f"{conf:.2%}")
        with col_b:
            score = safety.get("score", 0)
            st.metric("Safety Score", f"{score:.2%}")
        with col_c:
            st.metric("Sections Retrieved", len(result.get("sections", [])))

        # Retrieved sections (expandable)
        sections = result.get("sections", [])
        if sections:
            st.markdown("### 📚 Retrieved Statute Sections")
            for i, s in enumerate(sections, 1):
                meta = s["meta"]
                domain = meta.get("domain", "")
                color = DOMAIN_COLORS.get(domain, "#808080")
                domain_label = domain.replace("_", " ").title()

                with st.expander(
                    f"{i}. {meta['act']} — {meta['section_number']} "
                    f"({domain_label})"
                ):
                    if meta.get("level1"):
                        st.caption(f"📂 {meta['level1']}: {meta.get('level1_heading', '')}")
                    st.markdown(s["text"][:1000])
                    if len(s["text"]) > 1000:
                        st.caption("... (truncated)")

elif submit:
    st.warning("Please describe your legal situation above.")
