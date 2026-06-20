"""
LexaAI Constants
================
Central configuration for paths, model names, and hyperparameters.
"""

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Data Paths ────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
SECTIONS_DIR = DATA_DIR / "sections"
SECTIONS_JSON = SECTIONS_DIR / "all_sections.json"
CUAD_DIR = DATA_DIR / "cuad"
SYNTHETIC_QA_DIR = DATA_DIR / "synthetic_qa"
SYNTHETIC_QA_JSON = SYNTHETIC_QA_DIR / "synthetic_qa.json"

# ── Model Paths ───────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
STAGE1_MODEL_DIR = MODELS_DIR / "stage1_cuad"
STAGE2_MODEL_DIR = MODELS_DIR / "stage2_lexai"
AUTOENCODER_PATH = MODELS_DIR / "autoencoder.pt"
DISCRIMINATOR_DIR = MODELS_DIR / "discriminator"

# ── Embeddings ────────────────────────────────────────────────────
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
CHROMA_PERSIST_DIR = EMBEDDINGS_DIR / "chromadb"

# ── Evaluation ────────────────────────────────────────────────────
EVAL_DIR = PROJECT_ROOT / "evaluation"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# ── Model Configuration ──────────────────────────────────────────
BASE_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS_STAGE1 = 3
NUM_EPOCHS_STAGE2 = 5

# ── Autoencoder Configuration ────────────────────────────────────
INPUT_DIM = 768          # Legal-BERT hidden size
LATENT_DIM = 256         # Autoencoder bottleneck dimension
NOISE_STD = 0.1          # Gaussian noise for denoising
AE_EPOCHS = 50
AE_LR = 1e-3

# ── GAN Discriminator Configuration ──────────────────────────────
DISC_CONFIDENCE_THRESHOLD = 0.75  # Safety gate threshold
DISC_EPOCHS = 20
DISC_LR = 1e-4

# ── ChromaDB Configuration ───────────────────────────────────────
CHROMA_COLLECTION_STATUTES = "sri_lanka_statutes"
CHROMA_COLLECTION_PRECEDENTS = "sri_lanka_precedents"
RETRIEVAL_TOP_K = 8

# ── Evaluation Targets ───────────────────────────────────────────
TARGET_PRECISION_AT_5 = 0.70
TARGET_ADVERSE_FINDING_RATE = 0.80
TARGET_GAN_AUC = 0.80
TARGET_QUERY_LATENCY_MS = 200
NUM_TEST_QUERIES_PER_DOMAIN = 10
NUM_DOMAINS = 6

# ── Legal Domains ─────────────────────────────────────────────────
DOMAINS = [
    "family_law",
    "contract_law",
    "business_law",
    "traffic_law",
    "criminal_law",
]

# ── Disclaimer Text ───────────────────────────────────────────────
MANDATORY_DISCLAIMER = (
    "⚖️ LEGAL DISCLAIMER: This analysis is based on retrieved statute sections "
    "and does not constitute professional legal advice. For matters with significant "
    "consequences, consult a qualified Sri Lankan lawyer."
)

CRIMINAL_LAW_DISCLAIMER = (
    "⚠️ CRIMINAL LAW ALERT: This query involves potential criminal offences. "
    "This system provides INFORMATION ONLY. You should immediately consult "
    "a qualified criminal lawyer. Do NOT rely on this system for defence strategy."
)

# ── Output Modes ──────────────────────────────────────────────────
OUTPUT_MODES = [
    "plain_rights_summary",
    "adverse_finding",
    "courtroom_preparation",
    "formal_complaint_letter",
    "statute_reference_card",
]

OUTPUT_MODE_LABELS = {
    "plain_rights_summary": "Plain Rights Summary",
    "adverse_finding": "Adverse Finding",
    "courtroom_preparation": "Courtroom Preparation Script",
    "formal_complaint_letter": "Formal Complaint Letter",
    "statute_reference_card": "Statute Reference Card",
}
