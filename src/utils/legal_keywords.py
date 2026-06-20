"""
LexaAI Legal Keywords & PDF Mapping
====================================
Domain keyword lists for filtering and the PDF_MAP configuration
that maps each source PDF to its statute metadata.
"""

# ── PDF_MAP ───────────────────────────────────────────────────────
# Maps each PDF filename to its statute name, legal domain, and
# format type for selecting the appropriate cleaning function.
#
# IMPORTANT: Do NOT include LAW_Prevention_of_Frauds_Ordinance.pdf
# That is the old version without the 2022/2024 amendments.
# Use Prevention-of-Frauds-Consolidated-2024.pdf instead.

PDF_MAP = {
    # ── Family Law ──────────────────────────────────────────────
    "LAW_ord_1995_18_en.pdf": {
        "act":    "Marriage Registration Ordinance",
        "domain": "family_law",
        "format": "web_extract",   # text from lawnet.lk — clean text
    },
    "LAW_MATRIMONIAL RIGHTS AND INHERITANCE.pdf": {
        "act":    "Matrimonial Rights and Inheritance Ordinance",
        "domain": "family_law",
        "format": "web_extract",
    },
    # ── Contract Law ────────────────────────────────────────────
    "Prevention-of-Frauds-Consolidated-2024.pdf": {
        "act":    "Prevention of Frauds Ordinance (2024 Consolidated)",
        "domain": "contract_law",
        "format": "consolidated_pdf",   # formatted statute PDF
    },
    # ── Business Law ────────────────────────────────────────────
    "SOGO.pdf": {
        "act":    "Sale of Goods Ordinance",
        "domain": "business_law",
        "format": "scanned_columns",    # dual-column layout
    },
    # ── Traffic Law ─────────────────────────────────────────────
    "Motor-Traffic-Consolidated-2024.pdf": {
        "act":    "Motor Traffic Act (2024 Consolidated)",
        "domain": "traffic_law",
        "format": "consolidated_pdf",   # LARGE: 184 pages, 14 Parts
    },
    # ── Criminal Law (NEW in v4) ────────────────────────────────
    "LAW_Penal_Code.pdf": {
        "act":    "Penal Code",
        "domain": "criminal_law",
        "format": "scanned_columns",    # LARGE: 166 pages, 22 Chapters
    },
}

# Files to explicitly skip if found in raw_pdfs/
SKIP_FILES = [
    "LAW_Prevention of Frauds Ordinance.pdf",  # old version without 2022/2024 amendments
]

# Used for retrieval boosting and domain detection
DOMAIN_KEYWORDS = {
    "family_law": [
        "marriage", "matrimonial", "inheritance", "spouse", "divorce",
        "marriage registration", "consent", "minor", "widow", "child", 
        "husband", "wife", "wedding", "alimony",
    ],
    "contract_law": [
        "fraud", "notary", "deed", "immovable property", "will", "testament",
        "witness", "writing", "agreement", "notarial",
    ],
    "business_law": [
        "sale of goods", "buyer", "seller", "warranty", "delivery",
        "merchantable", "lien", "price", "goods", "contract of sale",
    ],
    "traffic_law": [
        "motor traffic", "vehicle", "licence", "driving", "vehicle registration",
        "insurance", "accident", "road", "speed", "traffic", "driver", "highway",
    ],
    "criminal_law": [
        "penal", "offence", "criminal", "punishment", "imprisonment",
        "theft", "assault", "murder", "forgery", "cheating",
        "culpable homicide", "grievous hurt", "jail",
    ],
}

# Flat list for backward compatibility
TARGET_DOMAINS = [k for v in DOMAIN_KEYWORDS.values() for k in v]

# ── Domain Display Names ──────────────────────────────────────────
DOMAIN_LABELS = {
    "family_law":    "Family Law",
    "contract_law":  "Contract Law",
    "business_law":  "Business Law",
    "traffic_law":   "Traffic Law",
    "criminal_law":  "Criminal Law",
}

# ── Q&A Generation Configuration ─────────────────────────────────
# Number of Q&A pairs to generate per section, by domain
QA_PAIRS_PER_SECTION = {
    "family_law":    3,
    "contract_law":  3,
    "business_law":  3,
    "traffic_law":   2,  # reduced — many procedural sections
    "criminal_law":  2,  # reduced — naturally adversarial
}

# Target user-liable percentage by domain
USER_LIABLE_TARGET = {
    "family_law":    0.30,
    "contract_law":  0.35,
    "business_law":  0.30,
    "traffic_law":   0.40,
    "criminal_law":  0.50,  # naturally high — sections define offences
}

# Minimum section character count for Q&A generation
MIN_SECTION_CHARS_FOR_QA = 100
