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
# Each list covers both formal legal terms AND the everyday language
# that real users naturally write — crucial for domain detection without
# forcing users to know legal vocabulary.
DOMAIN_KEYWORDS = {
    "family_law": [
        # Formal legal terms
        "marriage", "matrimonial", "inheritance", "spouse", "divorce",
        "marriage registration", "consent", "alimony", "affiliation",
        # Natural language — relationships
        "married", "marrying", "get married", "register marriage",
        "husband", "wife", "wedding", "couple",
        # Natural language — children & guardianship
        "child", "minor", "custody", "adoption", "guardian",
        "orphan", "parental", "children",
        # Natural language — death & estates
        "widow", "widower", "heir", "estate", "deceased", "dead",
        "death", "passed away", "died", "funeral", "probate",
        # Natural language — separation
        "separation", "separated", "divorce petition", "annulment",
    ],
    "contract_law": [
        # Formal legal terms
        "fraud", "notary", "deed", "immovable property", "will", "testament",
        "witness", "notarial", "frauds ordinance",
        # Natural language — property transactions
        "land", "property", "house", "sell land", "buy land", "purchase land",
        "transfer land", "title deed", "land deed", "property sale",
        # Natural language — agreements
        "contract", "signed", "agreement", "written agreement",
        "verbal agreement", "oral agreement", "document", "written",
        # Natural language — housing & tenancy
        "landlord", "tenant", "lease", "rent", "rental", "deposit",
        "eviction", "renting", "rent agreement",
        # Natural language — money/transactions
        "loan", "borrow", "lend", "money owed", "debt",
    ],
    "business_law": [
        # Formal legal terms
        "sale of goods", "buyer", "seller", "warranty", "delivery",
        "merchantable", "lien", "contract of sale",
        # Natural language — consumer purchases
        "shop", "bought", "purchased", "purchase", "product", "item",
        "goods", "defective", "broken", "damaged", "faulty", "fault",
        "refund", "replace", "replacement", "return", "receipt",
        "consumer", "customer",
        # Natural language — trade & commerce
        "merchant", "store", "sell", "sold", "trade", "business",
        "shop owner", "quality", "fit for purpose", "price",
        "invoice", "payment", "online shopping",
        # Natural language — after-purchase problems
        "doesn't work", "stopped working", "broke", "malfunction",
        "not fit", "not as described",
    ],
    "traffic_law": [
        # Formal legal terms
        "motor traffic", "vehicle registration", "commissioner of motor traffic",
        # Natural language — vehicles
        "vehicle", "car", "bus", "motorbike", "motorcycle", "truck",
        "van", "lorry", "tuk tuk", "three wheeler",
        # Natural language — driving
        "licence", "driving", "drive", "driver", "driving licence",
        "licence plate", "number plate", "registration plate",
        # Natural language — incidents
        "accident", "crash", "collision", "hit", "road", "highway",
        "traffic", "speeding", "speed limit", "drunk driving",
        # Natural language — regulations
        "insurance", "road tax", "permit", "parking", "lane",
        "traffic light", "road rules", "fine", "traffic fine",
        # Natural language — inspection
        "safety inspection", "fitness", "roadworthy",
    ],
    "criminal_law": [
        # Formal legal terms
        "penal", "offence", "culpable homicide", "grievous hurt", "forgery",
        "cheating", "abetment",
        # Natural language — crimes
        "criminal", "crime", "theft", "stolen", "steal", "robbed",
        "robbery", "assault", "murder", "kill", "killed", "attack",
        "attacked", "hurt", "harmed", "violence", "rape",
        # Natural language — process
        "arrested", "arrest", "police", "complaint", "FIR", "court",
        "suspect", "accused", "charge", "charged", "guilty",
        # Natural language — punishment
        "punishment", "imprisonment", "jail", "prison", "sentence",
        "fine", "penalty",
        # Natural language — victim reporting
        "victim", "perpetrator", "report to police", "file complaint",
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
