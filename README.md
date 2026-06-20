# LexaAI — Sri Lankan Legal AI Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> AI-powered legal information system that helps ordinary Sri Lankan citizens understand
> their legal rights using Legal-BERT, ChromaDB RAG retrieval, a denoising autoencoder,
> and a GAN hallucination discriminator.

## ⚠️ Disclaimer

**This system provides INFORMATION ONLY and does not constitute professional legal advice.**
For matters with significant consequences, consult a qualified Sri Lankan lawyer.

---

## Architecture

```
User Query
    → Legal-BERT Tokenizer
    → Legal-BERT Encoder ([CLS] embedding, 768-dim)
    → Denoising Autoencoder (latent vector z, 256-dim)
    → ChromaDB Retrieval (top-8 statute sections)
    → RAG Context Assembly
    → Legal-BERT QA Generation
    → GAN Discriminator Safety Gate (threshold: 0.75)
    → Output Renderer (5 modes)
    → Streamlit UI + Mandatory Disclaimer
```

## Legal Domains (6 Statutes)

| Domain | Statute | Sections |
|--------|---------|----------|
| Family Law | Marriage Registration Ordinance | ~64 |
| Family Law | Matrimonial Rights & Inheritance | ~36 |
| Contract Law | Prevention of Frauds (2024 Consolidated) | ~19 |
| Business Law | Sale of Goods Ordinance | ~59 |
| Traffic Law | Motor Traffic Act (2024 Consolidated) | ~239 |
| Criminal Law | Penal Code | ~490 |

## Output Modes

1. **Plain Rights Summary** — Plain-English explanation of rights and obligations
2. **Adverse Finding** — Identifies when the user has acted unlawfully
3. **Courtroom Preparation Script** — Structured script with statute citations
4. **Formal Complaint Letter** — Professionally formatted legal complaint
5. **Statute Reference Card** — Compact list of relevant Act/section numbers

## Setup

```bash
# 1. Clone and enter directory
cd LexaAI

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template and add your API key
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or GOOGLE_API_KEY

# 5. Place PDF statute files in data/raw_pdfs/
# See "Required PDF Files" below

# 6. Run the pipeline (in order)
python src/01_parse_pdfs.py          # Parse PDFs → sections JSON
python src/02_finetune_stage1.py     # Fine-tune on CUAD
python src/03_build_embeddings.py    # Generate embeddings
python src/04_generate_qa.py         # Generate synthetic Q&A
python src/05_finetune_stage2.py     # Fine-tune on Sri Lankan law
python src/06_train_autoencoder.py   # Train denoising autoencoder
python src/07_build_chromadb.py      # Build ChromaDB index
python src/08_train_discriminator.py # Train GAN discriminator

# 7. Launch the app
streamlit run src/11_app.py
```

## Required PDF Files

Place these files in `data/raw_pdfs/`:

| Filename | Statute |
|----------|---------|
| `LAW_ord_1995_18_en.pdf` | Marriage Registration Ordinance |
| `LAW_MATRIMONIAL_RIGHTS_AND_INHERITANCE.pdf` | Matrimonial Rights & Inheritance |
| `Prevention-of-Frauds-Consolidated-2024.pdf` | Prevention of Frauds (2024) |
| `SOGO.pdf` | Sale of Goods Ordinance |
| `Motor-Traffic-Consolidated-2024.pdf` | Motor Traffic Act (2024) |
| `LAW_Penal_Code.pdf` | Penal Code |

> ⚠️ Do NOT include `LAW_Prevention_of_Frauds_Ordinance.pdf` (old version without 2022/2024 amendments).

## Evaluation

```bash
python evaluation/eval_retrieval.py     # Precision@5 > 0.70 across 60 queries
python evaluation/eval_impartiality.py  # Adverse finding rate > 80%
python evaluation/eval_gan.py           # GAN discriminator AUC > 0.80
```

## Project Structure

```
LexaAI/
├── data/
│   ├── raw_pdfs/          # Source PDF statute files
│   ├── sections/          # Parsed sections JSON
│   ├── cuad/              # CUAD dataset cache
│   └── synthetic_qa/      # Generated Q&A pairs
├── embeddings/chromadb/   # ChromaDB persistent storage
├── models/                # Trained model checkpoints
├── evaluation/            # Evaluation scripts & results
├── src/                   # Source code (numbered pipeline)
│   └── utils/             # Shared utilities
├── notebooks/             # Exploration notebooks
└── Project plan/          # Planning documents
```

---

*LexAI Implementation v4.0 · Faculty of Computing · 2025/2026*
