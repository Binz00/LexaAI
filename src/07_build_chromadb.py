#!/usr/bin/env python3
"""
LexaAI — Step 07: Build ChromaDB Index
=========================================
Embeds all statute sections into ChromaDB with rich metadata including
domain, act, section number, and Part/Chapter-level structural info.

Usage:
    python src/07_build_chromadb.py
Output:
    - embeddings/chromadb/ (persistent ChromaDB)
"""
import json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import chromadb
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

from src.utils.constants import (
    SECTIONS_JSON, SECTIONS_DIR, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_STATUTES, RETRIEVAL_TOP_K,
)


def main():
    logger.info("🔨 LexaAI ChromaDB Builder")

    # Load sections
    if not SECTIONS_JSON.exists():
        logger.error(f"❌ Run src/01_parse_pdfs.py first")
        sys.exit(1)
    with open(SECTIONS_JSON, "r", encoding="utf-8") as f:
        sections = json.load(f)
    logger.info(f"📄 Loaded {len(sections)} sections")

    # Load embeddings (prefer latent, fall back to raw)
    latent_path = SECTIONS_DIR / "latent_embeddings.npy"
    raw_path = SECTIONS_DIR / "section_embeddings.npy"
    if latent_path.exists():
        embeddings = np.load(str(latent_path))
        logger.info(f"📊 Using latent embeddings: {embeddings.shape} (Norm sample: {np.linalg.norm(embeddings[0]):.4f})")
    elif raw_path.exists():
        embeddings = np.load(str(raw_path))
        logger.info(f"📊 Using raw embeddings: {embeddings.shape} (Norm sample: {np.linalg.norm(embeddings[0]):.4f})")
    else:
        logger.error("❌ No embeddings found. Run src/03_build_embeddings.py first")
        sys.exit(1)

    assert len(sections) == len(embeddings), \
        f"Mismatch: {len(sections)} sections vs {len(embeddings)} embeddings"

    # Initialize ChromaDB
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    # Delete existing collection if it exists
    try:
        client.delete_collection(CHROMA_COLLECTION_STATUTES)
        logger.info("🗑️ Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_STATUTES,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"📦 Created collection: {CHROMA_COLLECTION_STATUTES} with cosine similarity")

    # Deduplicate IDs globally
    seen_ids = set()
    dedup_sections = []
    dedup_indices = []
    dup_count = 0
    for idx, s in enumerate(sections):
        sid = s["id"]
        if sid in seen_ids:
            # Append counter to make unique
            counter = 1
            while f"{sid}_{counter}" in seen_ids:
                counter += 1
            sid = f"{sid}_{counter}"
            dup_count += 1
        seen_ids.add(sid)
        s_copy = dict(s)
        s_copy["id"] = sid
        dedup_sections.append(s_copy)
        dedup_indices.append(idx)

    if dup_count:
        logger.warning(f"⚠️ Resolved {dup_count} duplicate IDs")

    # Add sections in batches
    batch_size = 100
    for i in tqdm(range(0, len(dedup_sections), batch_size), desc="Adding to ChromaDB"):
        batch_secs = dedup_sections[i:i+batch_size]
        batch_idx = dedup_indices[i:i+batch_size]
        batch_embs = embeddings[batch_idx]

        ids = [s["id"] for s in batch_secs]
        documents = [s["text"][:5000] for s in batch_secs]  # ChromaDB doc limit
        metadatas = []
        for s in batch_secs:
            meta = {
                "act": s["act"],
                "domain": s["domain"],
                "section_number": s["section_number"],
                "header": s.get("header", "")[:500],
                "level1": s.get("level1", ""),
                "level1_heading": s.get("level1_heading", ""),
                "char_count": s["char_count"],
            }
            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=batch_embs.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    # Verify
    count = collection.count()
    logger.info(f"✅ ChromaDB populated with {count} entries")

    # Test query
    logger.info("🔍 Test query: 'penalty for theft' (using embedding 0 as dummy query)")
    results = collection.query(
        query_embeddings=[embeddings[0].tolist()],
        n_results=5,
    )
    if results and results["documents"]:
        for j, (doc, meta, dist) in enumerate(zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results.get("distances", [[0]*5])[0]
        )):
            logger.info(f"  {j+1}. [{meta['domain']}] {meta['section_number']} — {meta['act']} (Distance: {dist:.4f})")

    logger.info(f"✅ ChromaDB build complete! ({count} entries)")

if __name__ == "__main__":
    main()
