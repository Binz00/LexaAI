#!/usr/bin/env python3
"""
LexaAI — Step 03: Build Legal-BERT Embeddings
================================================
Generates [CLS] token embeddings for all parsed statute sections
using the Stage 1 fine-tuned Legal-BERT model. Produces a UMAP
visualization showing 6 domain clusters.

Usage:
    python src/03_build_embeddings.py

Output:
    - data/sections/section_embeddings.npy (768-dim embeddings)
    - evaluation/results/umap_initial_clusters.png (UMAP plot)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from src.utils.constants import (
    STAGE1_MODEL_DIR,
    BASE_MODEL_NAME,
    SECTIONS_JSON,
    SECTIONS_DIR,
    EVAL_RESULTS_DIR,
    MAX_SEQ_LEN,
    BATCH_SIZE,
)


def load_sections() -> list[dict]:
    """Load parsed sections from JSON."""
    if not SECTIONS_JSON.exists():
        print(f"❌ Sections file not found: {SECTIONS_JSON}")
        print("   Run src/01_parse_pdfs.py first.")
        sys.exit(1)

    with open(SECTIONS_JSON, "r", encoding="utf-8") as f:
        sections = json.load(f)

    print(f"📄 Loaded {len(sections)} sections")
    return sections


def generate_embeddings(
    sections: list[dict],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Generate [CLS] token embeddings for all sections.

    Uses the last hidden state's [CLS] token (position 0) as the
    sentence-level representation of each legal section.

    Args:
        sections: List of section dictionaries (must have 'text' key)
        tokenizer: Legal-BERT tokenizer
        model: Legal-BERT model
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        NumPy array of shape (n_sections, 768)
    """
    model.eval()
    all_embeddings = []
    texts = [s["text"][:2000] for s in sections]  # Truncate very long texts

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract [CLS] token embedding (first token of last hidden state)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def create_umap_visualization(
    embeddings: np.ndarray,
    sections: list[dict],
    output_path: Path,
):
    """
    Create a UMAP plot showing domain clusters.

    Args:
        embeddings: (n_sections, 768) embedding matrix
        sections: Section metadata for labels
        output_path: Path to save the plot
    """
    try:
        import umap
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  umap-learn or matplotlib not installed, skipping visualization")
        return

    print("\n🗺️  Generating UMAP visualization...")

    # Reduce to 2D
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # Color by domain
    domains = [s["domain"] for s in sections]
    unique_domains = sorted(set(domains))
    color_map = {
        "family_law": "#FF6B6B",
        "contract_law": "#4ECDC4",
        "business_law": "#45B7D1",
        "traffic_law": "#FFA07A",
        "criminal_law": "#9B59B6",
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    for domain in unique_domains:
        mask = [d == domain for d in domains]
        points = embedding_2d[mask]
        color = color_map.get(domain, "#808080")
        label = domain.replace("_", " ").title()
        ax.scatter(
            points[:, 0], points[:, 1],
            c=color, label=label,
            alpha=0.6, s=15, edgecolors="white", linewidth=0.3,
        )

    ax.set_title("LexaAI — Legal Section Embeddings (UMAP)", fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 UMAP plot saved to: {output_path}")


def main():
    print("🔨 LexaAI Embedding Builder")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load model — prefer Stage 1 fine-tuned, fall back to base
    model_path = str(STAGE1_MODEL_DIR) if STAGE1_MODEL_DIR.exists() else BASE_MODEL_NAME
    print(f"\n📦 Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)

    # Load sections
    sections = load_sections()

    # Generate embeddings
    print(f"\n🔄 Generating [CLS] embeddings for {len(sections)} sections...")
    embeddings = generate_embeddings(sections, tokenizer, model, device)
    print(f"  Embedding shape: {embeddings.shape}")

    # Save embeddings
    embeddings_path = SECTIONS_DIR / "section_embeddings.npy"
    np.save(str(embeddings_path), embeddings)
    print(f"  💾 Embeddings saved to: {embeddings_path}")

    # Save section IDs for mapping
    ids_path = SECTIONS_DIR / "section_ids.json"
    ids = [s["id"] for s in sections]
    with open(ids_path, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"  💾 Section IDs saved to: {ids_path}")

    # UMAP visualization
    umap_path = EVAL_RESULTS_DIR / "umap_initial_clusters.png"
    create_umap_visualization(embeddings, sections, umap_path)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  Sections embedded: {len(sections)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Model used: {model_path}")
    print("\n✅ Embedding generation complete!")


if __name__ == "__main__":
    main()
