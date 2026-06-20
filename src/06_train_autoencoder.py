#!/usr/bin/env python3
"""
LexaAI — Step 06: Train Denoising Autoencoder
================================================
Compresses Legal-BERT [CLS] embeddings (768-dim) to a 256-dim latent
vector z for ChromaDB retrieval. Uses Gaussian noise injection for
robustness so legally similar situations map to similar z vectors.

Usage:
    python src/06_train_autoencoder.py
Output:
    - models/autoencoder.pt
    - evaluation/results/umap_latent_clusters.png
"""
import json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.constants import (
    INPUT_DIM, LATENT_DIM, NOISE_STD, AE_EPOCHS, AE_LR,
    SECTIONS_DIR, AUTOENCODER_PATH, EVAL_RESULTS_DIR, SECTIONS_JSON,
)


class LegalSituationAutoencoder(nn.Module):
    """
    Denoising autoencoder for legal situation compression.
    Encoder: 768 → 512 → 256 (latent z)
    Decoder: 256 → 512 → 768 (reconstruction)
    """
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, input_dim),
        )

    def forward(self, x, noise_std=NOISE_STD):
        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def create_umap_plot(embeddings, sections, output_path):
    """Create UMAP visualization of latent space."""
    try:
        import umap, matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  umap/matplotlib not installed, skipping plot"); return

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=42)
    pts = reducer.fit_transform(embeddings)
    domains = [s["domain"] for s in sections]
    colors = {"family_law": "#FF6B6B", "contract_law": "#4ECDC4",
              "business_law": "#45B7D1", "traffic_law": "#FFA07A",
              "criminal_law": "#9B59B6"}
    fig, ax = plt.subplots(figsize=(14, 10))
    for d in sorted(set(domains)):
        mask = [dm == d for dm in domains]
        ax.scatter(pts[mask, 0], pts[mask, 1], c=colors.get(d, "#888"),
                   label=d.replace("_", " ").title(), alpha=0.6, s=15)
    ax.set_title("LexaAI — Latent Space (256-dim Autoencoder)", fontsize=16)
    ax.legend(fontsize=11); ax.grid(alpha=0.2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150); plt.close()
    print(f"  💾 UMAP plot: {output_path}")


def main():
    print("🔨 LexaAI Autoencoder Training")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load embeddings
    emb_path = SECTIONS_DIR / "section_embeddings.npy"
    if not emb_path.exists():
        print(f"❌ Run src/03_build_embeddings.py first"); sys.exit(1)
    embeddings = np.load(str(emb_path))
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load sections for UMAP labels
    with open(SECTIONS_JSON) as f:
        sections = json.load(f)

    # Prepare data
    tensor_data = torch.FloatTensor(embeddings)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model
    model = LegalSituationAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    # Train
    print(f"\n🏋️ Training for {AE_EPOCHS} epochs...")
    for epoch in range(AE_EPOCHS):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, z = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{AE_EPOCHS} — loss: {avg_loss:.6f}")

    # Save
    AUTOENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(AUTOENCODER_PATH))
    print(f"\n💾 Saved: {AUTOENCODER_PATH}")

    # Generate latent embeddings and UMAP
    model.eval()
    with torch.no_grad():
        latent = model.encode(tensor_data.to(device)).cpu().numpy()
    np.save(str(SECTIONS_DIR / "latent_embeddings.npy"), latent)
    print(f"  Latent shape: {latent.shape}")

    create_umap_plot(latent, sections, EVAL_RESULTS_DIR / "umap_latent_clusters.png")
    print("\n✅ Autoencoder training complete!")

if __name__ == "__main__":
    main()
