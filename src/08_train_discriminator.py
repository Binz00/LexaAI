#!/usr/bin/env python3
"""
LexaAI — Step 08: Train GAN Discriminator
============================================
Binary classifier trained to distinguish real lawyer-written legal text
from machine-generated text. Acts as a safety gate — outputs scoring
below 0.75 confidence are rejected before reaching the user.

Usage:
    python src/08_train_discriminator.py
Output:
    - models/discriminator/ (model checkpoint)
"""
import json, sys, random
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.utils.constants import (
    SECTIONS_JSON, SECTIONS_DIR, SYNTHETIC_QA_JSON,
    DISCRIMINATOR_DIR, DISC_EPOCHS, DISC_LR, INPUT_DIM,
)


class LegalTextDiscriminator(nn.Module):
    """Binary classifier: real legal text (1) vs machine-generated (0)."""
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def prepare_discriminator_data(sections, qa_pairs, embeddings):
    """
    Create training data for the discriminator.
    - Real (label=1): Original statute section embeddings
    - Fake (label=0): Noisy/shuffled versions simulating machine-generated text
    """
    n_real = len(embeddings)

    # Real data: actual statute embeddings
    real_embs = embeddings.copy()
    real_labels = np.ones(n_real)

    # Fake data: create synthetic "machine-generated" embeddings
    # Method 1: Add noise to real embeddings
    noise = np.random.randn(*embeddings.shape).astype(np.float32) * 0.3
    noisy_embs = embeddings + noise

    # Method 2: Shuffle embedding dimensions (breaks semantic meaning)
    shuffled_embs = embeddings.copy()
    for i in range(len(shuffled_embs)):
        np.random.shuffle(shuffled_embs[i])

    # Method 3: Random interpolation between unrelated sections
    n_interp = min(n_real, 200)
    interp_embs = []
    for _ in range(n_interp):
        i, j = random.sample(range(n_real), 2)
        alpha = random.random()
        interp_embs.append(embeddings[i] * alpha + embeddings[j] * (1 - alpha))
    interp_embs = np.array(interp_embs, dtype=np.float32)

    # Combine fake data
    fake_embs = np.vstack([noisy_embs[:n_real//2], shuffled_embs[:n_real//2], interp_embs])
    fake_labels = np.zeros(len(fake_embs))

    # Combine all
    all_embs = np.vstack([real_embs, fake_embs])
    all_labels = np.concatenate([real_labels, fake_labels])

    # Shuffle
    indices = np.random.permutation(len(all_embs))
    return all_embs[indices], all_labels[indices]


def main():
    print("🔨 LexaAI GAN Discriminator Training")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    emb_path = SECTIONS_DIR / "section_embeddings.npy"
    if not emb_path.exists():
        print("❌ Run src/03_build_embeddings.py first"); sys.exit(1)
    embeddings = np.load(str(emb_path))

    with open(SECTIONS_JSON) as f:
        sections = json.load(f)
    qa_pairs = []
    if SYNTHETIC_QA_JSON.exists():
        with open(SYNTHETIC_QA_JSON) as f:
            qa_pairs = json.load(f)

    print(f"Sections: {len(sections)}, Embeddings: {embeddings.shape}")

    # Prepare data
    X, y = prepare_discriminator_data(sections, qa_pairs, embeddings)
    print(f"Training data: {X.shape[0]} samples ({int(y.sum())} real, {int(len(y) - y.sum())} fake)")

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=64, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=64)

    # Model
    model = LegalTextDiscriminator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DISC_LR)
    criterion = nn.BCELoss()

    # Train
    print(f"\n🏋️ Training for {DISC_EPOCHS} epochs...")
    for epoch in range(DISC_EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{DISC_EPOCHS} — loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    auc = roc_auc_score(all_labels, all_preds)
    accuracy = np.mean((np.array(all_preds) > 0.5) == np.array(all_labels))

    print(f"\n📊 Results:")
    print(f"  AUC: {auc:.4f} (target: > 0.80)")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  {'✅' if auc > 0.80 else '❌'} AUC target met")

    # Save
    DISCRIMINATOR_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(DISCRIMINATOR_DIR / "discriminator.pt"))
    meta = {"auc": auc, "accuracy": float(accuracy), "epochs": DISC_EPOCHS,
            "train_samples": len(X_train), "test_samples": len(X_test)}
    with open(DISCRIMINATOR_DIR / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾 Saved: {DISCRIMINATOR_DIR}")
    print("✅ Discriminator training complete!")

if __name__ == "__main__":
    main()
