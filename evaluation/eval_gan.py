#!/usr/bin/env python3
"""
LexaAI — Evaluation: GAN Discriminator AUC
=============================================
Evaluates the GAN discriminator's ability to distinguish real from
machine-generated legal text. Target: AUC > 0.80

Usage:
    python evaluation/eval_gan.py
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.constants import (
    SECTIONS_DIR, DISCRIMINATOR_DIR, EVAL_RESULTS_DIR, INPUT_DIM,
)
from src.08_train_discriminator import LegalTextDiscriminator


def main():
    print("📊 GAN DISCRIMINATOR EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    disc_path = DISCRIMINATOR_DIR / "discriminator.pt"
    if not disc_path.exists():
        print("❌ Run src/08_train_discriminator.py first"); sys.exit(1)

    model = LegalTextDiscriminator().to(device)
    model.load_state_dict(torch.load(str(disc_path), map_location=device))
    model.eval()

    # Load embeddings
    emb_path = SECTIONS_DIR / "section_embeddings.npy"
    if not emb_path.exists():
        print("❌ Run src/03_build_embeddings.py first"); sys.exit(1)
    embeddings = np.load(str(emb_path))

    # Create test set: real + fake
    n = min(200, len(embeddings))
    real_embs = embeddings[:n]
    fake_embs = embeddings[:n] + np.random.randn(n, embeddings.shape[1]).astype(np.float32) * 0.3

    X_test = np.vstack([real_embs, fake_embs])
    y_test = np.concatenate([np.ones(n), np.zeros(n)])

    # Predict
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    # Metrics
    auc = roc_auc_score(y_test, preds)
    accuracy = np.mean((preds > 0.5) == y_test)

    print(f"\n  Test samples: {len(X_test)} ({n} real + {n} fake)")
    print(f"  AUC: {auc:.4f} (target: >0.80)")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  {'✅' if auc > 0.80 else '❌'} AUC target met")

    # ROC curve plot
    try:
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_test, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('GAN Discriminator ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(EVAL_RESULTS_DIR / "gan_roc_curve.png"), dpi=150)
        plt.close()
        print(f"  💾 ROC curve: {EVAL_RESULTS_DIR / 'gan_roc_curve.png'}")
    except ImportError:
        pass

    # Save
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_DIR / "gan_results.json", "w") as f:
        json.dump({"auc": auc, "accuracy": float(accuracy),
                    "test_samples": len(X_test)}, f, indent=2)
    print(f"  💾 Results: {EVAL_RESULTS_DIR / 'gan_results.json'}")

if __name__ == "__main__":
    main()
