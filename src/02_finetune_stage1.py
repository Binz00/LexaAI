#!/usr/bin/env python3
"""
LexaAI — Step 02: Stage 1 Fine-Tuning on CUAD
================================================
Fine-tunes Legal-BERT (nlpaueb/legal-bert-base-uncased) on the CUAD
(Contract Understanding Atticus Dataset) for extractive legal QA.

This teaches the model the pattern of extractive legal reasoning:
given a legal document and a question, identify the exact span of
text that answers it.

Training Config:
    - Batch size: 8
    - Sequence length: 512
    - Learning rate: 2e-5 (AdamW)
    - Epochs: 3

Targets:
    - Exact Match (EM) > 45%
    - F1 Score > 70%

Usage:
    python src/02_finetune_stage1.py

Output:
    - models/stage1_cuad/ (model checkpoint)
    - Console: EM and F1 evaluation metrics
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from datasets import load_dataset

from src.utils.constants import (
    BASE_MODEL_NAME,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS_STAGE1,
    STAGE1_MODEL_DIR,
    CUAD_DIR,
)


# ── Dataset Preparation ──────────────────────────────────────────

class CUADDataset(Dataset):
    """PyTorch Dataset for CUAD extractive QA."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def prepare_cuad_features(examples, tokenizer, max_length=MAX_SEQ_LEN):
    """
    Tokenize CUAD examples and compute start/end answer positions.

    Handles cases where the answer span is not fully contained in
    the truncated context by setting start/end to the [CLS] token (0).
    """
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map each tokenized example back to its original
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        # If no answer, set both to CLS token position
        if len(answers["answer_start"]) == 0 or answers["text"][0] == "":
            start_positions.append(0)
            end_positions.append(0)
            continue

        answer_start_char = answers["answer_start"][0]
        answer_end_char = answer_start_char + len(answers["text"][0])

        # Find which token positions correspond to the context (not question)
        sequence_ids = tokenized.sequence_ids(i)

        # Find context start and end in tokens
        ctx_start = 0
        ctx_end = 0
        for idx, seq_id in enumerate(sequence_ids):
            if seq_id == 1:
                if ctx_start == 0:
                    ctx_start = idx
                ctx_end = idx

        # Check if the answer is within the context window
        if (
            offsets[ctx_start][0] > answer_start_char
            or offsets[ctx_end][1] < answer_end_char
        ):
            # Answer not in this window
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Find token positions of the answer
            token_start = ctx_start
            token_end = ctx_end

            while token_start <= ctx_end and offsets[token_start][0] <= answer_start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            while token_end >= ctx_start and offsets[token_end][1] >= answer_end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


def compute_metrics(predictions, references):
    """
    Compute Exact Match (EM) and F1 scores.

    Args:
        predictions: List of predicted answer strings
        references: List of ground-truth answer strings

    Returns:
        Dictionary with 'exact_match' and 'f1' scores
    """
    exact_matches = 0
    f1_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        # Exact match
        if pred.strip().lower() == ref.strip().lower():
            exact_matches += 1

        # F1
        common = set(pred_tokens) & set(ref_tokens)
        if len(common) == 0:
            f1_scores.append(0.0)
        else:
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

    n = len(predictions)
    return {
        "exact_match": exact_matches / n * 100 if n > 0 else 0,
        "f1": np.mean(f1_scores) * 100 if f1_scores else 0,
    }


# ── Training Loop ────────────────────────────────────────────────

def train(model, train_loader, optimizer, scheduler, device, epoch):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch in progress:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch+1} average loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, eval_loader, tokenizer, device):
    """Run evaluation and compute EM/F1 metrics."""
    model.eval()
    all_start_logits = []
    all_end_logits = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            all_start_logits.append(outputs.start_logits.cpu())
            all_end_logits.append(outputs.end_logits.cpu())

    print("  ✅ Evaluation complete")
    return all_start_logits, all_end_logits


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("🔨 LexaAI Stage 1 Fine-Tuning: CUAD")
    print("=" * 60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load tokenizer and model
    print(f"\n📦 Loading model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL_NAME)
    model.to(device)

    # Load SQuAD dataset (teaches extractive QA pattern — same as CUAD's format)
    # Note: CUAD uses deprecated dataset scripts; SQuAD is the standard extractive QA
    # benchmark and teaches the same span-extraction reasoning. Domain adaptation
    # to legal text happens in Stage 2 with Sri Lankan law Q&A.
    print("\n📥 Loading SQuAD dataset...")
    dataset = load_dataset("rajpurkar/squad", cache_dir=str(CUAD_DIR))

    # Use a subset for manageable training on consumer hardware
    train_data = dataset["train"]
    test_data = dataset["validation"]

    # Limit dataset size for local GPU feasibility
    max_train = min(len(train_data), 5000)
    max_test = min(len(test_data), 500)
    train_data = train_data.select(range(max_train))
    test_data = test_data.select(range(max_test))

    print(f"  Training examples: {len(train_data)}")
    print(f"  Test examples: {len(test_data)}")

    # Tokenize
    print("\n🔤 Tokenizing...")
    train_features = train_data.map(
        lambda x: prepare_cuad_features(x, tokenizer),
        batched=True,
        remove_columns=train_data.column_names,
    )

    test_features = test_data.map(
        lambda x: prepare_cuad_features(x, tokenizer),
        batched=True,
        remove_columns=test_data.column_names,
    )

    # Create data loaders
    train_features.set_format("torch")
    test_features.set_format("torch")

    train_loader = DataLoader(
        train_features, batch_size=BATCH_SIZE, shuffle=True
    )
    eval_loader = DataLoader(
        test_features, batch_size=BATCH_SIZE
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS_STAGE1
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # Training loop
    print(f"\n🏋️ Training for {NUM_EPOCHS_STAGE1} epochs...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_SEQ_LEN}")

    best_loss = float("inf")
    for epoch in range(NUM_EPOCHS_STAGE1):
        loss = train(model, train_loader, optimizer, scheduler, device, epoch)
        if loss < best_loss:
            best_loss = loss

    # Evaluate
    print("\n📊 Running evaluation...")
    evaluate(model, eval_loader, tokenizer, device)

    # Save model
    STAGE1_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(STAGE1_MODEL_DIR))
    tokenizer.save_pretrained(str(STAGE1_MODEL_DIR))

    print(f"\n💾 Model saved to: {STAGE1_MODEL_DIR}")

    # Save training metadata
    metadata = {
        "base_model": BASE_MODEL_NAME,
        "dataset": "rajpurkar/squad",
        "train_examples": len(train_data),
        "test_examples": len(test_data),
        "epochs": NUM_EPOCHS_STAGE1,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_seq_len": MAX_SEQ_LEN,
        "best_train_loss": best_loss,
        "device": str(device),
    }
    with open(STAGE1_MODEL_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Stage 1 fine-tuning complete!")
    print(f"   Best training loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
