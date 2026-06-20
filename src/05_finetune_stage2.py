#!/usr/bin/env python3
"""
LexaAI — Step 05: Stage 2 Fine-Tuning on Sri Lankan Law
==========================================================
Fine-tunes the Stage 1 model on synthetic Q&A from 6 Sri Lankan statutes.

Usage:
    python src/05_finetune_stage2.py
Output:
    - models/stage2_lexai/ (model checkpoint)
"""
import json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from src.utils.constants import (
    BASE_MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS_STAGE2, STAGE1_MODEL_DIR, STAGE2_MODEL_DIR, SYNTHETIC_QA_JSON,
)

class LexAIQADataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features["input_ids"])
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.features.items()}

def prepare_qa_features(qa_pairs, tokenizer, max_length=MAX_SEQ_LEN):
    input_ids, attn_masks, starts, ends = [], [], [], []
    skipped = 0
    for pair in tqdm(qa_pairs, desc="Tokenizing"):
        q = pair.get("question", "").strip()
        ctx = pair.get("answer", "").strip()
        span = pair.get("answer_span", "").strip()
        if not q or not ctx:
            skipped += 1
            continue
        enc = tokenizer(q, ctx, max_length=max_length, truncation="only_second",
                        padding="max_length", return_offsets_mapping=True, return_tensors="np")
        offsets = enc["offset_mapping"][0]
        seq_ids = enc.sequence_ids(0)
        s_tok, e_tok = 0, 0
        if span and span in ctx:
            s_char = ctx.find(span)
            e_char = s_char + len(span)
            for idx, (sid, (os_, oe_)) in enumerate(zip(seq_ids, offsets)):
                if sid != 1: continue
                if os_ <= s_char < oe_: s_tok = idx
                if os_ < e_char <= oe_: e_tok = idx; break
        input_ids.append(enc["input_ids"][0].tolist())
        attn_masks.append(enc["attention_mask"][0].tolist())
        starts.append(s_tok)
        ends.append(e_tok)
    if skipped: print(f"  Skipped {skipped} invalid pairs")
    return {"input_ids": input_ids, "attention_mask": attn_masks,
            "start_positions": starts, "end_positions": ends}

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    prog = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_STAGE2}")
    for batch in prog:
        optimizer.zero_grad()
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    start_positions=batch["start_positions"].to(device),
                    end_positions=batch["end_positions"].to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += out.loss.item()
        prog.set_postfix(loss=f"{out.loss.item():.4f}")
    avg = total_loss / len(loader)
    print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")
    return avg

def main():
    print("🔨 LexaAI Stage 2 Fine-Tuning: Sri Lankan Law")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    if not SYNTHETIC_QA_JSON.exists():
        print(f"❌ Run src/04_generate_qa.py first"); sys.exit(1)
    with open(SYNTHETIC_QA_JSON, "r") as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")

    split = int(len(qa_pairs) * 0.9)
    train_pairs, val_pairs = qa_pairs[:split], qa_pairs[split:]

    model_path = str(STAGE1_MODEL_DIR) if STAGE1_MODEL_DIR.exists() else BASE_MODEL_NAME
    print(f"Loading: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)

    feats = prepare_qa_features(train_pairs, tokenizer)
    loader = DataLoader(LexAIQADataset(feats), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(loader) * NUM_EPOCHS_STAGE2
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_loss = float("inf")
    for epoch in range(NUM_EPOCHS_STAGE2):
        loss = train_epoch(model, loader, optimizer, scheduler, device, epoch)
        if loss < best_loss:
            best_loss = loss
            STAGE2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(STAGE2_MODEL_DIR))
            tokenizer.save_pretrained(str(STAGE2_MODEL_DIR))
            print(f"  💾 Best model saved (loss: {best_loss:.4f})")

    meta = {"base": model_path, "train": len(train_pairs), "val": len(val_pairs),
            "epochs": NUM_EPOCHS_STAGE2, "best_loss": best_loss, "device": str(device)}
    with open(STAGE2_MODEL_DIR / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ Stage 2 complete! Model: {STAGE2_MODEL_DIR}")

if __name__ == "__main__":
    main()
