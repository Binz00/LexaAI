#!/usr/bin/env python3
"""
LexaAI — Step 09: RAG Pipeline
=================================
Full inference pipeline: query → encode → retrieve → generate → discriminate → output.

Components:
    1. Legal-BERT tokenizer + encoder
    2. Denoising autoencoder (768 → 256 latent)
    3. ChromaDB retrieval (top-8 sections)
    4. Context assembly (with Chapter injection for Penal Code)
    5. Legal-BERT QA generation
    6. GAN discriminator safety gate (threshold: 0.75)
"""
import json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib

import numpy as np
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering

from src.utils.constants import (
    BASE_MODEL_NAME, STAGE2_MODEL_DIR, STAGE1_MODEL_DIR,
    AUTOENCODER_PATH, DISCRIMINATOR_DIR, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_STATUTES, RETRIEVAL_TOP_K,
    DISC_CONFIDENCE_THRESHOLD, MAX_SEQ_LEN,
    MANDATORY_DISCLAIMER, CRIMINAL_LAW_DISCLAIMER,
    INPUT_DIM, LATENT_DIM,
)

# Dynamic imports for numbered modules (Python doesn't allow "from src.06_...")
_ae_mod = importlib.import_module("src.06_train_autoencoder")
LegalSituationAutoencoder = _ae_mod.LegalSituationAutoencoder

_disc_mod = importlib.import_module("src.08_train_discriminator")
LegalTextDiscriminator = _disc_mod.LegalTextDiscriminator


class LexAIRAGPipeline:
    """Full RAG inference pipeline for LexAI."""

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._load_models()
        self._load_chromadb()

    def _load_models(self):
        """Load all model components."""
        # Determine best available model (check config.json exists, not just directory)
        if (STAGE2_MODEL_DIR / "config.json").exists():
            model_path = str(STAGE2_MODEL_DIR)
        elif (STAGE1_MODEL_DIR / "config.json").exists():
            model_path = str(STAGE1_MODEL_DIR)
        else:
            model_path = BASE_MODEL_NAME
        print(f"📦 Loading model: {model_path}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Encoder (for embeddings)
        self.encoder = AutoModel.from_pretrained(model_path).to(self.device)
        self.encoder.eval()

        # QA model
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.qa_model.eval()

        # Autoencoder (optional)
        self.autoencoder = None
        if AUTOENCODER_PATH.exists():
            self.autoencoder = LegalSituationAutoencoder().to(self.device)
            self.autoencoder.load_state_dict(torch.load(str(AUTOENCODER_PATH), map_location=self.device))
            self.autoencoder.eval()
            print("  ✅ Autoencoder loaded")

        # Discriminator (optional)
        self.discriminator = None
        disc_path = DISCRIMINATOR_DIR / "discriminator.pt"
        if disc_path.exists():
            self.discriminator = LegalTextDiscriminator().to(self.device)
            self.discriminator.load_state_dict(torch.load(str(disc_path), map_location=self.device))
            self.discriminator.eval()
            print("  ✅ Discriminator loaded")

    def _load_chromadb(self):
        """Load ChromaDB collection."""
        if not CHROMA_PERSIST_DIR.exists():
            print("⚠️  ChromaDB not found — retrieval disabled")
            self.collection = None
            return
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        try:
            self.collection = client.get_collection(CHROMA_COLLECTION_STATUTES)
            print(f"  ✅ ChromaDB: {self.collection.count()} entries")
        except Exception:
            self.collection = None
            print("⚠️  ChromaDB collection not found")

    def _get_cls_embedding(self, query: str) -> torch.Tensor:
        """Get raw [CLS] embedding (768-dim) for a query."""
        inputs = self.tokenizer(
            query, max_length=MAX_SEQ_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu()

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query for retrieval (256-dim if autoencoder available, else 768-dim)."""
        cls_emb = self._get_cls_embedding(query)
        if self.autoencoder:
            with torch.no_grad():
                z = self.autoencoder.encode(cls_emb.to(self.device)).cpu()
            return z.numpy().flatten()
        return cls_emb.numpy().flatten()

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        """Retrieve relevant statute sections with domain-aware boosting."""
        if not self.collection:
            return []

        # 1. Detect target domain based on keywords
        from src.utils.legal_keywords import DOMAIN_KEYWORDS
        detected_domains = []
        q_lower = query.lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(k in q_lower for k in keywords):
                detected_domains.append(domain)

        # 2. Build where filter if domains detected
        where_filter = None
        if len(detected_domains) == 1:
            where_filter = {"domain": detected_domains[0]}
        elif len(detected_domains) > 1:
            where_filter = {"$or": [{"domain": d} for d in detected_domains]}

        # 3. Perform query
        query_emb = self.encode_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            where=where_filter,
        )

        # 4. Fallback: if filtered search returns nothing, try global search
        if not results or not results["documents"] or len(results["documents"][0]) == 0:
            if where_filter:
                results = self.collection.query(
                    query_embeddings=[query_emb.tolist()],
                    n_results=top_k,
                )

        sections = []
        if results and results["documents"] and len(results["documents"][0]) > 0:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                sections.append({
                    "text": doc,
                    "meta": meta,
                    "distance": dist,
                })
        return sections

    def build_context(self, sections: list[dict]) -> str:
        """Build RAG context from retrieved sections, with Chapter injection for Penal Code."""
        contexts = []
        for s in sections:
            meta = s["meta"]
            prefix = f"[{meta['act']} — {meta['section_number']}]"
            # Inject Chapter heading for criminal law sections
            if meta.get("domain") == "criminal_law" and meta.get("level1_heading"):
                prefix = f"[{meta['level1_heading']}] {prefix}"
            contexts.append(f"{prefix}\n{s['text']}")
        return "\n\n---\n\n".join(contexts)

    def generate_answer(self, query: str, context: str) -> dict:
        """Generate extractive QA answer from context."""
        inputs = self.tokenizer(
            query, context[:3000],  # Truncate context for token limit
            max_length=MAX_SEQ_LEN, padding="max_length",
            truncation="only_second", return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.qa_model(**inputs)
        start_idx = torch.argmax(outputs.start_logits, dim=1).item()
        end_idx = torch.argmax(outputs.end_logits, dim=1).item()

        if end_idx < start_idx:
            end_idx = start_idx

        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Confidence score
        start_conf = torch.softmax(outputs.start_logits, dim=1).max().item()
        end_conf = torch.softmax(outputs.end_logits, dim=1).max().item()
        confidence = (start_conf + end_conf) / 2

        return {"answer": answer, "confidence": confidence,
                "start_idx": start_idx, "end_idx": end_idx}

    def check_safety(self, query: str) -> dict:
        """Run discriminator safety check using raw 768-dim embedding."""
        if not self.discriminator:
            return {"safe": True, "score": 1.0, "reason": "No discriminator loaded"}
        # Discriminator expects raw 768-dim [CLS] embedding (not compressed)
        cls_emb = self._get_cls_embedding(query)
        with torch.no_grad():
            score = self.discriminator(cls_emb.to(self.device)).item()
        return {
            "safe": score >= DISC_CONFIDENCE_THRESHOLD,
            "score": score,
            "reason": "Passed" if score >= DISC_CONFIDENCE_THRESHOLD
                      else f"Below threshold ({score:.3f} < {DISC_CONFIDENCE_THRESHOLD})",
        }

    def query(self, user_query: str, output_mode: str = "plain_rights_summary") -> dict:
        """
        Full pipeline: query → encode → retrieve → generate → discriminate.

        Returns a dict with: answer, sections, disclaimers, safety, output_mode.
        """
        # 1. Retrieve relevant sections
        sections = self.retrieve(user_query)

        # 2. Build context
        context = self.build_context(sections) if sections else ""

        # 3. Generate answer
        qa_result = self.generate_answer(user_query, context) if context else {
            "answer": "No relevant statute sections found for your query.",
            "confidence": 0.0,
        }

        # 4. Safety check
        safety = self.check_safety(user_query)

        # 5. Determine disclaimers
        disclaimers = [MANDATORY_DISCLAIMER]
        has_criminal = any(
            s["meta"].get("domain") == "criminal_law" for s in sections
        )
        if has_criminal:
            disclaimers.insert(0, CRIMINAL_LAW_DISCLAIMER)

        # 6. Build response
        return {
            "query": user_query,
            "answer": qa_result["answer"],
            "confidence": qa_result.get("confidence", 0),
            "output_mode": output_mode,
            "sections": sections,
            "context": context,
            "safety": safety,
            "disclaimers": disclaimers,
            "has_criminal_law": has_criminal,
        }


# Singleton for the Streamlit app
_pipeline = None

def get_pipeline() -> LexAIRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LexAIRAGPipeline()
    return _pipeline


if __name__ == "__main__":
    print("🔨 LexaAI RAG Pipeline — Interactive Test")
    print("=" * 60)
    pipeline = LexAIRAGPipeline()
    while True:
        query = input("\n❓ Enter legal query (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        result = pipeline.query(query)
        print(f"\n📋 Answer: {result['answer']}")
        print(f"🔒 Safety: {result['safety']}")
        print(f"📚 Sections retrieved: {len(result['sections'])}")
        for d in result["disclaimers"]:
            print(f"  ⚠️  {d}")
