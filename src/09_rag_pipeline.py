#!/usr/bin/env python3
"""
LexaAI — Step 09: RAG Pipeline
=================================
Full inference pipeline: query → encode → hybrid retrieve → rerank → generate → output.

Components:
    1. Legal-BERT tokenizer + encoder (768-dim CLS embeddings)
    2. ChromaDB dense retrieval      (semantic vector search, top-20 candidates)
    3. BM25 sparse retrieval         (keyword-based exact match, top-20 candidates)
    4. Reciprocal Rank Fusion        (merges dense + sparse ranked lists)
    5. Cross-Encoder Reranker        (precise (query, passage) scoring, top-8 final)
    6. Context assembly              (with Chapter injection for Penal Code)
    7. Legal-BERT QA generation      (extractive span prediction)
    8. GAN discriminator safety gate (optional, threshold: 0.75)

Retrieval Pipeline:
    Query
      ├─► Dense (ChromaDB, top-20)  ──┐
      └─► Sparse (BM25, top-20)     ──┴─► RRF merge ──► Cross-Encoder ──► top-8
"""

import json
import sys
import time
import importlib
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering

from src.utils.logger import get_logger
from src.utils.retrieval import BM25Index, CrossEncoderReranker, DomainClassifier, reciprocal_rank_fusion
from src.utils.constants import (
    BASE_MODEL_NAME, STAGE2_MODEL_DIR, STAGE1_MODEL_DIR,
    AUTOENCODER_PATH, DISCRIMINATOR_DIR, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_STATUTES, RETRIEVAL_TOP_K,
    DISC_CONFIDENCE_THRESHOLD, MAX_SEQ_LEN,
    MANDATORY_DISCLAIMER, CRIMINAL_LAW_DISCLAIMER,
    INPUT_DIM, LATENT_DIM,
    DENSE_CANDIDATE_K, BM25_CANDIDATE_K, RRF_K,
    SECTIONS_JSON, SECTIONS_DIR,
)

logger = get_logger(__name__)

# Dynamic imports for numbered modules (Python doesn't allow "from src.06_...")
_ae_mod = importlib.import_module("src.06_train_autoencoder")
LegalSituationAutoencoder = _ae_mod.LegalSituationAutoencoder

_disc_mod = importlib.import_module("src.08_train_discriminator")
LegalTextDiscriminator = _disc_mod.LegalTextDiscriminator


class LexAIRAGPipeline:
    """
    Full RAG inference pipeline for LexaAI with hybrid retrieval and reranking.

    Attributes:
        device:           Torch device (CUDA / MPS / CPU).
        tokenizer:        Legal-BERT tokenizer.
        encoder:          Legal-BERT encoder for CLS embeddings.
        qa_model:         Legal-BERT QA model for span extraction.
        autoencoder:      Optional denoising autoencoder (disabled by default).
        discriminator:    Optional GAN safety gate (disabled by default).
        collection:       ChromaDB collection for dense retrieval.
        bm25:             BM25 index for sparse retrieval.
        reranker:         Cross-encoder reranker for final precision scoring.
        domain_classifier:Semantic fallback domain classifier.
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"🖥️  Device: {self.device}")

        self._load_models()
        self._load_chromadb()
        self._load_bm25()
        self._build_domain_prototypes()
        self._load_reranker()

    # ── Model Loading ──────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load Legal-BERT encoder, QA model, and optional components."""
        # Select best available model checkpoint
        if (STAGE2_MODEL_DIR / "config.json").exists():
            model_path = str(STAGE2_MODEL_DIR)
        elif (STAGE1_MODEL_DIR / "config.json").exists():
            model_path = str(STAGE1_MODEL_DIR)
        else:
            model_path = BASE_MODEL_NAME
        logger.info(f"📦 Loading model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Encoder — used to produce CLS embeddings for dense retrieval
        self.encoder = AutoModel.from_pretrained(model_path).to(self.device)
        self.encoder.eval()

        # QA model — used to extract answer spans from retrieved context
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.qa_model.eval()

        logger.info("✅ Legal-BERT encoder and QA model loaded")

        # Autoencoder (optional — disabled for pure RAG evaluation)
        self.autoencoder = None
        # if AUTOENCODER_PATH.exists():
        #     self.autoencoder = LegalSituationAutoencoder().to(self.device)
        #     self.autoencoder.load_state_dict(torch.load(str(AUTOENCODER_PATH), map_location=self.device))
        #     self.autoencoder.eval()
        #     logger.info("✅ Autoencoder loaded")

        # Discriminator (optional — disabled for pure RAG evaluation)
        self.discriminator = None
        # disc_path = DISCRIMINATOR_DIR / "discriminator.pt"
        # if disc_path.exists():
        #     self.discriminator = LegalTextDiscriminator().to(self.device)
        #     self.discriminator.load_state_dict(torch.load(str(disc_path), map_location=self.device))
        #     self.discriminator.eval()
        #     logger.info("✅ Discriminator loaded")

    def _load_chromadb(self) -> None:
        """Load ChromaDB persistent collection for dense vector retrieval."""
        if not CHROMA_PERSIST_DIR.exists():
            logger.warning("⚠️ ChromaDB not found — dense retrieval disabled")
            self.collection = None
            return

        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        try:
            self.collection = client.get_collection(CHROMA_COLLECTION_STATUTES)
            logger.info(f"✅ ChromaDB: {self.collection.count()} entries")
        except Exception as e:
            self.collection = None
            logger.warning(f"⚠️ ChromaDB collection not found: {e}")

    def _load_bm25(self) -> None:
        """Build BM25 index from sections JSON for sparse keyword retrieval."""
        self.bm25 = BM25Index()

        if not SECTIONS_JSON.exists():
            logger.warning(
                f"⚠️ Sections JSON not found at {SECTIONS_JSON} — BM25 disabled"
            )
            return

        try:
            with open(SECTIONS_JSON, "r", encoding="utf-8") as f:
                sections = json.load(f)

            # Convert sections list to retrieval format expected by BM25Index
            bm25_sections = [
                {
                    "text": s["text"],
                    "meta": {
                        "act": s["act"],
                        "domain": s["domain"],
                        "section_number": s["section_number"],
                        "header": s.get("header", ""),
                        "level1": s.get("level1", ""),
                        "level1_heading": s.get("level1_heading", ""),
                        "char_count": s["char_count"],
                    },
                }
                for s in sections
            ]

            self.bm25.build(bm25_sections)
            logger.info(f"✅ BM25 index built over {len(bm25_sections)} sections")

        except Exception as e:
            logger.error(f"❌ Failed to build BM25 index: {e}", exc_info=True)

    def _build_domain_prototypes(self) -> None:
        """
        Build semantic domain classifier from section embeddings.

        Loads all_sections.json and section_embeddings.npy, then
        computes a mean prototype embedding per domain. Used as a
        semantic fallback when keyword-based domain detection fires nothing.
        """
        self.domain_classifier = DomainClassifier()

        emb_path = SECTIONS_DIR / "section_embeddings.npy"
        if not SECTIONS_JSON.exists() or not emb_path.exists():
            logger.warning(
                "⚠️ section_embeddings.npy or all_sections.json not found — "
                "semantic domain classifier disabled"
            )
            return

        try:
            with open(SECTIONS_JSON, "r", encoding="utf-8") as f:
                sections = json.load(f)
            embeddings = np.load(str(emb_path))
            self.domain_classifier.build(sections, embeddings)
            logger.info("✅ Semantic domain classifier ready")
        except Exception as e:
            logger.error(
                f"❌ Failed to build domain classifier: {e}",
                exc_info=True,
            )

    def _load_reranker(self) -> None:
        """Load cross-encoder reranker for final precision scoring."""
        self.reranker = CrossEncoderReranker()
        try:
            self.reranker.load()
            logger.info("✅ Cross-encoder reranker ready")
        except Exception as e:
            logger.error(
                f"❌ Failed to load cross-encoder reranker: {e}. "
                "Falling back to RRF ordering only.",
                exc_info=True,
            )

    # ── Embedding ──────────────────────────────────────────────────

    def _get_cls_embedding(self, query: str) -> torch.Tensor:
        """
        Generate a 768-dim [CLS] token embedding for the query.

        Uses the last hidden state's [CLS] token (position 0) as the
        sentence-level representation of the query.

        Args:
            query: The user's natural language query.

        Returns:
            Tensor of shape (1, 768) on CPU.
        """
        inputs = self.tokenizer(
            query, max_length=MAX_SEQ_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()

        logger.debug(
            f"[CLS] Embedding → Shape: {cls_emb.shape}, "
            f"L2 Norm: {torch.norm(cls_emb):.4f}"
        )
        return cls_emb

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query for dense retrieval.

        If the autoencoder is active, compresses 768-dim → 256-dim latent.
        Otherwise returns the raw 768-dim CLS embedding.

        Args:
            query: The user's natural language query.

        Returns:
            1-D NumPy float array of embedding dimension.
        """
        logger.debug(f"[Encode] Query: '{query[:80]}...'")
        cls_emb = self._get_cls_embedding(query)

        if self.autoencoder:
            with torch.no_grad():
                z = self.autoencoder.encode(cls_emb.to(self.device)).cpu()
            final_emb = z.numpy().flatten()
            logger.debug(
                f"[Encode] Autoencoder applied → "
                f"Shape: {final_emb.shape}, Norm: {np.linalg.norm(final_emb):.4f}"
            )
            return final_emb

        final_emb = cls_emb.numpy().flatten()
        logger.debug(
            f"[Encode] Raw CLS → Shape: {final_emb.shape}, "
            f"Norm: {np.linalg.norm(final_emb):.4f}"
        )
        return final_emb

    # ── Dense Retrieval ────────────────────────────────────────────

    def _dense_retrieve(
        self, query_emb: np.ndarray, top_k: int, where_filter: Optional[dict]
    ) -> list[dict]:
        """
        Query ChromaDB with the dense query embedding.

        Applies an optional domain filter and falls back to a global
        search if the filtered search returns no results.

        Args:
            query_emb:    1-D numpy embedding for the query.
            top_k:        Number of candidates to retrieve.
            where_filter: Optional ChromaDB metadata filter dict.

        Returns:
            List of section dicts with distance scores.
        """
        if not self.collection:
            logger.warning("[Dense] ChromaDB not available.")
            return []

        logger.debug(
            f"[Dense] Querying ChromaDB for top {top_k} | filter: {where_filter}"
        )
        t0 = time.perf_counter()

        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            where=where_filter,
        )

        # Fallback to global search if domain filter yields nothing
        if not results or not results["documents"] or len(results["documents"][0]) == 0:
            if where_filter:
                logger.debug(
                    "[Dense] Domain-filtered search returned 0 results. "
                    "Falling back to global search."
                )
                results = self.collection.query(
                    query_embeddings=[query_emb.tolist()],
                    n_results=top_k,
                )

        elapsed = (time.perf_counter() - t0) * 1000
        sections = []
        if results and results["documents"] and len(results["documents"][0]) > 0:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                logger.debug(
                    f"[Dense] Rank {i+1} | Distance: {dist:.4f} | "
                    f"Act: {meta.get('act')} | Section: {meta.get('section_number')} | "
                    f"Domain: {meta.get('domain')}"
                )
                sections.append({
                    "text": doc,
                    "meta": meta,
                    "distance": dist,
                    "dense_rank": i + 1,
                })
        else:
            logger.debug("[Dense] No results returned from ChromaDB.")

        logger.info(
            f"[Dense] Retrieved {len(sections)} candidates in {elapsed:.1f}ms"
        )
        return sections

    # ── Hybrid Retrieve + Rerank ───────────────────────────────────

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        """
        Full hybrid retrieval pipeline: Dense + BM25 → RRF → Reranker.

        Steps:
            1. Detect legal domain from query keywords.
            2. Dense retrieval (ChromaDB cosine, top-20 candidates).
            3. Sparse retrieval (BM25 keyword, top-20 candidates).
            4. Reciprocal Rank Fusion (merge both ranked lists).
            5. Cross-encoder reranking (score fused top-20, return top-8).

        Args:
            query: The user's natural language query.
            top_k: Number of final sections to return after reranking.

        Returns:
            List of top-k section dicts with retrieval metadata.
        """
        if not self.collection and not self.bm25.is_ready:
            logger.error("[Retrieve] Neither ChromaDB nor BM25 is available.")
            return []

        logger.info(f"[Retrieve] Starting hybrid retrieval for: '{query[:80]}...'")
        t_total = time.perf_counter()

        # ── Step 1a: Keyword-based domain detection ───────────────
        from src.utils.legal_keywords import DOMAIN_KEYWORDS
        detected_domains = []
        q_lower = query.lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(k in q_lower for k in keywords):
                detected_domains.append(domain)

        if detected_domains:
            logger.info(
                f"[Retrieve] Keyword detection → domains: {detected_domains}"
            )
        else:
            logger.info(
                "[Retrieve] Keyword detection fired nothing — "
                "falling back to semantic domain classifier."
            )

        # ── Step 1b: Semantic fallback (if keywords missed) ───────
        query_emb = self.encode_query(query)
        if not detected_domains and self.domain_classifier.is_ready:
            detected_domains = self.domain_classifier.classify(query_emb)
            if detected_domains:
                logger.info(
                    f"[Retrieve] Semantic classifier → domains: {detected_domains}"
                )
            else:
                logger.info(
                    "[Retrieve] Semantic classifier also found no confident domain. "
                    "Running global retrieval (no filter)."
                )

        where_filter = None
        if len(detected_domains) == 1:
            where_filter = {"domain": detected_domains[0]}
        elif len(detected_domains) > 1:
            where_filter = {"$or": [{"domain": d} for d in detected_domains]}

        logger.info(
            f"[Retrieve] Final filter: {where_filter}"
        )
        dense_results = self._dense_retrieve(query_emb, DENSE_CANDIDATE_K, where_filter)

        # ── Step 3: Sparse BM25 retrieval ─────────────────────────
        sparse_results: list[dict] = []
        if self.bm25.is_ready:
            sparse_results = self.bm25.search(query, top_k=BM25_CANDIDATE_K)
            # Apply domain filter to BM25 results if detected
            if detected_domains:
                before = len(sparse_results)
                sparse_results = [
                    s for s in sparse_results
                    if s["meta"].get("domain") in detected_domains
                ]
                logger.debug(
                    f"[BM25] Domain filter applied: {before} → {len(sparse_results)} results"
                )
        else:
            logger.warning("[BM25] Index not ready — skipping sparse retrieval.")

        # ── Step 4: Reciprocal Rank Fusion ────────────────────────
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=RRF_K,
            top_n=max(DENSE_CANDIDATE_K, BM25_CANDIDATE_K),
        )

        if not fused:
            logger.warning("[Retrieve] RRF returned no candidates.")
            return []

        # ── Step 5: Cross-encoder reranking ───────────────────────
        final_sections = self.reranker.rerank(query, fused, top_k=top_k)

        total_elapsed = (time.perf_counter() - t_total) * 1000
        logger.info(
            f"[Retrieve] Pipeline complete: {len(final_sections)} final sections "
            f"| Total latency: {total_elapsed:.1f}ms"
        )

        return final_sections

    # ── Context Assembly ───────────────────────────────────────────

    def build_context(self, sections: list[dict]) -> str:
        """
        Build the RAG context string from the final retrieved sections.

        Injects Chapter/Part headings for criminal law sections to give
        the QA model hierarchical legal structure context.

        Args:
            sections: Final ranked sections from retrieval pipeline.

        Returns:
            Concatenated context string separated by horizontal rules.
        """
        contexts = []
        for s in sections:
            meta = s["meta"]
            prefix = f"[{meta['act']} — {meta['section_number']}]"
            if meta.get("domain") == "criminal_law" and meta.get("level1_heading"):
                prefix = f"[{meta['level1_heading']}] {prefix}"
            contexts.append(f"{prefix}\n{s['text']}")
        context = "\n\n---\n\n".join(contexts)
        logger.debug(f"[Context] Built context: {len(context)} chars from {len(sections)} sections")
        return context

    # ── Answer Generation ──────────────────────────────────────────

    def generate_answer(self, query: str, context: str) -> dict:
        """
        Generate an extractive QA answer from the assembled context.

        Uses Legal-BERT's QA head to predict start/end token positions
        for the answer span within the context.

        Args:
            query:   The user's natural language query.
            context: The assembled statute text context.

        Returns:
            Dict with 'answer', 'confidence', 'start_idx', 'end_idx'.
        """
        logger.debug(f"[QA] Generating answer | Context length: {len(context)} chars")
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            query, context[:3000],
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

        start_conf = torch.softmax(outputs.start_logits, dim=1).max().item()
        end_conf = torch.softmax(outputs.end_logits, dim=1).max().item()
        confidence = (start_conf + end_conf) / 2

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[QA] Answer generated in {elapsed:.1f}ms | "
            f"Confidence: {confidence:.4f} | "
            f"Span: [{start_idx}, {end_idx}] | "
            f"Answer: '{answer[:80]}'"
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }

    # ── Safety Check ───────────────────────────────────────────────

    def check_safety(self, query: str) -> dict:
        """
        Run the GAN discriminator safety check on the query embedding.

        When the discriminator is disabled (default), returns a
        pass-through safe result.

        Args:
            query: The user's natural language query.

        Returns:
            Dict with 'safe' (bool), 'score' (float), 'reason' (str).
        """
        if not self.discriminator:
            logger.debug("[Safety] Discriminator disabled — bypassing safety gate.")
            return {"safe": True, "score": 1.0, "reason": "No discriminator loaded"}

        cls_emb = self._get_cls_embedding(query)
        with torch.no_grad():
            score = self.discriminator(cls_emb.to(self.device)).item()

        safe = score >= DISC_CONFIDENCE_THRESHOLD
        logger.info(
            f"[Safety] Discriminator score: {score:.4f} | "
            f"Threshold: {DISC_CONFIDENCE_THRESHOLD} | Safe: {safe}"
        )
        return {
            "safe": safe,
            "score": score,
            "reason": "Passed" if safe
                      else f"Below threshold ({score:.3f} < {DISC_CONFIDENCE_THRESHOLD})",
        }

    # ── Full Pipeline ──────────────────────────────────────────────

    def query(self, user_query: str, output_mode: str = "plain_rights_summary") -> dict:
        """
        Execute the full RAG pipeline for a user query.

        Pipeline steps:
            1. Hybrid retrieve (Dense + BM25 → RRF → Reranker)
            2. Build context from final sections
            3. Extract answer via Legal-BERT QA
            4. Run safety check (GAN discriminator, if loaded)
            5. Determine disclaimers (mandatory + criminal law if applicable)

        Args:
            user_query:  The user's natural language legal query.
            output_mode: Controls downstream output formatting.

        Returns:
            Dict containing answer, sections, context, safety, disclaimers, etc.
        """
        logger.info(f"[Pipeline] Query received: '{user_query[:100]}...'")
        t_pipeline = time.perf_counter()

        # 1. Hybrid retrieval
        sections = self.retrieve(user_query)

        # 2. Build context
        context = self.build_context(sections) if sections else ""

        # 3. Generate answer
        if context:
            qa_result = self.generate_answer(user_query, context)
        else:
            logger.warning("[Pipeline] No context retrieved — returning fallback answer.")
            qa_result = {
                "answer": "No relevant statute sections found for your query.",
                "confidence": 0.0,
            }

        # 4. Safety check
        safety = self.check_safety(user_query)

        # 5. Disclaimers
        disclaimers = [MANDATORY_DISCLAIMER]
        has_criminal = any(
            s["meta"].get("domain") == "criminal_law" for s in sections
        )
        if has_criminal:
            disclaimers.insert(0, CRIMINAL_LAW_DISCLAIMER)
            logger.info("[Pipeline] Criminal law sections detected — adding criminal disclaimer.")

        total_elapsed = (time.perf_counter() - t_pipeline) * 1000
        logger.info(
            f"[Pipeline] Complete | Sections: {len(sections)} | "
            f"Confidence: {qa_result.get('confidence', 0):.4f} | "
            f"Safe: {safety['safe']} | "
            f"Total latency: {total_elapsed:.1f}ms"
        )

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


# ── Singleton for Streamlit ────────────────────────────────────────
_pipeline: Optional[LexAIRAGPipeline] = None


def get_pipeline() -> LexAIRAGPipeline:
    """Return the shared pipeline singleton (creates it on first call)."""
    global _pipeline
    if _pipeline is None:
        logger.info("[Singleton] Initialising LexAIRAGPipeline...")
        _pipeline = LexAIRAGPipeline()
        logger.info("[Singleton] Pipeline ready.")
    return _pipeline


# ── CLI Test Mode ──────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("🔨 LexaAI RAG Pipeline — Interactive Test (Hybrid + Reranker)")
    pipeline = LexAIRAGPipeline()
    while True:
        user_q = input("\n❓ Enter legal query (or 'quit'): ").strip()
        if user_q.lower() in ("quit", "exit", "q"):
            break
        result = pipeline.query(user_q)
        logger.info(f"📋 Answer: {result['answer']}")
        logger.info(f"🔒 Safety: {result['safety']}")
        logger.info(f"📚 Sections retrieved: {len(result['sections'])}")
        for d in result["disclaimers"]:
            logger.warning(f"⚠️ {d}")
