"""
LexaAI — Hybrid Retrieval Engine
===================================
Implements a production-grade retrieval pipeline for the LexaAI RAG system:

    1. Dense Retrieval     — ChromaDB cosine similarity (semantic meaning)
    2. Sparse Retrieval    — BM25 keyword matching (exact legal terminology)
    3. RRF Fusion          — Reciprocal Rank Fusion merges both ranked lists
    4. Cross-Encoder Reranker — Precise (query, passage) relevance scoring

Architecture:
    Query
      ├─► Dense (ChromaDB, top-20)  ──┐
      └─► Sparse (BM25, top-20)     ──┴─► RRF merge (top-20) ──► Reranker ──► top-8
"""

import time
from typing import Optional

import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import (
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
    RETRIEVAL_TOP_K,
    RRF_K,
    RERANKER_MODEL,
    RERANKER_MAX_LENGTH,
)

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# BM25 Index
# ──────────────────────────────────────────────────────────────────

class BM25Index:
    """
    BM25 sparse retrieval index over statute sections.

    Tokenises each section's text at build time and supports
    fast keyword-based ranking at query time via the rank_bm25
    library. This complements dense vector search by catching
    exact legal term matches (e.g. "notarial execution", "Section 2A").
    """

    def __init__(self):
        self._bm25 = None
        self._sections: list[dict] = []

    def build(self, sections: list[dict]) -> None:
        """
        Build the BM25 index from a list of section dicts.

        Args:
            sections: List of dicts with at least a 'text' key.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error(
                "rank_bm25 not installed. Run: pip install rank-bm25"
            )
            raise

        self._sections = sections
        logger.info(f"[BM25] Building index over {len(sections)} sections...")
        t0 = time.perf_counter()

        # Simple whitespace tokeniser — adequate for legal statute text
        tokenised_corpus = [
            s["text"].lower().split() for s in sections
        ]
        self._bm25 = BM25Okapi(tokenised_corpus)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[BM25] Index built in {elapsed:.1f}ms "
            f"(avg doc length: {np.mean([len(t) for t in tokenised_corpus]):.1f} tokens)"
        )

    def search(self, query: str, top_k: int = BM25_CANDIDATE_K) -> list[dict]:
        """
        Retrieve top-k sections by BM25 score.

        Args:
            query: The user's natural language query.
            top_k: Number of candidates to return.

        Returns:
            List of dicts: {text, meta, bm25_score, bm25_rank}
        """
        if self._bm25 is None:
            logger.warning("[BM25] Index not built. Call build() first.")
            return []

        t0 = time.perf_counter()
        tokenised_query = query.lower().split()
        scores = self._bm25.get_scores(tokenised_query)

        # Get top-k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"[BM25] Query '{query[:60]}...' scored {len(scores)} docs in {elapsed:.1f}ms"
        )

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                # BM25 score of 0 means no keyword overlap — stop early
                logger.debug(
                    f"[BM25] Stopping at rank {rank + 1}: BM25 score dropped to 0"
                )
                break
            s = self._sections[idx]
            logger.debug(
                f"[BM25] Rank {rank + 1} | Score: {score:.4f} | "
                f"Act: {s['meta'].get('act')} | Section: {s['meta'].get('section_number')}"
            )
            results.append({
                "text": s["text"],
                "meta": s["meta"],
                "bm25_score": score,
                "bm25_rank": rank + 1,
            })

        logger.info(
            f"[BM25] Retrieved {len(results)} candidates "
            f"(top score: {float(scores[top_indices[0]]):.4f})"
        )
        return results

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None


# ──────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
    top_n: int = DENSE_CANDIDATE_K,
) -> list[dict]:
    """
    Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

    RRF Score = sum(1 / (k + rank_i)) for each list that contains
    the document. This is parameter-robust and outperforms weighted
    score combination in most retrieval benchmarks.

    Args:
        dense_results: Ranked list from ChromaDB dense search.
        sparse_results: Ranked list from BM25 sparse search.
        k: RRF smoothing constant (default 60, per original paper).
        top_n: Number of fused candidates to return.

    Returns:
        Merged list of section dicts with an 'rrf_score' key, sorted
        by descending RRF score.
    """
    logger.debug(
        f"[RRF] Fusing {len(dense_results)} dense + {len(sparse_results)} sparse results "
        f"(k={k}, top_n={top_n})"
    )
    t0 = time.perf_counter()

    # Build a lookup by section ID for deduplication
    # Key: (act, section_number) tuple — unique identifier for a statute section
    all_sections: dict[tuple, dict] = {}
    rrf_scores: dict[tuple, float] = {}

    def _section_key(s: dict) -> tuple:
        meta = s.get("meta", {})
        return (meta.get("act", ""), meta.get("section_number", ""))

    # Accumulate RRF scores from dense list
    for rank, section in enumerate(dense_results, start=1):
        key = _section_key(section)
        rrf_score = 1.0 / (k + rank)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score
        if key not in all_sections:
            all_sections[key] = section
        logger.debug(
            f"[RRF] Dense rank {rank} | Key: {key} | Contribution: {rrf_score:.5f}"
        )

    # Accumulate RRF scores from sparse list
    for rank, section in enumerate(sparse_results, start=1):
        key = _section_key(section)
        rrf_score = 1.0 / (k + rank)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score
        if key not in all_sections:
            all_sections[key] = section
        logger.debug(
            f"[RRF] Sparse rank {rank} | Key: {key} | Contribution: {rrf_score:.5f}"
        )

    # Sort all candidates by descending RRF score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k_: rrf_scores[k_], reverse=True)

    merged = []
    for rank, key in enumerate(sorted_keys[:top_n], start=1):
        section = dict(all_sections[key])
        section["rrf_score"] = rrf_scores[key]
        section["rrf_rank"] = rank
        merged.append(section)
        logger.debug(
            f"[RRF] Fused rank {rank} | Score: {rrf_scores[key]:.5f} | Key: {key}"
        )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[RRF] Fusion complete: {len(merged)} unique candidates "
        f"from {len(dense_results) + len(sparse_results)} total in {elapsed:.1f}ms"
    )
    return merged


# ──────────────────────────────────────────────────────────────────
# Cross-Encoder Reranker
# ──────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Cross-Encoder based reranker for precise (query, passage) relevance scoring.

    Unlike bi-encoders (which embed query and passage independently),
    a cross-encoder processes both together in a single forward pass,
    allowing it to model fine-grained interactions between query tokens
    and passage tokens. This produces significantly more accurate
    relevance scores at the cost of higher latency.

    Used as the final stage of the retrieval pipeline on the top-N
    candidates from RRF fusion.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        """Lazy-load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            raise

        logger.info(f"[Reranker] Loading cross-encoder: {self.model_name}")
        t0 = time.perf_counter()
        self._model = CrossEncoder(
            self.model_name,
            max_length=RERANKER_MAX_LENGTH,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"[Reranker] Cross-encoder loaded in {elapsed:.0f}ms")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict]:
        """
        Rerank candidate sections using the cross-encoder and return top_k.

        Each (query, section_text) pair is scored independently. The
        cross-encoder score is a raw logit — higher means more relevant.

        Args:
            query: The user's natural language query.
            candidates: List of section dicts from RRF fusion.
            top_k: Number of final sections to return.

        Returns:
            Top-k sections sorted by descending cross-encoder score,
            each with a 'reranker_score' and 'reranker_rank' key.
        """
        if not candidates:
            logger.warning("[Reranker] No candidates to rerank.")
            return []

        if self._model is None:
            logger.warning(
                "[Reranker] Model not loaded. Call load() first. "
                "Falling back to RRF ordering."
            )
            return candidates[:top_k]

        logger.info(
            f"[Reranker] Scoring {len(candidates)} candidates for query: "
            f"'{query[:80]}...'"
        )
        t0 = time.perf_counter()

        # Build (query, passage) pairs — truncate passage to avoid token overflow
        pairs = [
            [query, c["text"][:1500]]
            for c in candidates
        ]

        scores: list[float] = self._model.predict(pairs).tolist()

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[Reranker] Scored {len(candidates)} candidates in {elapsed:.1f}ms "
            f"({elapsed / len(candidates):.1f}ms per candidate)"
        )

        # Attach scores and sort
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            candidate = dict(candidate)
            candidate["reranker_score"] = score
            candidates[i] = candidate

        reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)

        for rank, section in enumerate(reranked, start=1):
            section["reranker_rank"] = rank
            logger.debug(
                f"[Reranker] Rank {rank} | Score: {section['reranker_score']:.4f} | "
                f"Act: {section['meta'].get('act')} | "
                f"Section: {section['meta'].get('section_number')} | "
                f"Domain: {section['meta'].get('domain')}"
            )

        top = reranked[:top_k]
        logger.info(
            f"[Reranker] Final top-{top_k} sections selected. "
            f"Score range: [{top[-1]['reranker_score']:.4f}, {top[0]['reranker_score']:.4f}]"
        )
        return top

    @property
    def is_ready(self) -> bool:
        return self._model is not None


# ──────────────────────────────────────────────────────────────────
# Semantic Domain Classifier
# ──────────────────────────────────────────────────────────────────

class DomainClassifier:
    """
    Semantic domain classifier using prototype embeddings.

    At startup, computes a mean "prototype" embedding for each legal
    domain by averaging the Legal-BERT embeddings of all statute
    sections in that domain. At query time, computes cosine similarity
    between the query embedding and each prototype to determine which
    domain(s) the query belongs to.

    This acts as a semantic fallback when keyword-based detection fails
    to match any domain (e.g., user writes "I bought a broken product"
    instead of "sale of goods warranty").

    Thresholds are tunable via DOMAIN_CLASSIFIER_THRESHOLD and
    DOMAIN_CLASSIFIER_MARGIN in constants.py.
    """

    def __init__(self):
        self._prototypes: dict[str, np.ndarray] = {}

    def build(
        self,
        sections: list[dict],
        embeddings: np.ndarray,
    ) -> None:
        """
        Compute mean prototype embedding per domain.

        Args:
            sections:   List of section dicts (must have 'domain' key).
            embeddings: NumPy array of shape (N, dim) — one row per section,
                        in the same order as `sections`.
        """
        if len(sections) != len(embeddings):
            raise ValueError(
                f"[DomainClassifier] sections ({len(sections)}) and "
                f"embeddings ({len(embeddings)}) length mismatch."
            )

        logger.info(
            f"[DomainClassifier] Building prototypes from {len(sections)} sections..."
        )
        t0 = time.perf_counter()

        # Group embedding indices by domain
        domain_indices: dict[str, list[int]] = {}
        for i, s in enumerate(sections):
            domain = s.get("domain", "unknown")
            domain_indices.setdefault(domain, []).append(i)

        # Compute mean embedding per domain (the prototype)
        for domain, indices in domain_indices.items():
            domain_embs = embeddings[indices]           # shape: (n, dim)
            prototype = domain_embs.mean(axis=0)        # shape: (dim,)
            # Normalise to unit vector for efficient cosine similarity via dot product
            norm = np.linalg.norm(prototype)
            if norm > 0:
                prototype = prototype / norm
            self._prototypes[domain] = prototype
            logger.debug(
                f"[DomainClassifier] Prototype for '{domain}': "
                f"{len(indices)} sections | Norm before normalisation: {norm:.4f}"
            )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[DomainClassifier] Built {len(self._prototypes)} domain prototypes "
            f"in {elapsed:.1f}ms | Domains: {list(self._prototypes.keys())}"
        )

    def classify(
        self,
        query_embedding: np.ndarray,
        threshold: float = None,
        margin: float = None,
    ) -> list[str]:
        """
        Classify a query into one or more legal domains via cosine similarity.

        Computes cosine similarity between the (unit-normalised) query
        embedding and each domain prototype. Returns domain names whose
        similarity exceeds the threshold, with an optional margin to
        include a close second domain for ambiguous queries.

        Args:
            query_embedding: 1-D float array from Legal-BERT encoder.
            threshold:       Min cosine similarity to include a domain.
                             Defaults to DOMAIN_CLASSIFIER_THRESHOLD.
            margin:          Include 2nd domain if within this margin of top.
                             Defaults to DOMAIN_CLASSIFIER_MARGIN.

        Returns:
            List of matching domain strings, empty if none exceed threshold.
        """
        from src.utils.constants import (
            DOMAIN_CLASSIFIER_THRESHOLD,
            DOMAIN_CLASSIFIER_MARGIN,
        )
        if threshold is None:
            threshold = DOMAIN_CLASSIFIER_THRESHOLD
        if margin is None:
            margin = DOMAIN_CLASSIFIER_MARGIN

        if not self._prototypes:
            logger.warning("[DomainClassifier] No prototypes built. Call build() first.")
            return []

        # Normalise query embedding for cosine similarity via dot product
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            logger.warning("[DomainClassifier] Zero-norm query embedding — cannot classify.")
            return []
        q_unit = query_embedding / q_norm

        # Compute cosine similarity against each domain prototype
        similarities: dict[str, float] = {}
        for domain, prototype in self._prototypes.items():
            sim = float(np.dot(q_unit, prototype))  # dot product of unit vectors = cosine sim
            similarities[domain] = sim

        sorted_domains = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        logger.debug(
            "[DomainClassifier] Cosine similarities: "
            + " | ".join(f"{d}: {s:.4f}" for d, s in sorted_domains)
        )

        # Select domains above threshold, with second-place margin inclusion
        selected = []
        top_score = sorted_domains[0][1] if sorted_domains else 0.0

        for i, (domain, score) in enumerate(sorted_domains):
            if i == 0 and score >= threshold:
                selected.append(domain)
                logger.info(
                    f"[DomainClassifier] Top domain: '{domain}' | "
                    f"Similarity: {score:.4f} (threshold={threshold})"
                )
            elif i > 0 and selected and score >= (top_score - margin) and score >= threshold:
                # Include this domain too — close second (ambiguous query)
                selected.append(domain)
                logger.info(
                    f"[DomainClassifier] Secondary domain included: '{domain}' | "
                    f"Similarity: {score:.4f} (within margin={margin} of top)"
                )
            else:
                break  # Scores are sorted; no point checking further

        if not selected:
            logger.info(
                f"[DomainClassifier] No domain exceeded threshold {threshold}. "
                f"Top match was '{sorted_domains[0][0]}' at {sorted_domains[0][1]:.4f}. "
                "Retrieval will run globally."
            )

        return selected

    @property
    def is_ready(self) -> bool:
        return len(self._prototypes) > 0
