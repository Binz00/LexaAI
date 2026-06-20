#!/usr/bin/env python3
"""
LexaAI — Evaluation: Retrieval Precision
==========================================
Tests Precision@5 across 60 queries (10 per domain).
Target: Precision@5 > 0.70

Usage:
    python evaluation/eval_retrieval.py
"""
import sys, time, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EVAL_RESULTS_DIR

# Test queries with expected domains
TEST_QUERIES = {
    "family_law": [
        {"q": "what age can a person legally marry in Sri Lanka", "domain": "family_law"},
        {"q": "how to register a marriage", "domain": "family_law"},
        {"q": "rights of a widow to inherit property", "domain": "family_law"},
        {"q": "can a minor get married with parental consent", "domain": "family_law"},
        {"q": "division of matrimonial property after divorce", "domain": "family_law"},
        {"q": "who can object to a marriage registration", "domain": "family_law"},
        {"q": "inheritance rights of children born out of wedlock", "domain": "family_law"},
        {"q": "procedure for marriage registration at registrar office", "domain": "family_law"},
        {"q": "matrimonial rights of a spouse", "domain": "family_law"},
        {"q": "can a marriage be declared void", "domain": "family_law"},
    ],
    "contract_law": [
        {"q": "when does a contract for land sale need to be in writing", "domain": "contract_law"},
        {"q": "can an oral promise to sell property be enforced", "domain": "contract_law"},
        {"q": "what documents need a notary attestation", "domain": "contract_law"},
        {"q": "is a verbal lease agreement valid", "domain": "contract_law"},
        {"q": "requirements for a valid deed of transfer", "domain": "contract_law"},
        {"q": "prevention of frauds in property transactions", "domain": "contract_law"},
        {"q": "signing a deed without understanding its contents", "domain": "contract_law"},
        {"q": "can I cancel a property sale agreement", "domain": "contract_law"},
        {"q": "notarial requirements for immovable property", "domain": "contract_law"},
        {"q": "fraud in a land transaction", "domain": "contract_law"},
    ],
    "business_law": [
        {"q": "warranty rights when buying defective goods", "domain": "business_law"},
        {"q": "can I return goods that are not fit for purpose", "domain": "business_law"},
        {"q": "seller's obligation to deliver goods on time", "domain": "business_law"},
        {"q": "when does ownership of goods pass to buyer", "domain": "business_law"},
        {"q": "rights of an unpaid seller", "domain": "business_law"},
        {"q": "implied conditions in a sale of goods", "domain": "business_law"},
        {"q": "buyer received damaged goods during delivery", "domain": "business_law"},
        {"q": "can a seller refuse to accept returned goods", "domain": "business_law"},
        {"q": "auction sale and bidding rules", "domain": "business_law"},
        {"q": "merchantable quality of goods", "domain": "business_law"},
    ],
    "traffic_law": [
        {"q": "penalty for driving without a licence", "domain": "traffic_law"},
        {"q": "how to register a motor vehicle", "domain": "traffic_law"},
        {"q": "what happens if my vehicle insurance expires", "domain": "traffic_law"},
        {"q": "speed limits on highways in Sri Lanka", "domain": "traffic_law"},
        {"q": "penalty for drunk driving", "domain": "traffic_law"},
        {"q": "requirements for vehicle fitness certificate", "domain": "traffic_law"},
        {"q": "can police impound my vehicle", "domain": "traffic_law"},
        {"q": "procedure after a road accident", "domain": "traffic_law"},
        {"q": "transferring vehicle ownership", "domain": "traffic_law"},
        {"q": "penalty for driving without a seatbelt", "domain": "traffic_law"},
    ],
    "criminal_law": [
        {"q": "penalty for theft of property", "domain": "criminal_law"},
        {"q": "what is the punishment for assault", "domain": "criminal_law"},
        {"q": "I forged a signature on a document", "domain": "criminal_law"},
        {"q": "what constitutes criminal breach of trust", "domain": "criminal_law"},
        {"q": "penalty for cheating and dishonestly inducing", "domain": "criminal_law"},
        {"q": "I was threatened with physical harm", "domain": "criminal_law"},
        {"q": "trespassing on private property at night", "domain": "criminal_law"},
        {"q": "defamation against a public servant", "domain": "criminal_law"},
        {"q": "kidnapping or abduction of a person", "domain": "criminal_law"},
        {"q": "bribery of a government official", "domain": "criminal_law"},
    ],
}


def evaluate_retrieval():
    from src.09_rag_pipeline import LexAIRAGPipeline
    pipeline = LexAIRAGPipeline()

    results = []
    domain_scores = {}
    total_latency = 0

    for domain, queries in TEST_QUERIES.items():
        domain_hits = 0
        domain_total = 0

        for q_data in queries:
            query = q_data["q"]
            expected_domain = q_data["domain"]

            start = time.time()
            sections = pipeline.retrieve(query, top_k=5)
            latency = (time.time() - start) * 1000
            total_latency += latency

            # Check precision: how many of top-5 are from expected domain
            hits = sum(1 for s in sections if s["meta"].get("domain") == expected_domain)
            precision = hits / min(5, len(sections)) if sections else 0

            domain_hits += hits
            domain_total += min(5, len(sections)) if sections else 5

            results.append({
                "query": query,
                "expected_domain": expected_domain,
                "precision_at_5": precision,
                "latency_ms": latency,
                "sections_retrieved": len(sections),
            })

        domain_precision = domain_hits / domain_total if domain_total > 0 else 0
        domain_scores[domain] = domain_precision

    # Summary
    overall_precision = sum(r["precision_at_5"] for r in results) / len(results)
    avg_latency = total_latency / len(results)

    print("\n" + "=" * 60)
    print("📊 RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Domain':<20} {'Precision@5':>12}")
    print("-" * 34)
    for domain, score in sorted(domain_scores.items()):
        status = "✅" if score >= 0.70 else "❌"
        print(f"{status} {domain:<18} {score:>11.2%}")
    print("-" * 34)
    print(f"   {'OVERALL':<18} {overall_precision:>11.2%}")
    print(f"\n  Avg latency: {avg_latency:.1f}ms (target: <200ms)")
    print(f"  {'✅' if overall_precision >= 0.70 else '❌'} Precision@5 target (>0.70): {overall_precision:.2%}")
    print(f"  {'✅' if avg_latency < 200 else '❌'} Latency target (<200ms): {avg_latency:.1f}ms")

    # Save results
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_DIR / "retrieval_results.json", "w") as f:
        json.dump({"overall_precision": overall_precision, "avg_latency_ms": avg_latency,
                    "domain_scores": domain_scores, "details": results}, f, indent=2)
    print(f"\n💾 Results saved to: {EVAL_RESULTS_DIR / 'retrieval_results.json'}")


if __name__ == "__main__":
    evaluate_retrieval()
