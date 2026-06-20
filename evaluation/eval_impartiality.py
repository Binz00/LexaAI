#!/usr/bin/env python3
"""
LexaAI — Evaluation: Impartiality
====================================
Tests adverse finding rate across scenarios where the user is legally liable.
Target: Adverse finding rate > 80%

Usage:
    python evaluation/eval_impartiality.py
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EVAL_RESULTS_DIR

ADVERSE_SCENARIOS = [
    {"q": "I took property belonging to another person without their knowledge",
     "domain": "criminal_law", "expected": "liable"},
    {"q": "I drove my car without a valid driving licence",
     "domain": "traffic_law", "expected": "liable"},
    {"q": "I sold goods that I knew were defective without telling the buyer",
     "domain": "business_law", "expected": "liable"},
    {"q": "I signed a property deed under a false name",
     "domain": "contract_law", "expected": "liable"},
    {"q": "I dismissed my employee without giving any notice",
     "domain": "family_law", "expected": "liable"},
    {"q": "I forged my employer's signature on a cheque",
     "domain": "criminal_law", "expected": "liable"},
    {"q": "I was caught speeding at 120 km/h in a 60 km/h zone",
     "domain": "traffic_law", "expected": "liable"},
    {"q": "I threatened my neighbour with physical violence",
     "domain": "criminal_law", "expected": "liable"},
    {"q": "I refused to return the deposit to my tenant when they moved out",
     "domain": "family_law", "expected": "liable"},
    {"q": "I sold my car to someone and the engine broke down the next day",
     "domain": "business_law", "expected": "liable"},
    {"q": "I entered someone's property at night without permission",
     "domain": "criminal_law", "expected": "liable"},
    {"q": "I drove my vehicle with an expired insurance",
     "domain": "traffic_law", "expected": "liable"},
    {"q": "I kept money that was entrusted to me by my employer",
     "domain": "criminal_law", "expected": "liable"},
    {"q": "I made false statements about someone publicly",
     "domain": "criminal_law", "expected": "liable"},
    {"q": "I made a verbal agreement to sell land but now I want to back out",
     "domain": "contract_law", "expected": "liable"},
]

# Consistency test pairs (same situation, different framing)
CONSISTENCY_PAIRS = [
    {
        "sympathetic": "My employer fired me without any warning after 10 years of loyal service",
        "unsympathetic": "I was terminated from my position for repeated absences over 10 years",
        "domain": "family_law",
    },
    {
        "sympathetic": "A shop sold me a broken phone and refused to replace it",
        "unsympathetic": "I bought a phone but it stopped working and now I want my money back",
        "domain": "business_law",
    },
]


def evaluate_impartiality():
    from src.09_rag_pipeline import LexAIRAGPipeline
    pipeline = LexAIRAGPipeline()

    print("📊 IMPARTIALITY EVALUATION")
    print("=" * 60)

    # Adverse finding test
    adverse_hits = 0
    results = []

    for scenario in ADVERSE_SCENARIOS:
        result = pipeline.query(scenario["q"], "adverse_finding")
        answer = result.get("answer", "").lower()

        # Check if the answer identifies liability
        liable_keywords = ["liable", "offence", "penalty", "violation", "illegal",
                          "unlawful", "punishment", "fine", "guilty", "breach",
                          "wrong", "fault", "prohibited"]
        found_adverse = any(kw in answer for kw in liable_keywords)

        adverse_hits += int(found_adverse)
        results.append({
            "query": scenario["q"],
            "domain": scenario["domain"],
            "found_adverse": found_adverse,
            "answer_snippet": answer[:200],
        })

        status = "✅" if found_adverse else "❌"
        print(f"  {status} {scenario['q'][:60]}...")

    rate = adverse_hits / len(ADVERSE_SCENARIOS) * 100
    print(f"\n  Adverse finding rate: {rate:.1f}% (target: >80%)")
    print(f"  {'✅' if rate > 80 else '❌'} Target met")

    # Consistency test
    print(f"\n{'=' * 60}")
    print("📊 CONSISTENCY TEST")
    discrepancies = 0
    for pair in CONSISTENCY_PAIRS:
        r1 = pipeline.query(pair["sympathetic"])
        r2 = pipeline.query(pair["unsympathetic"])
        # Simple check: do they retrieve similar sections?
        s1_ids = set(s["meta"]["section_number"] for s in r1.get("sections", [])[:3])
        s2_ids = set(s["meta"]["section_number"] for s in r2.get("sections", [])[:3])
        overlap = len(s1_ids & s2_ids)
        consistent = overlap >= 1
        if not consistent:
            discrepancies += 1
        status = "✅" if consistent else "❌"
        print(f"  {status} Overlap: {overlap}/3 — {pair['domain']}")

    disc_rate = discrepancies / len(CONSISTENCY_PAIRS) * 100 if CONSISTENCY_PAIRS else 0
    print(f"\n  Discrepancy rate: {disc_rate:.1f}% (target: <10%)")

    # Save
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "adverse_finding_rate": rate,
        "adverse_hits": adverse_hits,
        "adverse_total": len(ADVERSE_SCENARIOS),
        "discrepancy_rate": disc_rate,
        "details": results,
    }
    with open(EVAL_RESULTS_DIR / "impartiality_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Saved: {EVAL_RESULTS_DIR / 'impartiality_results.json'}")


if __name__ == "__main__":
    evaluate_impartiality()
