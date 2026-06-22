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
import mlflow
from src.rag_pipeline import LexaAIRetriever

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
        {"q": "implied warranty of merchantable quality", "domain": "business_law"},
        {"q": "rights of a buyer when seller delivers wrong quantity", "domain": "business_law"},
        {"q": "contract for sale of goods", "domain": "business_law"},
    ],
    "criminal_law": [
        {"q": "what is the punishment for theft", "domain": "criminal_law"}, 
        {"q": "defamation and harm to reputation", "domain": "criminal_law"},
        {"q": "dishonestly receiving stolen property", "domain": "criminal_law"},
        {"q": "voluntarily causing hurt", "domain": "criminal_law"},
        {"q": "what is considered criminal force", "domain": "criminal_law"},
        {"q": "punishment for assault", "domain": "criminal_law"},
        {"q": "what is the difference between theft and extortion", "domain": "criminal_law"},
        {"q": "cheating and dishonestly inducing delivery of property", "domain": "criminal_law"},
        {"q": "forgery for the purpose of cheating", "domain": "criminal_law"},
        {"q": "dishonest misappropriation of property", "domain": "criminal_law"},
    ],
    "traffic_law": [
        {"q": "driving without a valid license", "domain": "traffic_law"},
        {"q": "what is the legal alcohol limit for driving", "domain": "traffic_law"},
        {"q": "using a mobile phone while driving", "domain": "traffic_law"},
        {"q": "who has right of way at an intersection", "domain": "traffic_law"},
        {"q": "road signs and markings", "domain": "traffic_law"},
        {"q": "what to do after a minor traffic accident", "domain": "traffic_law"},
        {"q": "parking in a no parking zone", "domain": "traffic_law"},
        {"q": "can police search your vehicle without a warrant", "domain": "traffic_law"},
        {"q": "failing to stop at a red light", "domain": "traffic_law"},
        {"q": "driving an unregistered vehicle", "domain": "traffic_law"},
    ],
}

@mlflow.autolog()
def evaluate_retrieval():
    retriever = LexaAIRetriever()
    
    total_queries = 0
    correct_at_5 = 0

    for domain, queries in TEST_QUERIES.items():
        for query in queries:
            total_queries += 1
            results = retriever.retrieve(query["q"], top_k=5)
            
            if any(r['domain'] == query['domain'] for r in results):
                correct_at_5 += 1

    precision_at_5 = correct_at_5 / total_queries
    mlflow.log_metric("Precision@5", precision_at_5)

    print(f"Precision@5: {precision_at_5:.3f}")

if __name__ == "__main__":
    evaluate_retrieval()
