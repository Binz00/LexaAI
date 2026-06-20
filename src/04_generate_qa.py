#!/usr/bin/env python3
"""
LexaAI — Step 04: Local Synthetic Q&A Generation
================================================
Generates synthetic question-answer pairs from all 6 statutes locally.
Uses templates and keyword extraction to create ~2,500 training samples
to teach the model domain boundaries (e.g., distinguishing Marriage from Traffic).

Usage:
    python src/04_generate_qa.py
"""

import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from src.utils.constants import SECTIONS_JSON, SYNTHETIC_QA_DIR, SYNTHETIC_QA_JSON

# ── Templates ─────────────────────────────────────────────────────

TEMPLATES = {
    "family_law": [
        "What does the law say about {topic} in Sri Lanka?",
        "I need information regarding {topic} under the {act}.",
        "What are the requirements for {topic} according to {section}?",
        "Can you explain the rules for {topic}?",
        "I have a question about {topic}. What is the legal position?",
    ],
    "traffic_law": [
        "What is the penalty for {topic} under the Motor Traffic Act?",
        "I was caught for {topic}. What does Section {num} say?",
        "What are the regulations for {topic} in Sri Lanka?",
        "What are the legal consequences of {topic}?",
        "Tell me about the rules regarding {topic} for drivers.",
    ],
    "criminal_law": [
        "What is the punishment for {topic} according to the Penal Code?",
        "How does the law define {topic}?",
        "Is {topic} a criminal offence in Sri Lanka?",
        "What happens if someone commits {topic}?",
        "What does Section {num} of the Penal Code say about {topic}?",
    ],
    "business_law": [
        "What are my rights regarding {topic} when buying goods?",
        "What does the Sale of Goods Ordinance say about {topic}?",
        "What are the implied conditions for {topic}?",
        "Who is liable for {topic} in a sale contract?",
        "Can you explain the rules for {topic} in business?",
    ],
    "contract_law": [
        "What is the law regarding {topic} in property transactions?",
        "Does {topic} require a written agreement under the Prevention of Frauds Ordinance?",
        "How is {topic} handled in land contracts?",
        "What are the legal formalities for {topic}?",
        "Is {topic} legally binding without a notary?",
    ]
}

def clean_topic(text):
    """Clean a section header to use as a topic."""
    # Remove leading numbers/dots (e.g., "3. Interpretation" -> "Interpretation")
    topic = re.sub(r'^\d+\.?\s*', '', text)
    # Remove citations in parens
    topic = re.sub(r'\[.*?\]', '', topic)
    return topic.strip()

def generate_local_qa(sections):
    """Generate Q&A pairs using local heuristics."""
    qa_pairs = []
    
    for sec in tqdm(sections, desc="Generating Local Q&A"):
        domain = sec["domain"]
        act = sec["act"]
        num = sec["section_number"].replace("Section ", "")
        header = sec.get("header", "this provision")
        text = sec["text"]
        
        # 1. Clean the topic from the header
        topic = clean_topic(header)
        if len(topic) < 3 or topic.lower() == "repealed":
            continue
            
        # 2. Pick 3-4 templates for this domain
        domain_templates = TEMPLATES.get(domain, TEMPLATES["family_law"])
        selected = random.sample(domain_templates, min(3, len(domain_templates)))
        
        for temp in selected:
            question = temp.format(topic=topic, act=act, section=sec["section_number"], num=num)
            
            # Simple answer logic: Use first two sentences + citation
            sentences = text.split('.')
            answer_text = ". ".join(sentences[:2]).strip()
            if not answer_text.endswith('.'):
                answer_text += '.'
                
            answer = f"According to {sec['section_number']} of the {act}, {answer_text}"
            
            # Determine scenario
            scenario = "user-liable" if any(k in text.lower() for k in ["penalty", "punish", "offence", "fine"]) else "user-wronged"
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "answer_span": sentences[0].strip(),
                "scenario": scenario,
                "section_number": sec["section_number"],
                "act": act,
                "domain": domain,
                "section_id": sec["id"]
            })
            
    return qa_pairs

def main():
    print("🔨 LexaAI Local Synthetic Q&A Generator")
    print("=" * 60)
    
    if not SECTIONS_JSON.exists():
        print("❌ Run src/01_parse_pdfs.py first")
        sys.exit(1)
        
    with open(SECTIONS_JSON, "r") as f:
        sections = json.load(f)
        
    print(f"📄 Processing {len(sections)} sections...")
    
    qa_pairs = generate_local_qa(sections)
    
    SYNTHETIC_QA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SYNTHETIC_QA_JSON, "w") as f:
        json.dump(qa_pairs, f, indent=2)
        
    print(f"\n✅ Generated {len(qa_pairs)} Q&A pairs locally.")
    print(f"💾 Saved to: {SYNTHETIC_QA_JSON}")

if __name__ == "__main__":
    main()
