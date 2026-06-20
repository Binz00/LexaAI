#!/usr/bin/env python3
"""
LexaAI — Step 01: Parse PDFs into Structured Sections
=======================================================
Reads all 6 Sri Lankan statute PDFs from data/raw_pdfs/,
applies format-specific cleaning, performs section chunking
(with hierarchical splitting for Motor Traffic and Penal Code),
and outputs a consolidated JSON file at data/sections/all_sections.json.

Usage:
    python src/01_parse_pdfs.py

Expected output:
    - data/sections/all_sections.json with 800+ sections
    - Console summary of section counts by domain
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF not installed. Run: pip install PyMuPDF")
    sys.exit(1)

from src.utils.constants import RAW_PDFS_DIR, SECTIONS_DIR, SECTIONS_JSON
from src.utils.legal_keywords import PDF_MAP, SKIP_FILES
from src.utils.cleaning import get_cleaner
from src.utils.chunking import (
    split_into_sections,
    hierarchical_split,
    sub_chunk_overflow,
    merge_sogo_marginals,
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text


def parse_single_pdf(pdf_path: Path, config: dict) -> list[dict]:
    """
    Parse a single PDF into structured sections.

    Args:
        pdf_path: Path to the PDF file
        config: Entry from PDF_MAP with keys: act, domain, format

    Returns:
        List of section dictionaries
    """
    act_name = config["act"]
    domain = config["domain"]
    format_type = config["format"]

    print(f"\n📄 Processing: {pdf_path.name}")
    print(f"   Act: {act_name}")
    print(f"   Domain: {domain}")
    print(f"   Format: {format_type}")

    # Step 1: Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"   Raw chars: {len(raw_text):,}")

    # Step 2: Apply format-specific cleaning
    cleaner = get_cleaner(format_type)
    cleaned_text = cleaner(raw_text)
    print(f"   Cleaned chars: {len(cleaned_text):,}")

    # Step 3: Split into sections
    # Use hierarchical splitting for large documents
    if "Motor Traffic" in act_name or "Penal" in act_name:
        sections = hierarchical_split(cleaned_text, act_name, domain)
    else:
        sections = split_into_sections(cleaned_text, act_name, domain)
        print(f"  ✅ {act_name}: {len(sections)} sections")

    # Step 4: Post-processing for specific formats
    if "Sale of Goods" in act_name:
        sections = merge_sogo_marginals(sections)

    # Step 5: Handle token overflow (sections > 512 tokens)
    sections = sub_chunk_overflow(sections)

    return sections


def parse_all_pdfs() -> list[dict]:
    """
    Parse all PDFs defined in PDF_MAP and return consolidated sections.

    Returns:
        List of all section dictionaries across all statutes
    """
    # Verify raw_pdfs directory exists
    if not RAW_PDFS_DIR.exists():
        print(f"❌ Directory not found: {RAW_PDFS_DIR}")
        print("   Please create the directory and place your PDF files in it.")
        print(f"   Expected files: {list(PDF_MAP.keys())}")
        sys.exit(1)

    # List available PDFs
    available_pdfs = [f.name for f in RAW_PDFS_DIR.glob("*.pdf")]
    print(f"📁 Found {len(available_pdfs)} PDF files in {RAW_PDFS_DIR}")

    # Check for skipped files
    for skip_file in SKIP_FILES:
        if skip_file in available_pdfs:
            print(f"   ⏭️  Skipping: {skip_file} (old version)")

    # Process each PDF in the map
    all_sections = []
    processed = 0
    missing = []

    for pdf_filename, config in PDF_MAP.items():
        pdf_path = RAW_PDFS_DIR / pdf_filename
        if not pdf_path.exists():
            print(f"\n⚠️  Missing: {pdf_filename}")
            missing.append(pdf_filename)
            continue

        sections = parse_single_pdf(pdf_path, config)
        all_sections.extend(sections)
        processed += 1

    return all_sections, processed, missing


def print_summary(sections: list[dict], processed: int, missing: list[str]):
    """Print a summary of the parsing results."""
    print("\n" + "=" * 60)
    print("📊 PARSING SUMMARY")
    print("=" * 60)

    # Count by domain
    domain_counts = {}
    domain_chars = {}
    for sec in sections:
        d = sec["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
        domain_chars[d] = domain_chars.get(d, 0) + sec["char_count"]

    print(f"\n{'Domain':<20} {'Sections':>10} {'Characters':>12}")
    print("-" * 44)
    for domain in sorted(domain_counts.keys()):
        print(f"{domain:<20} {domain_counts[domain]:>10,} {domain_chars[domain]:>12,}")
    print("-" * 44)
    print(f"{'TOTAL':<20} {len(sections):>10,} {sum(domain_chars.values()):>12,}")

    # Count by act
    print(f"\n{'Act':<45} {'Sections':>10}")
    print("-" * 57)
    act_counts = {}
    for sec in sections:
        act_counts[sec["act"]] = act_counts.get(sec["act"], 0) + 1
    for act in sorted(act_counts.keys()):
        print(f"{act:<45} {act_counts[act]:>10,}")

    # Count sub-chunks
    subchunks = sum(1 for s in sections if s.get("is_subchunk", False))
    if subchunks:
        print(f"\n📐 Sub-chunked sections: {subchunks}")

    # Level-1 stats
    with_level1 = sum(1 for s in sections if s.get("level1"))
    if with_level1:
        print(f"📂 Sections with Part/Chapter metadata: {with_level1}")

    # Missing files
    if missing:
        print(f"\n⚠️  Missing PDFs ({len(missing)}):")
        for m in missing:
            print(f"   - {m}")

    print(f"\n✅ Processed {processed} of {len(PDF_MAP)} PDFs")
    print(f"📄 Total sections: {len(sections)}")


def main():
    """Main entry point."""
    print("🔨 LexaAI PDF Parser v4.0")
    print("=" * 60)

    # Ensure output directory exists
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Parse all PDFs
    sections, processed, missing = parse_all_pdfs()

    if not sections:
        print("\n❌ No sections were parsed. Check that PDF files exist in data/raw_pdfs/")
        sys.exit(1)

    # Save to JSON
    with open(SECTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {SECTIONS_JSON}")

    # Print summary
    print_summary(sections, processed, missing)

    # Validation checks
    print("\n" + "=" * 60)
    print("✅ VALIDATION CHECKS")
    print("=" * 60)

    checks = [
        ("Total sections >= 800", len(sections) >= 800),
        ("All 6 statutes processed", processed == 6),
        ("No missing PDFs", len(missing) == 0),
        ("Family law sections exist", any(s["domain"] == "family_law" for s in sections)),
        ("Contract law sections exist", any(s["domain"] == "contract_law" for s in sections)),
        ("Business law sections exist", any(s["domain"] == "business_law" for s in sections)),
        ("Traffic law sections exist", any(s["domain"] == "traffic_law" for s in sections)),
        ("Criminal law sections exist", any(s["domain"] == "criminal_law" for s in sections)),
        ("Motor Traffic has level1 metadata", any(
            s.get("level1") and "Motor Traffic" in s["act"] for s in sections
        )),
        ("Penal Code has level1 metadata", any(
            s.get("level1") and "Penal" in s["act"] for s in sections
        )),
    ]

    all_pass = True
    for label, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {label}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 All validation checks passed!")
    else:
        print("\n⚠️  Some checks failed — review output above")


if __name__ == "__main__":
    main()
