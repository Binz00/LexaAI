"""
LexaAI PDF Text Cleaning Functions
====================================
Format-specific cleaning functions for each PDF type.
Each function removes format-specific artifacts (headers, footers,
page markers, URLs) while preserving legal content and amendment references.
"""

import re


def clean_text(text: str) -> str:
    """
    Standard text cleaning applied to ALL PDFs after format-specific cleaning.
    Normalizes whitespace, removes excessive blank lines, and trims.
    """
    # Normalize Unicode characters
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    text = text.replace("\u00a0", " ")  # non-breaking space

    # Collapse multiple spaces to single space (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def clean_lawnet_pdf(text: str) -> str:
    """
    Clean PDFs extracted from lawnet.lk (Marriage Registration, Matrimonial Rights).

    These web extracts contain:
    - lawnet.lk URLs on every page
    - Page markers like "(N of 26) DD/MM/YYYY HH:MM:SS AM"
    - "Print | Close" headers
    """
    # Remove lawnet URLs and page markers
    text = re.sub(r"http://www\.lawnet\.lk/[^\n]+", "", text)
    text = re.sub(r"\(\d+ of \d+\)\s*\d+/\d+/\d+ \d+:\d+:\d+ [AP]M", "", text)

    # Remove "Print | Close" headers
    text = re.sub(r"Print\s*\|\s*Close", "", text)

    # Remove any remaining URL patterns
    text = re.sub(r"https?://[^\s]+", "", text)

    return clean_text(text)


def clean_column_pdf(text: str) -> str:
    """
    Clean dual-column PDFs (Sale of Goods Ordinance, Penal Code).

    These have:
    - Page headers like "Cap. 93]" or "[Cap. 19"
    - Running headers like "SALE OF GOODS" or "PENAL CODE"
    - Page numbers formatted as "IV/185"
    """
    # Remove page headers like "[Cap. 93]" or "Cap. 93]" or "[Cap. 19"
    text = re.sub(r"\[?Cap\.\s*\d+\]?", "", text)

    # Remove running headers
    text = re.sub(r"(?m)^SALE OF GOODS\s*$", "", text)
    text = re.sub(r"(?m)^PENAL CODE\s*$", "", text)

    # Remove page numbers like "IV/185" or "I/3"
    text = re.sub(r"(?m)^[IVX]+/\d+\s*$", "", text)

    # Remove standalone page numbers
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    return clean_text(text)


def clean_consolidated_pdf(text: str) -> str:
    """
    Clean 2024 consolidated edition PDFs (Motor Traffic, Prevention of Frauds).

    These are cleanly formatted but have:
    - Amendment references like [3, 8 of 2009] inline — KEEP these as metadata
    - Standalone page number lines
    """
    # Keep amendment references [3, 8 of 2009] intact — they are metadata
    # Only remove standalone page number lines
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    return clean_text(text)


# ── Cleaner Dispatcher ────────────────────────────────────────────

CLEANERS = {
    "web_extract":      clean_lawnet_pdf,
    "scanned_columns":  clean_column_pdf,
    "consolidated_pdf": clean_consolidated_pdf,
}


def get_cleaner(format_type: str):
    """
    Return the appropriate cleaning function for a given PDF format type.

    Args:
        format_type: One of 'web_extract', 'scanned_columns', 'consolidated_pdf'

    Returns:
        Cleaning function

    Raises:
        ValueError: If format_type is not recognized
    """
    if format_type not in CLEANERS:
        raise ValueError(
            f"Unknown format type '{format_type}'. "
            f"Expected one of: {list(CLEANERS.keys())}"
        )
    return CLEANERS[format_type]
