"""
LexaAI Section Chunking
========================
Splits cleaned PDF text into individual statutory sections with metadata.
Includes hierarchical (two-level) splitting for large documents
(Motor Traffic Act: 14 Parts, Penal Code: 22 Chapters) and overflow handling
for sections exceeding Legal-BERT's 512-token limit.
"""

import re
from typing import Optional


# ── Standard Section Splitting ────────────────────────────────────

def split_into_sections(
    text: str,
    act_name: str,
    domain: str,
    level1: Optional[str] = None,
    level1_heading: Optional[str] = None,
) -> list[dict]:
    """
    Split statute text into individual numbered sections.

    Matches patterns like:
      "Section 3." or "3." or "3.-" or "3. (1)" at the start of a line.

    Args:
        text: Cleaned statute text
        act_name: Full name of the statute
        domain: Legal domain (e.g., 'family_law', 'criminal_law')
        level1: Optional Part/Chapter identifier (e.g., 'Part VII')
        level1_heading: Optional Part/Chapter heading text

    Returns:
        List of section dictionaries with metadata
    """
    # Pattern to match section starts:
    # "Section 3." or "3." or "3.-" at the start of a line
    # Also handles "Section 3A." and sub-numbering like "3. (1)"
    section_pattern = re.compile(
        r"(?m)^(?:Section\s+)?(\d+[A-Z]?)\s*[\.\-\(]"
    )

    matches = list(section_pattern.finditer(text))

    if not matches:
        # No numbered sections found — treat entire text as one section
        return [{
            "id": f"{domain}_full",
            "act": act_name,
            "domain": domain,
            "section_number": "Full Text",
            "header": "",
            "text": text.strip(),
            "char_count": len(text.strip()),
            "level1": level1 or "",
            "level1_heading": level1_heading or "",
        }]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        section_num = match.group(1)

        # Extract header: first line of the section (often contains the section title)
        first_line = section_text.split("\n")[0].strip()
        header = first_line[:200] if len(first_line) > 200 else first_line

        # Create a short act abbreviation for unique IDs
        act_abbr = "".join(
            w[0].lower() for w in act_name.split() if w[0].isalpha()
        )[:6]

        section_id_parts = [domain, act_abbr]
        if level1:
            # Sanitize level1 for ID (e.g., "Part VII" → "Part_VII")
            level1_clean = level1.replace(" ", "_")
            section_id_parts.append(level1_clean)
        section_id_parts.append(f"s{section_num}")
        section_id = "_".join(section_id_parts)

        sections.append({
            "id": section_id,
            "act": act_name,
            "domain": domain,
            "section_number": f"Section {section_num}",
            "header": header,
            "text": section_text,
            "char_count": len(section_text),
            "level1": level1 or "",
            "level1_heading": level1_heading or "",
        })

    return sections


# ── Hierarchical Splitting (Motor Traffic, Penal Code) ────────────

def hierarchical_split(text: str, act_name: str, domain: str) -> list[dict]:
    """
    Two-level split for large statutes:
      Level 1: Split by PART (Motor Traffic) or CHAPTER (Penal Code)
      Level 2: Split by numbered section within each Part/Chapter

    This preserves structural context — e.g., knowing that Section 300
    falls under 'Chapter XVI — Of Offences Affecting the Human Body'.

    Args:
        text: Full cleaned statute text
        act_name: Full name of the statute
        domain: Legal domain

    Returns:
        List of section dictionaries with level1 metadata
    """
    # Determine level-1 structural unit
    if "Motor Traffic" in act_name:
        level1_pattern = re.compile(r"(?m)^[\s]*PART\s+([IVXivx]+)\b")
        level1_label = "Part"
    elif "Penal" in act_name:
        level1_pattern = re.compile(r"(?m)^[\s]*CHAPTER\s+([IVXivx]+)\b")
        level1_label = "Chapter"
    else:
        # Not a large document — use standard split
        return split_into_sections(text, act_name, domain)

    # Find all level-1 divisions
    matches = list(level1_pattern.finditer(text))

    if not matches:
        # No level-1 divisions found — fall back to standard split
        print(f"  ⚠️  No {level1_label}s found in {act_name}, using standard split")
        return split_into_sections(text, act_name, domain)

    # Extract positions for each level-1 division
    positions = [m.start() for m in matches]
    positions.append(len(text))

    all_sections = []
    for i, start in enumerate(positions[:-1]):
        end = positions[i + 1]
        chunk = text[start:end]

        # Extract level-1 heading (first non-empty line of the chunk)
        heading_lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        level1_heading = heading_lines[0] if heading_lines else f"{level1_label} {i+1}"

        # Extract level-1 number
        level1_match = level1_pattern.match(chunk.lstrip())
        level1_num = level1_match.group(1).upper() if level1_match else f"{i+1}"
        level1_id = f"{level1_label} {level1_num}"

        # Level 2: split by numbered section within this Part/Chapter
        sections = split_into_sections(
            chunk, act_name, domain,
            level1=level1_id,
            level1_heading=level1_heading,
        )

        # Update IDs to include level-1 info
        for sec in sections:
            sec["id"] = f"{domain}_{level1_label}_{level1_num}_{sec['section_number'].replace('Section ', 's')}"

        all_sections.extend(sections)

    print(f"  ✅ {act_name}: {len(matches)} {level1_label}s → {len(all_sections)} sections")
    return all_sections


# ── Token Overflow Handling ───────────────────────────────────────

def sub_chunk_overflow(
    sections: list[dict],
    max_tokens: int = 512,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
    chars_per_token: float = 5.2,
) -> list[dict]:
    """
    Split sections that exceed the token limit into overlapping sub-chunks.

    Uses character-based estimation (avg ~5.2 chars/token for legal text).
    Sections within the limit are passed through unchanged.

    Args:
        sections: List of section dictionaries
        max_tokens: Maximum token count (Legal-BERT limit)
        chunk_tokens: Target chunk size in tokens
        overlap_tokens: Overlap between consecutive chunks
        chars_per_token: Estimated characters per token

    Returns:
        Updated list with long sections replaced by overlapping sub-chunks
    """
    max_chars = int(max_tokens * chars_per_token)
    chunk_chars = int(chunk_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)
    step = chunk_chars - overlap_chars

    result = []
    overflow_count = 0

    for sec in sections:
        if sec["char_count"] <= max_chars:
            result.append(sec)
        else:
            # Section exceeds token limit — split into overlapping sub-chunks
            overflow_count += 1
            text = sec["text"]
            chunk_idx = 0
            pos = 0

            while pos < len(text):
                end = min(pos + chunk_chars, len(text))
                chunk_text = text[pos:end].strip()

                if chunk_text:
                    sub = sec.copy()
                    sub["id"] = f"{sec['id']}_chunk_{chunk_idx}"
                    sub["text"] = chunk_text
                    sub["char_count"] = len(chunk_text)
                    sub["header"] = f"{sec['header']} [chunk {chunk_idx}]"
                    sub["is_subchunk"] = True
                    sub["parent_id"] = sec["id"]
                    result.append(sub)
                    chunk_idx += 1

                pos += step

    if overflow_count > 0:
        print(f"  📐 Sub-chunked {overflow_count} sections exceeding {max_tokens} tokens")

    return result


# ── SOGO Marginal Title Merger ────────────────────────────────────

def merge_sogo_marginals(sections: list[dict]) -> list[dict]:
    """
    Post-process Sale of Goods Ordinance sections to merge marginal titles.

    SOGO uses a dual-column layout with marginal section titles positioned
    to the left of the main text. PyMuPDF may extract these as very short
    separate sections. This function merges them with the following section.

    Args:
        sections: Parsed sections from SOGO

    Returns:
        Merged sections with marginal titles incorporated into headers
    """
    merged = []
    skip_next = False

    for i, sec in enumerate(sections):
        if skip_next:
            skip_next = False
            continue

        # If this section is very short and the next section exists,
        # it's likely a marginal title — merge with next section
        if sec["char_count"] < 80 and i + 1 < len(sections):
            next_sec = sections[i + 1].copy()
            marginal_title = sec["text"].strip()
            if marginal_title:
                next_sec["header"] = f"{marginal_title} — {next_sec['header']}"
            merged.append(next_sec)
            skip_next = True
        else:
            merged.append(sec)

    if len(merged) < len(sections):
        print(f"  🔗 Merged {len(sections) - len(merged)} SOGO marginal titles")

    return merged
