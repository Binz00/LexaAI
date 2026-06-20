#!/usr/bin/env python3
"""
LexaAI — Step 10: Output Renderer
=====================================
Renders the RAG pipeline output in 5 different modes:
1. Plain Rights Summary
2. Adverse Finding
3. Courtroom Preparation Script
4. Formal Complaint Letter
5. Statute Reference Card
"""


def render_plain_rights_summary(result: dict) -> str:
    """Plain-English explanation of rights and obligations."""
    sections = result.get("sections", [])
    answer = result.get("answer", "")

    output = "## 📋 Your Legal Rights Summary\n\n"
    output += f"{answer}\n\n"

    if sections:
        output += "### Relevant Law\n\n"
        for i, s in enumerate(sections[:5], 1):
            meta = s["meta"]
            output += f"**{i}. {meta['act']} — {meta['section_number']}**\n"
            if meta.get("header"):
                output += f"*{meta['header'][:150]}*\n"
            output += f"> {s['text'][:300]}...\n\n"

    return output


def render_adverse_finding(result: dict) -> str:
    """States the user has acted unlawfully + consequences."""
    sections = result.get("sections", [])
    answer = result.get("answer", "")

    output = "## ⚠️ Adverse Legal Finding\n\n"
    output += "**Based on the applicable law, the following finding applies to your situation:**\n\n"
    output += f"{answer}\n\n"

    if sections:
        output += "### Provisions You May Have Violated\n\n"
        for i, s in enumerate(sections[:5], 1):
            meta = s["meta"]
            output += f"**{i}. {meta['act']} — {meta['section_number']}**\n"
            output += f"> {s['text'][:300]}...\n\n"

    output += "### What This Means\n\n"
    output += ("You may be legally liable under the provisions cited above. "
               "The consequences may include penalties, fines, or legal action. "
               "You should seek qualified legal counsel immediately.\n")

    return output


def render_courtroom_preparation(result: dict) -> str:
    """Structured script with statute citations for court."""
    sections = result.get("sections", [])
    answer = result.get("answer", "")

    output = "## 🏛️ Courtroom Preparation Script\n\n"
    output += "**Use this script to present your case. Speak clearly and cite the law.**\n\n"
    output += "---\n\n"

    output += "### Opening Statement\n\n"
    output += f'*"Your Honour / Mr. Chairman, I appear before this court/tribunal '
    output += f'regarding the following matter..."*\n\n'
    output += f"{answer}\n\n"

    if sections:
        output += "### Key Legal Provisions to Cite\n\n"
        output += "*When presenting your case, reference these specific sections:*\n\n"
        for i, s in enumerate(sections[:5], 1):
            meta = s["meta"]
            output += f'**{i}.** *"I draw the court\'s attention to '
            output += f'**{meta["act"]}, {meta["section_number"]}**, which states:"*\n'
            output += f'> "{s["text"][:250]}..."\n\n'

    output += "### Closing Statement\n\n"
    output += ('*"Based on the provisions I have cited, I respectfully request '
               'that the court/tribunal consider my position and grant the '
               'appropriate relief."*\n')

    return output


def render_formal_complaint_letter(result: dict) -> str:
    """Professionally formatted legal complaint letter."""
    sections = result.get("sections", [])
    answer = result.get("answer", "")

    # Determine the relevant authority
    domains = set(s["meta"].get("domain", "") for s in sections)
    if "traffic_law" in domains:
        authority = "The Commissioner of Motor Traffic"
    elif "criminal_law" in domains:
        authority = "The Officer-in-Charge, [Police Station]"
    elif "business_law" in domains:
        authority = "The Director General, Consumer Affairs Authority"
    else:
        authority = "The Relevant Authority / [Recipient Name]"

    output = "## 📝 Formal Complaint Letter\n\n"
    output += "---\n\n"
    output += f"**To:** {authority}\n\n"
    output += "**From:** [Your Full Name]\n"
    output += "**Address:** [Your Address]\n"
    output += "**Date:** [Date]\n"
    output += "**NIC No:** [Your NIC Number]\n\n"
    output += "---\n\n"
    output += "**Subject:** Formal Complaint Regarding [Brief Description]\n\n"
    output += "Dear Sir/Madam,\n\n"
    output += ("I write to formally bring to your attention the following matter "
               "and to request appropriate action under the law.\n\n")
    output += f"**Facts of the Matter:**\n\n{answer}\n\n"

    if sections:
        output += "**Applicable Legal Provisions:**\n\n"
        for i, s in enumerate(sections[:3], 1):
            meta = s["meta"]
            output += (f"{i}. Under **{meta['act']}, {meta['section_number']}**, "
                       f"the law provides that: *\"{s['text'][:200]}...\"*\n\n")

    output += ("**Relief Sought:**\n\n"
               "I respectfully request that you investigate this matter and take "
               "appropriate action as prescribed by law.\n\n"
               "I am prepared to provide any further information or documentation "
               "that may be required.\n\n"
               "Yours faithfully,\n\n"
               "[Your Signature]\n"
               "[Your Full Name]\n")

    return output


def render_statute_reference_card(result: dict) -> str:
    """Compact list of relevant Act/section numbers."""
    sections = result.get("sections", [])

    output = "## 📌 Statute Reference Card\n\n"
    output += "*Print this and bring it to your legal consultation.*\n\n"
    output += "| # | Act | Section | Domain | Relevance |\n"
    output += "|---|-----|---------|--------|-----------|\n"

    for i, s in enumerate(sections[:8], 1):
        meta = s["meta"]
        domain = meta.get("domain", "").replace("_", " ").title()
        header = meta.get("header", "")[:60]
        output += f"| {i} | {meta['act']} | {meta['section_number']} | {domain} | {header} |\n"

    output += "\n"
    if any(s["meta"].get("level1") for s in sections):
        output += "### Structural Context\n\n"
        for s in sections[:5]:
            meta = s["meta"]
            if meta.get("level1"):
                output += f"- {meta['section_number']} falls under **{meta['level1']}**: {meta.get('level1_heading', '')}\n"

    return output


# ── Renderer Dispatcher ──────────────────────────────────────────

RENDERERS = {
    "plain_rights_summary": render_plain_rights_summary,
    "adverse_finding": render_adverse_finding,
    "courtroom_preparation": render_courtroom_preparation,
    "formal_complaint_letter": render_formal_complaint_letter,
    "statute_reference_card": render_statute_reference_card,
}


def render_output(result: dict, mode: str = "plain_rights_summary") -> str:
    """Render the pipeline result in the specified output mode."""
    renderer = RENDERERS.get(mode, render_plain_rights_summary)
    return renderer(result)
