"""
Core extraction logic for clinical phenotype tokens.

For each discharge note, returns the set of PHENOTYPE_TOKENS present.

Design principles:
  - Section-aware: split note into sections; avoid follow-up / discharge-meds
    sections which describe future plans, not current-admission facts.
  - Negation-aware: 80-char lookback window for negation terms.
  - Cancer-staging guard: staging tokens only fire when a cancer keyword
    appears within 200 chars of the staging match.
  - Temporal guard: RECEIVED variants require absence of future-plan terms
    in lookback window; PLANNED variants require their presence.
  - PMH section: only HISTORY tokens fire (SUICIDE_ATTEMPT_HISTORY).

Usage:
    from extract_phenotypes import extract_phenotype_tokens
    tokens = extract_phenotype_tokens(text)   # returns set of stored IDs
"""

import re
import sys
from pathlib import Path
from typing import Set

sys.path.insert(0, str(Path(__file__).parent))
from phenotype_dict import (
    PATTERNS, TOKEN_NAME_TO_ID, SECTION_RE,
    CANCER_KW_RE, context_flags, AVOID_SECTIONS,
)

# Staging tokens that need a nearby cancer keyword to fire
STAGING_TOKENS = {
    'CANCER_STAGE_I', 'CANCER_STAGE_II', 'CANCER_STAGE_III', 'CANCER_STAGE_IV',
    'CANCER_STAGE_UNKNOWN',
}
CANCER_CONTEXT_WINDOW = 200   # chars around staging match to look for cancer KW

# Tokens that should NOT fire from the PMH section
SKIP_IN_PMH = {
    'CANCER_STAGE_I', 'CANCER_STAGE_II', 'CANCER_STAGE_III', 'CANCER_STAGE_IV',
    'CANCER_METASTATIC', 'CANCER_RECURRENT',
    'CHEMO_RECEIVED', 'RADIOTHERAPY_RECEIVED',
    'IMMUNOTHERAPY_RECEIVED', 'HORMONE_THERAPY_RECEIVED',
    'CANCER_RESECTED',
    'SUICIDAL_IDEATION_PRESENT', 'SUICIDE_ATTEMPT_CURRENT',
    'HOMICIDAL_IDEATION_PRESENT', 'PSYCHOSIS_ACTIVE', 'SELF_HARM_PRESENT',
    'PSYCHIATRIC_HOLD',
    'ALCOHOL_WITHDRAWAL_ACTIVE', 'OPIOID_WITHDRAWAL_ACTIVE',
    'SUBSTANCE_USE_ACTIVE', 'NALOXONE_ADMINISTERED',
    'SEPSIS_PRESENT', 'INTUBATED_DURING_STAY', 'COMFORT_MEASURES_ONLY',
    'DNR_PRESENT', 'AKI_PRESENT', 'DELIRIUM_PRESENT', 'ICU_ADMISSION',
}

# RECEIVED tokens must NOT have a future-plan lookback
RECEIVED_TOKENS = {
    'CHEMO_RECEIVED', 'RADIOTHERAPY_RECEIVED', 'IMMUNOTHERAPY_RECEIVED',
    'HORMONE_THERAPY_RECEIVED', 'CANCER_RESECTED',
    'NALOXONE_ADMINISTERED',
}
# PLANNED tokens must HAVE a future-plan lookback
PLANNED_TOKENS = {
    'CHEMO_PLANNED', 'RADIOTHERAPY_PLANNED',
}


def _split_sections(text: str) -> dict:
    """
    Split a discharge note into named sections.
    Returns {section_name: (start_char, end_char)} dict,
    plus a special key '__full__' spanning the whole text.
    """
    hits = []
    for name, pat in SECTION_RE.items():
        for m in pat.finditer(text):
            hits.append((m.start(), m.end(), name))
    hits.sort()

    sections = {'__full__': (0, len(text))}
    for i, (start, end, name) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        sections[name] = (start, next_start)

    return sections


def extract_phenotype_tokens(text: str) -> Set[int]:
    """
    Extract phenotype token IDs from a single discharge note text.
    Returns a set of stored token IDs.
    """
    found: Set[int] = set()
    if not text or len(text) < 50:
        return found

    text_lower = text   # keep original case for matching (patterns are IGNORECASE)
    sections   = _split_sections(text)

    for token_name, pattern in PATTERNS.items():
        stored_id = TOKEN_NAME_TO_ID[token_name]

        for m in pattern.finditer(text_lower):
            mstart, mend = m.start(), m.end()

            # ── Section check ─────────────────────────────────────────────
            in_pmh       = False
            in_avoid     = False

            for sname, (sstart, send) in sections.items():
                if sname == '__full__':
                    continue
                if sstart <= mstart < send:
                    if sname == 'pmh':
                        in_pmh = True
                    if sname in AVOID_SECTIONS:
                        in_avoid = True
                    break

            if in_avoid:
                continue
            if in_pmh and token_name in SKIP_IN_PMH:
                continue

            # ── Negation / temporal context ───────────────────────────────
            ctx = context_flags(text_lower, mstart)

            # SUICIDAL_IDEATION_DENIED already encodes its own negation
            if token_name == 'SUICIDAL_IDEATION_PRESENT' and ctx['negated']:
                # Might be a denial — let SUICIDAL_IDEATION_DENIED catch it
                continue
            if token_name == 'SUICIDAL_IDEATION_DENIED':
                # This pattern itself contains negation; always fire
                pass
            elif ctx['negated']:
                continue

            if token_name in RECEIVED_TOKENS and ctx['planned']:
                continue   # future plan, not current receipt
            if token_name in PLANNED_TOKENS and not ctx['planned']:
                # Some PLANNED patterns already encode the future context
                # inside the regex; don't double-filter here
                pass

            # ── Cancer staging context guard ──────────────────────────────
            if token_name in STAGING_TOKENS:
                win_start = max(0, mstart - CANCER_CONTEXT_WINDOW)
                win_end   = min(len(text_lower), mend + CANCER_CONTEXT_WINDOW)
                window    = text_lower[win_start:win_end]
                if not CANCER_KW_RE.search(window):
                    continue

            found.add(stored_id)
            break   # one match per token per note is enough

    return found
