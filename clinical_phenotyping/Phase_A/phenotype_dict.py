"""
Phenotype token dictionary for Delphi v6.1.

Token IDs are stored-space IDs (model = stored + 1).
v5 vocab ends at stored 1535 (model 1536), so new tokens start at stored 1536.

29 tokens total → new vocab_size (stored) = 1536 + 29 = 1565.
"""

import re

# ── Token registry ────────────────────────────────────────────────────────────
# Ordered list: (name, stored_id, description)
PHENOTYPE_TOKENS = [
    # Cancer staging (14)
    ('CANCER_STAGE_I',           1536, 'Cancer staged as I (localised)'),
    ('CANCER_STAGE_II',          1537, 'Cancer staged as II'),
    ('CANCER_STAGE_III',         1538, 'Cancer staged as III'),
    ('CANCER_STAGE_IV',          1539, 'Cancer staged as IV (advanced/distant)'),
    ('CANCER_METASTATIC',        1540, 'Metastatic disease (without explicit stage IV)'),
    ('CANCER_RECURRENT',         1541, 'Recurrent / relapsed cancer'),
    ('CHEMO_RECEIVED',           1542, 'Chemotherapy received this admission'),
    ('CHEMO_PLANNED',            1543, 'Chemotherapy planned / recommended'),
    ('RADIOTHERAPY_RECEIVED',    1544, 'Radiation therapy received this admission'),
    ('RADIOTHERAPY_PLANNED',     1545, 'Radiation therapy planned / recommended'),
    ('IMMUNOTHERAPY_RECEIVED',   1546, 'Immunotherapy / checkpoint inhibitor received'),
    ('HORMONE_THERAPY_RECEIVED', 1547, 'Hormone / endocrine therapy received'),
    ('CANCER_RESECTED',          1548, 'Surgical resection of tumour this admission'),
    ('CANCER_STAGE_UNKNOWN',     1549, 'Cancer present but stage not specified'),

    # Mental health (8)
    ('SUICIDAL_IDEATION_PRESENT', 1550, 'Suicidal ideation present / active'),
    ('SUICIDAL_IDEATION_DENIED',  1551, 'Suicidal ideation explicitly denied'),
    ('SUICIDE_ATTEMPT_CURRENT',   1552, 'Suicide attempt precipitating this admission'),
    ('SUICIDE_ATTEMPT_HISTORY',   1553, 'History of prior suicide attempt(s)'),
    ('HOMICIDAL_IDEATION_PRESENT',1554, 'Homicidal ideation present'),
    ('PSYCHOSIS_ACTIVE',          1555, 'Active psychosis / psychotic episode this admission'),
    ('SELF_HARM_PRESENT',         1556, 'Non-suicidal self-injury present this admission'),
    ('PSYCHIATRIC_HOLD',          1557, 'Involuntary psychiatric hold / section placed'),

    # Substance use (5)
    ('ALCOHOL_WITHDRAWAL_ACTIVE', 1558, 'Active alcohol withdrawal; CIWA protocol'),
    ('OPIOID_WITHDRAWAL_ACTIVE',  1559, 'Active opioid withdrawal; COWS protocol'),
    ('SUBSTANCE_USE_ACTIVE',      1560, 'Active substance use / recent relapse this admission'),
    ('NALOXONE_ADMINISTERED',     1561, 'Naloxone / Narcan administered this admission'),

    # High-acuity clinical flags (8)
    ('SEPSIS_PRESENT',           1562, 'Sepsis / septic shock diagnosed this admission'),
    ('INTUBATED_DURING_STAY',    1563, 'Endotracheal intubation performed this admission'),
    ('COMFORT_MEASURES_ONLY',    1564, 'Goals of care: comfort / CMO / hospice'),
    ('DNR_PRESENT',              1565, 'DNR / DNI order in place this admission'),
    ('AKI_PRESENT',              1566, 'Acute kidney injury diagnosed this admission'),
    ('DELIRIUM_PRESENT',         1567, 'Delirium / acute confusion state this admission'),
    ('ICU_ADMISSION',            1568, 'Admitted to or transferred to ICU this admission'),
]

# Convenience lookups
TOKEN_NAME_TO_ID = {name: sid for name, sid, _ in PHENOTYPE_TOKENS}
TOKEN_ID_TO_NAME = {sid: name for name, sid, _ in PHENOTYPE_TOKENS}
N_PHENOTYPE_TOKENS = len(PHENOTYPE_TOKENS)
FIRST_STORED_ID   = PHENOTYPE_TOKENS[0][1]   # 1536
LAST_STORED_ID    = PHENOTYPE_TOKENS[-1][1]   # 1568
NEW_VOCAB_SIZE_STORED = LAST_STORED_ID + 1    # 1569  (stored 0..1568)

# ── Section header patterns ───────────────────────────────────────────────────
# Used to split notes into sections; higher-trust sections get stronger signal.
SECTION_RE = {
    'hpi':            re.compile(r'(?i)^\s*history\s+of\s+present\s+illness\s*:', re.M),
    'pmh':            re.compile(r'(?i)^\s*past\s+(medical\s+)?history\s*:', re.M),
    'hospital_course':re.compile(r'(?i)^\s*(brief\s+)?hospital\s+course\s*:', re.M),
    'assessment':     re.compile(r'(?i)^\s*assessment(\s+and\s+plan)?\s*:', re.M),
    'plan':           re.compile(r'(?i)^\s*plan\s*:', re.M),
    'discharge_dx':   re.compile(r'(?i)^\s*discharge\s+diagnos(is|es)\s*:', re.M),
    'followup':       re.compile(r'(?i)^\s*follow[- ]?up(\s+instructions?)?\s*:', re.M),
    'discharge_meds': re.compile(r'(?i)^\s*discharge\s+medications?\s*:', re.M),
    'discharge_instr':re.compile(r'(?i)^\s*discharge\s+instructions?\s*:', re.M),
}

# Sections where extracted findings represent THIS admission (high trust)
CURRENT_SECTIONS = {'hpi', 'hospital_course', 'assessment', 'plan', 'discharge_dx'}
# Sections to avoid for current-admission claims
AVOID_SECTIONS   = {'followup', 'discharge_meds', 'discharge_instr'}
# PMH is useful but marks historical facts


# ── Negation & temporal context patterns ─────────────────────────────────────
# Applied to the text window BEFORE a match.

NEG_PRE = re.compile(
    r'\b(no|not|without|denies?|denied|denying|'
    r'no\s+evidence\s+of|no\s+sign\s+of|'
    r'negative\s+for|rules?\s+out|ruled\s+out|'
    r'absent|absence\s+of|'
    r'no\s+current|no\s+active|no\s+acute|'
    r'no\s+history\s+of|no\s+prior|no\s+known)\b',
    re.I,
)

FUTURE_PRE = re.compile(
    r'\b(plan(?:ned|s|ning|s\s+to)?|'
    r'will\s+(?:be\s+)?(?:start|receive|undergo|need|require)|'
    r'scheduled\s+(?:to|for)|'
    r'recommend(?:ed|s|ing)?|'
    r'suggest(?:ed|s|ing)?|'
    r'consider(?:ed|ing|ation)?|'
    r'discuss(?:ed|ing)?|'
    r'to\s+be\s+(?:started|initiated|given|continued|held\s+pending)|'
    r'if\s+(?:patient|pt)\s+(?:agrees?|tolerates?|remains?)\b)\b',
    re.I,
)

def context_flags(text: str, start: int, pre_window: int = 80) -> dict:
    """
    Returns negation and future-plan flags for a regex match at `start`.
    Looks back `pre_window` characters.
    """
    pre = text[max(0, start - pre_window): start]
    return {
        'negated': bool(NEG_PRE.search(pre)),
        'planned': bool(FUTURE_PRE.search(pre)),
    }


# ── Cancer keywords (used to validate staging matches) ───────────────────────
CANCER_KW_RE = re.compile(
    r'\b(cancer|carcinoma|adenocarcinoma|squamous\s+cell|'
    r'sarcoma|lymphoma|leukemia|leukaemia|melanoma|'
    r'mesothelioma|myeloma|glioma|glioblastoma|'
    r'neoplasm|tumor|tumour|malignancy|malignant|'
    r'metastatic|metastases|mets\b)',
    re.I,
)

# ── Per-token extraction rules ────────────────────────────────────────────────
# Each rule is a compiled regex. Extraction logic in extract_phenotypes.py
# uses these patterns + context_flags().

PATTERNS = {}

# --- Cancer staging ---
# Requires cancer keyword within ±200 chars to avoid "Stage II hypertension"
PATTERNS['CANCER_STAGE_I'] = re.compile(
    r'\bstage\s+(?:i\b(?!i|v)|1\b)(?!\s*[b-z])',   # Stage I, Stage 1 (not Ib, etc.)
    re.I,
)
PATTERNS['CANCER_STAGE_II'] = re.compile(
    r'\bstage\s+(?:ii\b(?!i)|2\b)',
    re.I,
)
PATTERNS['CANCER_STAGE_III'] = re.compile(
    r'\bstage\s+(?:iii\b|3\b)',
    re.I,
)
PATTERNS['CANCER_STAGE_IV'] = re.compile(
    r'\bstage\s+(?:iv\b|4\b)',
    re.I,
)
PATTERNS['CANCER_METASTATIC'] = re.compile(
    r'\b(metastatic|metastases|distant\s+metastasis|'
    r'mets\s+to|metastatic\s+disease|'
    r'widely\s+metastatic)\b',
    re.I,
)
PATTERNS['CANCER_RECURRENT'] = re.compile(
    r'\b(recurrent\s+(?:cancer|tumor|tumour|disease|malignancy)|'
    r'cancer\s+recurrence|tumor\s+recurrence|'
    r'relapsed\s+(?:cancer|disease|lymphoma|leukemia))\b',
    re.I,
)
PATTERNS['CHEMO_RECEIVED'] = re.compile(
    r'\b((?:received|underw(?:ent|ent)|completed?|given|administered|started)\s+'
    r'(?:\w+\s+){0,3}chemotherapy|'
    r'chemotherapy\s+(?:was\s+)?(?:given|administered|started|completed?|initiated)|'
    r'(?:cycle[s]?\s+(?:\d+\s+)?of\s+)?chemotherapy\b(?!\s*plan|\s*recommend))\b',
    re.I,
)
PATTERNS['CHEMO_PLANNED'] = re.compile(
    r'\b(chemotherapy\s+(?:plan(?:ned)?|recommend(?:ed)?|scheduled|discussed)|'
    r'plan(?:ned|s)?\s+(?:for|to\s+(?:start|initiate|receive))\s+'
    r'(?:\w+\s+){0,3}chemotherapy|'
    r'will\s+(?:need|require|receive|undergo)\s+'
    r'(?:\w+\s+){0,3}chemotherapy)\b',
    re.I,
)
PATTERNS['RADIOTHERAPY_RECEIVED'] = re.compile(
    r'\b((?:received|completed?|underw(?:ent|ent)|given|administered|started)\s+'
    r'(?:\w+\s+){0,3}radiation\s+(?:therapy|treatment)|'
    r'radiation\s+(?:therapy|treatment)\s+(?:was\s+)?'
    r'(?:given|administered|started|completed?|initiated)|'
    r'\b(?:xrt|ebrt|sbrt|imrt)\b(?!\s*plan|\s*recommend))\b',
    re.I,
)
PATTERNS['RADIOTHERAPY_PLANNED'] = re.compile(
    r'\b(radiation\s+(?:therapy|treatment)\s+'
    r'(?:plan(?:ned)?|recommend(?:ed)?|scheduled|discussed)|'
    r'plan(?:ned|s)?\s+(?:for\s+)?radiation\s+(?:therapy|treatment)|'
    r'will\s+(?:need|require|receive|undergo)\s+'
    r'(?:\w+\s+){0,3}radiation\s+(?:therapy|treatment))\b',
    re.I,
)
PATTERNS['IMMUNOTHERAPY_RECEIVED'] = re.compile(
    r'\b((?:received|given|administered|started|initiated|underw(?:ent|ent))\s+'
    r'(?:\w+\s+){0,3}(?:immunotherapy|checkpoint\s+inhibitor|'
    r'nivolumab|pembrolizumab|atezolizumab|durvalumab|ipilimumab|'
    r'bevacizumab|trastuzumab|rituximab))\b',
    re.I,
)
PATTERNS['HORMONE_THERAPY_RECEIVED'] = re.compile(
    r'\b((?:on|received|started|taking)\s+'
    r'(?:\w+\s+){0,3}(?:hormone\s+therapy|hormonal\s+therapy|'
    r'tamoxifen|letrozole|anastrozole|exemestane|'
    r'leuprolide|bicalutamide|enzalutamide|abiraterone))\b',
    re.I,
)
PATTERNS['CANCER_RESECTED'] = re.compile(
    r'\b(resection\s+of\s+(?:the\s+)?(?:tumor|tumour|cancer|mass|lesion)|'
    r'tumor(?:al)?\s+resection|surgical\s+resection|'
    r'(?:underwent|status\s+post|s/p)\s+'
    r'(?:\w+\s+){0,3}(?:resection|colectomy|mastectomy|nephrectomy|'
    r'lobectomy|pneumonectomy|gastrectomy|cystectomy|prostatectomy))\b',
    re.I,
)
PATTERNS['CANCER_STAGE_UNKNOWN'] = re.compile(
    r'\b((?:cancer|carcinoma|adenocarcinoma|sarcoma|lymphoma|'
    r'leukemia|melanoma|tumor|tumour|malignancy)\s+'
    r'(?:of\s+(?:the\s+)?(?:breast|lung|colon|rectum|prostate|'
    r'ovary|pancreas|liver|kidney|bladder|stomach|cervix|'
    r'thyroid|skin|brain|bone|esophagus|endometrium)))\b',
    re.I,
)

# --- Mental health ---
PATTERNS['SUICIDAL_IDEATION_PRESENT'] = re.compile(
    r'\b(suicidal\s+ideation|'
    r'\bsi\b(?=[\s,]+(?:present|active|reported|noted|with|endorsed))|'
    r'thoughts\s+of\s+(?:suicide|killing\s+(?:himself|herself|themselves?|him/herself))|'
    r'want(?:s|ed)?\s+to\s+(?:die|kill\s+(?:himself|herself|themselves?))|'
    r'suicidal\s+thoughts|suicidal\s+plan|expressed\s+suicidal|'
    r'endorses?\s+suicidal)\b',
    re.I,
)
PATTERNS['SUICIDAL_IDEATION_DENIED'] = re.compile(
    r'\b(deni(?:es?|ed)\s+suicidal\s+ideation|'
    r'no\s+suicidal\s+ideation|'
    r'deni(?:es?|ed)\s+(?:any\s+)?thoughts\s+of\s+(?:suicide|self-harm)|'
    r'no\s+(?:active\s+)?suicidal\s+(?:ideation|thoughts|plan))\b',
    re.I,
)
PATTERNS['SUICIDE_ATTEMPT_CURRENT'] = re.compile(
    r'\b(suicide\s+attempt(?:ed)?|'
    r'attempted\s+suicide|'
    r'overdose\s+(?:with\s+intent|in\s+(?:a\s+)?suicide|attempt)|'
    r'self-inflicted\s+(?:injury|wound|overdose)|'
    r'intentional\s+(?:overdose|ingestion|self-harm))\b',
    re.I,
)
PATTERNS['SUICIDE_ATTEMPT_HISTORY'] = re.compile(
    r'\b(history\s+of\s+(?:suicide\s+attempt|'
    r'attempted\s+suicide|suicidal\s+(?:gesture|behavior))|'
    r'prior\s+suicide\s+attempt|'
    r'past\s+(?:suicide\s+attempt|suicidal\s+behavior))\b',
    re.I,
)
PATTERNS['HOMICIDAL_IDEATION_PRESENT'] = re.compile(
    r'\b(homicidal\s+ideation|'
    r'\bhi\b(?=[\s,]+(?:present|active|reported|noted|with))|'
    r'thoughts\s+of\s+(?:harming|hurting|killing)\s+(?:others?|someone|people)|'
    r'homicidal\s+thoughts|expressed\s+homicidal|endorses?\s+homicidal)\b',
    re.I,
)
PATTERNS['PSYCHOSIS_ACTIVE'] = re.compile(
    r'\b(active\s+psychosis|acute\s+psychosis|'
    r'psychotic\s+(?:episode|break|features|symptoms|decompensation|exacerbation)|'
    r'(?:experiencing|presenting\s+with)\s+'
    r'(?:\w+\s+){0,2}(?:hallucinations?|delusions?|paranoia)|'
    r'command\s+auditory\s+hallucinations?|'
    r'first[-\s]episode\s+psychosis|'
    r'psychotic\s+disorder\s+(?:not|with))\b',
    re.I,
)
PATTERNS['SELF_HARM_PRESENT'] = re.compile(
    r'\b(self[- ]harm(?:ing)?|'
    r'non[- ]suicidal\s+self[- ]injury|'
    r'cutting\s+(?:himself|herself|themselves?)|'
    r'self[- ](?:mutilation|inflicted\s+laceration|inflicted\s+wound))\b',
    re.I,
)
PATTERNS['PSYCHIATRIC_HOLD'] = re.compile(
    r'\b(psychiatric\s+hold|involuntary\s+(?:psychiatric\s+)?(?:hold|admission|commitment)|'
    r'section\s+12|mha\s+form|'
    r'placed\s+on\s+(?:a\s+)?(?:hold|section)|'
    r'committed\s+(?:involuntarily|to\s+(?:the\s+)?(?:unit|ward|floor)))\b',
    re.I,
)

# --- Substance use ---
PATTERNS['ALCOHOL_WITHDRAWAL_ACTIVE'] = re.compile(
    r'\b(alcohol\s+withdrawal|'
    r'etoh\s+withdrawal|'
    r'ciwa\s+(?:protocol|score|monitoring|scale)|'
    r'withdrawal\s+seizure(?:\s+from\s+alcohol)?|'
    r'delirium\s+tremens|dts\b|'
    r'alcohol(?:ic)?\s+hallucinosis)\b',
    re.I,
)
PATTERNS['OPIOID_WITHDRAWAL_ACTIVE'] = re.compile(
    r'\b(opioid\s+withdrawal|'
    r'opiate\s+withdrawal|'
    r'cows\s+(?:protocol|score|monitoring|scale)|'
    r'narcotic\s+withdrawal|heroin\s+withdrawal|'
    r'suboxone\s+(?:initiated|started|given|administered)\s+for\s+withdrawal)\b',
    re.I,
)
PATTERNS['SUBSTANCE_USE_ACTIVE'] = re.compile(
    r'\b(active\s+(?:substance|drug|alcohol)\s+use|'
    r'currently\s+(?:using|drinking|abusing)|'
    r'relapse(?:d)?\s+(?:to|on|with|of)\s+'
    r'(?:alcohol|drugs?|opioids?|cocaine|heroin|methamphetamine)|'
    r'(?:cocaine|heroin|methamphetamine|amphetamine|opioid)\s+intoxication|'
    r'positive\s+urine\s+(?:toxicology|tox|drug\s+screen))\b',
    re.I,
)
PATTERNS['NALOXONE_ADMINISTERED'] = re.compile(
    r'\b(naloxone\s+(?:was\s+)?(?:given|administered|used)|'
    r'narcan\s+(?:was\s+)?(?:given|administered|used)|'
    r'received\s+naloxone|received\s+narcan)\b',
    re.I,
)

# --- High-acuity flags ---
PATTERNS['SEPSIS_PRESENT'] = re.compile(
    r'\b(sepsis|septic\s+shock|'
    r'severe\s+sepsis|bacteremia\s+with\s+sepsis|'
    r'urosepsis|'
    r'(?:meets?|met|consistent\s+with)\s+(?:sirs|sepsis)\s+criteria)\b',
    re.I,
)
PATTERNS['INTUBATED_DURING_STAY'] = re.compile(
    r'\b((?:was\s+)?intubated|'
    r'endotracheal\s+intubation|'
    r'mechanical\s+ventilation|'
    r'(?:placed\s+on|requiring|required)\s+'
    r'(?:\w+\s+){0,2}(?:mechanical\s+ventilation|ventilator|intubation)|'
    r'orotracheal\s+intubation|'
    r'emergent\s+intubation)\b',
    re.I,
)
PATTERNS['COMFORT_MEASURES_ONLY'] = re.compile(
    r'\b(comfort\s+(?:measures?\s+only|care\s+only|focused\s+care)|'
    r'\bcmo\b|'
    r'transitioned\s+to\s+(?:comfort|hospice)|'
    r'goals\s+of\s+care\s+(?:changed|transitioned|converted)\s+to\s+comfort|'
    r'hospice\s+(?:care|transfer|referral|enrollment))\b',
    re.I,
)
PATTERNS['DNR_PRESENT'] = re.compile(
    r'\b(dnr(?:/dni)?|'
    r'do\s+not\s+resuscitate|'
    r'do\s+not\s+intubate|'
    r'full\s+(?:code\s+)?status\s+changed|'
    r'code\s+status\s*:\s*(?:dnr|comfort|dnr/dni))\b',
    re.I,
)
PATTERNS['AKI_PRESENT'] = re.compile(
    r'\b(acute\s+kidney\s+injury|'
    r'\baki\b(?!\s+prevention|\s+protocol)|'
    r'acute\s+renal\s+(?:failure|insufficiency)|'
    r'(?:creatinine|cr)\s+(?:elevated|increased|rose|raising)\s+'
    r'(?:from|above|to)\s+\d|'
    r'(?:new|onset)\s+(?:aki|acute\s+kidney))\b',
    re.I,
)
PATTERNS['DELIRIUM_PRESENT'] = re.compile(
    r'\b(delirium|delirious|'
    r'acute\s+(?:confusion(?:al\s+state)?|encephalopathy)|'
    r'altered\s+mental\s+status(?!\s+(?:at\s+)?(?:baseline|chronic))|'
    r'waxing\s+and\s+waning\s+(?:mental\s+status|consciousness)|'
    r'cam\s+positive|cam[-\s](?:icu)?\s+positive)\b',
    re.I,
)
PATTERNS['ICU_ADMISSION'] = re.compile(
    r'\b(admitted\s+to\s+(?:the\s+)?(?:icu|micu|sicu|ticu|nicu|picu|cvicu)|'
    r'transferred\s+to\s+(?:the\s+)?(?:icu|micu|sicu|ticu|nicu|picu|cvicu)|'
    r'(?:icu|micu|sicu)\s+(?:admission|stay|course|transfer)|'
    r'(?:required|requiring)\s+icu\s+(?:level\s+)?care|'
    r'intensive\s+care\s+unit\s+(?:admission|stay|transfer))\b',
    re.I,
)

assert len(PATTERNS) == len(PHENOTYPE_TOKENS), (
    f"Pattern count {len(PATTERNS)} != token count {len(PHENOTYPE_TOKENS)}"
)
