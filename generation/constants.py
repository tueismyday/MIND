"""
Constants and templates for the document generation pipeline.

This module centralizes all magic strings, prompts, markers, delimiters,
and configuration constants used throughout the generation process.
"""

from typing import Final

# =============================================================================
# SECTION MARKERS AND DELIMITERS
# =============================================================================

SECTION_MARKER: Final[str] = "Overskrift:"
SUBSECTION_MARKER: Final[str] = "SUBSECTION_TITLE:"
SOURCE_MARKER: Final[str] = "[Kilde:"
UNANSWERABLE_MARKER: Final[str] = "UNANSWERABLE"
VALID_MARKER: Final[str] = "VALID"

# =============================================================================
# PROMPT TEMPLATES - FACT ANSWERING
# =============================================================================

FACT_ANSWERING_PROMPT_TEMPLATE: Final[str] = """Du er en sygeplejerske der skal besvare ét specifikt faktum baseret på patientjournalen.

FAKTUM AT BESVARE:
{fact_description}

KILDER FRA PATIENTJOURNAL (fuld tekst):
{sources_text}

DIN OPGAVE:
1. Besvar faktumet fra sektionen '{subsection_title}' kort og præcist PÅ DANSK
2. Du skal ikke finde på mulige sammenhænge mellem information fra patientjournal og det faktum du skal besvare. Det er KUN hvis informationen svarer direkte på dit spørgsmål, at informationen er klinisk relevant, så VÆR KRITISK!
3. Inkludér KUN information der er klinisk relevant og vigtig. Hvis noget er normalt/ukompliceret, nævn det kort eller spring over. Fokusér på det der kræver opmærksomhed eller opfølgning.
4. Tilføj kildehenvisning: [Kilde: "Notetype" - DD.MM.YYYY]
5. Hvis informationen IKKE findes i kilderne, skriv præcis: "UNANSWERABLE"

EKSEMPEL PÅ GODT SVAR:
"Patienten har diabetes type 2 diagnosticeret i 2020 [Kilde: Lægenotat - 15.03.2024]."

EKSEMPEL PÅ UNANSWERABLE:
"UNANSWERABLE"

Skriv dit svar nu (kun svaret, ingen forklaring):"""

# =============================================================================
# PROMPT TEMPLATES - FACT ASSEMBLY
# =============================================================================

FACT_ASSEMBLY_PROMPT_TEMPLATE: Final[str] = """Du er en sygeplejerske der skal sammensætte underafsnittet '{subsection_title}' under sektionen '{section_title}'.

## GENERELLE INSTRUKTIONER FOR HELE SEKTIONEN:
{section_intro}
VIGTIG: Disse instruktioner gælder for ALLE underafsnit i denne sektion

## SPECIFIKKE INSTRUKTIONER FOR DETTE UNDERAFSNIT:
{format_instructions}

## BESVAREDE FAKTA (med kildehenvisninger):
{combined_facts}

DIN OPGAVE:
1. Skriv underafsnittet '{subsection_title}' som sammenhængende, naturlig tekst PÅ DANSK MED MAX 60 ORD, så vælg det vigtigste.
2. Følg BÅDE de generelle sektionsinstruktioner OG de specifikke underafsnit-instruktioner
3. Brug KUN relevant klinisk information og undgå gentagelser
4. Behold alle brugte kildehenvisninger [Kilde: Type - Dato] præcis som de er
5. Skriv IKKE underafsnit-titlen i din tekst
6. Vær kortfattet og præcis - brug IKKE fed skrift
7. Tilføj IKKE information der ikke er i de besvarede fakta
8. Lav "new-line" for hvert punktum du sætter

Skriv kun den sammensatte tekst (ingen forklaring før eller efter):
"""

# =============================================================================
# PROMPT TEMPLATES - FACT VALIDATION
# =============================================================================

FACT_VALIDATION_PROMPT_TEMPLATE: Final[str] = """Du er en kvalitetskontrollør der skal validere et svar om en patient.

FAKTUM DER BLEV BESVARET:
{fact_description}

SVAR DER SKAL VALIDERES:
{answer}

KILDER FRA PATIENTJOURNAL:
{sources_text}

DIN OPGAVE:
1. Tjek om svaret er korrekt baseret på kilderne
2. Tjek om kildehenvisningen [Kilde: Type - Dato] er præcis
3. Tjek om den nyeste information er brugt

Hvis svaret er KORREKT, skriv præcis: "VALID"

Hvis svaret har FEJL, ret det og returner det korrigerede svar med korrekte kildehenvisninger.

Skriv kun enten "VALID" eller det korrigerede svar:
"""

# =============================================================================
# PROMPT TEMPLATES - GUIDELINE PARSING
# =============================================================================

GUIDELINE_PARSING_PROMPT_TEMPLATE: Final[str] = """Du er en ekspert i at analysere medicinske retningslinjer.

SEKTION: {section_title}
UNDERAFSNIT: {subsection_title}

RETNINGSLINJER:
{cleaned_guidelines}

Din opgave er at identificere hvilke KONKRETE FAKTA der skal findes i patientjournalen,
samt udtrække SPECIFIKKE FORMAT-INSTRUKTIONER for dette underafsnit.

VIGTIGT - Håndter følgende:
1. Hvis retningslinjen siger "Hvis ja...", udtræk BEGGE muligheder (ja og nej scenarios)
2. Når der står "Brug X notat", betyder det der skal søge i den notetype
3. FORMAT KRAV skal indeholde instruktioner om HVORDAN der skal svares (f.eks. "besvar kort", "pas på ikke at konkludere", "giv bedste bud")
4. Lister med "f.eks." betyder ALLE eksemplerne er potentielle fakta
5. Når du laver "search_query", skal du optimere søgestrengen til RAG:
   - Brug præcise og beskrivende nøgleord
   - Undgå generiske ord (som “patienten”, “hvis”, “skal”)
   - Medtag synonymer og fagtermer fra konteksten
   - Sørg for at søgestrengen fungerer for både semantisk og nøgleordsbaseret søgning

## OUTPUT FORMAT - DU SKAL RETURNERE JSON ##

Returner KUN et JSON objekt med denne struktur (ingen tekst før eller efter):

{{
  "required_facts": [
    {{
      "description": "Præcis beskrivelse af faktum",
      "search_query": "optimeret søgestreng for RAG"
    }}
  ],
  "format_instructions": "Alle instruktioner om HVORDAN der skal svares for dette underafsnit"
}}

## EKSEMPEL ##

For retningslinje:
"Får patienten behov for medicindosering? Hvis ja: tabletter, øjendråber? Brug hjemmesygepleje notat. Svar meget kort og pas på ikke at konkludere."

Korrekt JSON:
{{
  "required_facts": [
    {{
      "description": "Behov for hjælp til medicindosering (ja/nej)",
      "search_query": "medicindosering hjælp behov"
    }},
    {{
      "description": "Type medicindosering: tabletter, øjendråber, salve, injektion",
      "search_query": "medicin tabletter øjendråber administration"
    }}
  ],
  "format_instructions": "Svar meget kort. Pas på ikke at konkludere. Giv bedste bud baseret på patientens tilstand."
}}

Vær SPECIFIK. Inkluder ALLE detaljer fra retningslinjen, også "hvis ja" scenarios.
Returner KUN valid JSON - ingen forklaring før eller efter!
"""

# =============================================================================
# FORMAT STRINGS
# =============================================================================

SOURCE_REFERENCE_FORMAT: Final[str] = "[Kilde: {entry_type} - {timestamp}]"
SUBSECTION_TITLE_FORMAT: Final[str] = "SUBSECTION_TITLE: {title}\n\n"
UNANSWERABLE_SECTION_FORMAT: Final[str] = "\n\nKunne ikke besvares ud fra patientjournalen:\n"

# =============================================================================
# APPENDIX HEADERS
# =============================================================================

REFERENCE_APPENDIX_HEADER: Final[str] = "KILDEHENVISNINGER OG DOKUMENTATION"
VALIDATION_APPENDIX_HEADER: Final[str] = "KVALITETSSIKRINGSRAPPORT - TO-TRINS VALIDERING"
APPENDIX_SEPARATOR: Final[str] = "=" * 50

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

# Default number of validation cycles
DEFAULT_MAX_VALIDATION_CYCLES: Final[int] = 2

# Minimum validation cycles (cannot go below this)
MIN_VALIDATION_CYCLES: Final[int] = 1

# Maximum validation cycles (cannot exceed this)
MAX_VALIDATION_CYCLES_LIMIT: Final[int] = 5

# Validation stages
VALIDATION_STAGE_FACT_CHECK: Final[str] = "fact_check"
VALIDATION_STAGE_GUIDELINE: Final[str] = "guideline_check"

# =============================================================================
# REGEX PATTERNS
# =============================================================================

NOTE_TYPES_PATTERN: Final[str] = r'\[NOTE_TYPES:\s*([^\]]+)\]'
JSON_EXTRACTION_PATTERN: Final[str] = r'\{[^}]+\}'

# =============================================================================
# LOGGING PREFIXES
# =============================================================================

LOG_INFO_PREFIX: Final[str] = "[INFO]"
LOG_ERROR_PREFIX: Final[str] = "[ERROR]"
LOG_WARNING_PREFIX: Final[str] = "[WARNING]"
LOG_DEBUG_PREFIX: Final[str] = "[DEBUG]"
LOG_SUCCESS_PREFIX: Final[str] = "[SUCCESS]"
LOG_RESULT_PREFIX: Final[str] = "[RESULT]"
LOG_CRITICAL_PREFIX: Final[str] = "[CRITICAL]"

# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_MAX_SOURCES_PER_FACT: Final[int] = 5
DEFAULT_MAX_REFERENCES_PER_SECTION: Final[int] = 3
DEFAULT_TEMPERATURE: Final[float] = 0.1
DEFAULT_MAX_TOKENS: Final[int] = 6056
DEFAULT_MAX_WORD_COUNT: Final[int] = 60
DEFAULT_SOURCE_SNIPPET_LENGTH: Final[int] = 80
DEFAULT_TOP_SOURCES_PER_TYPE: Final[int] = 10

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================

REPORTS_DIRECTORY: Final[str] = "reports"
APPENDIX_FILE_SUFFIX: Final[str] = "_appendices.txt"

# =============================================================================
# BATCH PROCESSING
# =============================================================================

BATCH_RETRY_MAX_ATTEMPTS: Final[int] = 2
BATCH_RETRY_BASE_DELAY: Final[int] = 2  # seconds
