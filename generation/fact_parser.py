"""
Guideline parsing and fact requirement extraction.

This module parses medical guideline text to identify what specific facts
need to be retrieved from patient records. It uses an LLM to convert
guideline instructions into structured RequiredFact objects with optimized
search queries for RAG retrieval.

The parser also extracts format instructions and note type filters from
guidelines to guide the generation process.

Example:
    >>> parser = GuidelineFactParser()
    >>> requirements = parser.parse_subsection_requirements(
    ...     section_title="Treatment Plan",
    ...     subsection_title="Current Medications",
    ...     subsection_guidelines="List all current medications..."
    ... )
    >>> print(f"Found {len(requirements.required_facts)} facts to retrieve")
"""

import logging
import json
import re
from typing import List, Dict, Optional

from generation.models import RequiredFact, SubsectionRequirements, FactPriority
from generation.constants import (
    GUIDELINE_PARSING_PROMPT_TEMPLATE,
    NOTE_TYPES_PATTERN
)
from generation.exceptions import FactParsingError
from config.llm_config import llm_config
from utils.error_handling import safe_llm_invoke

# Configure module logger
logger = logging.getLogger(__name__)


class GuidelineFactParser:
    """Parses medical guidelines to extract factual requirements.

    The parser analyzes guideline text and identifies:
    1. What specific facts need to be retrieved from patient records
    2. Optimized search queries for RAG retrieval
    3. Note type filters to improve retrieval accuracy
    4. Format instructions for how to present the information

    The parser uses JSON-based LLM output for structured, reliable parsing.

    Attributes:
        llm: Language model instance for parsing (retrieval-optimized model)

    Example:
        >>> parser = GuidelineFactParser()
        >>> guidelines = '''
        ... List current medications. Include dosage and frequency.
        ... [NOTE_TYPES: Medical Note, Prescription]
        ... '''
        >>> requirements = parser.parse_subsection_requirements(
        ...     section_title="Treatment",
        ...     subsection_title="Medications",
        ...     subsection_guidelines=guidelines
        ... )
        >>> for fact in requirements.required_facts:
        ...     print(f"Fact: {fact.description}")
        ...     print(f"Query: {fact.search_query}")
    """

    def __init__(self):
        """Initialize the guideline parser with the retrieval LLM."""
        self.llm = llm_config.llm_retrieve
        logger.debug("GuidelineFactParser initialized with retrieval LLM")

    def parse_subsection_requirements(
        self,
        section_title: str,
        subsection_title: str,
        subsection_guidelines: str
    ) -> SubsectionRequirements:
        """Parse guideline text to extract required facts and instructions.

        This method analyzes guideline text and produces a structured
        SubsectionRequirements object containing all facts that need to
        be answered and formatting instructions.

        Args:
            section_title: Title of the parent section
            subsection_title: Title of this subsection
            subsection_guidelines: Raw guideline text to parse

        Returns:
            SubsectionRequirements with structured fact list and instructions

        Raises:
            FactParsingError: If parsing fails critically

        Example:
            >>> parser = GuidelineFactParser()
            >>> requirements = parser.parse_subsection_requirements(
            ...     section_title="Medical History",
            ...     subsection_title="Chronic Conditions",
            ...     subsection_guidelines="List all chronic conditions..."
            ... )
            >>> print(requirements.complexity_score)
            5
        """
        logger.info(f"Parsing requirements for subsection: '{subsection_title}'")

        try:
            # STEP 1: Extract note types for hardcoded filtering
            note_types = self._extract_note_types(subsection_guidelines)

            # STEP 2: Remove NOTE_TYPES markers from text before sending to LLM
            # These markers are for hardcoded filtering, not LLM processing
            cleaned_guidelines = self._remove_note_types_marker(subsection_guidelines)

            # STEP 3: Call LLM to parse facts using JSON output
            requirements = self._parse_with_llm(
                section_title=section_title,
                subsection_title=subsection_title,
                cleaned_guidelines=cleaned_guidelines,
                note_types=note_types
            )

            if not requirements:
                logger.warning(
                    f"LLM parsing failed for '{subsection_title}', using fallback"
                )
                return self._create_fallback_requirements(
                    subsection_title, subsection_guidelines, note_types
                )

            logger.info(
                f"Successfully parsed {len(requirements.required_facts)} facts "
                f"for '{subsection_title}'"
            )
            return requirements

        except Exception as e:
            logger.error(
                f"Critical parsing failure for '{subsection_title}': {e}",
                exc_info=True
            )
            return self._create_fallback_requirements(
                subsection_title, subsection_guidelines, note_types
            )

    def _extract_note_types(self, subsection_guidelines: str) -> Optional[List[str]]:
        """Extract NOTE_TYPES marker from guideline text.

        Looks for patterns like:
        - [NOTE_TYPES: Medical Note, Nursing Note]
        - [NOTE_TYPES: ALL]

        Args:
            subsection_guidelines: Raw guideline text

        Returns:
            List of note types, or None if [NOTE_TYPES: ALL] or not found

        Example:
            >>> parser = GuidelineFactParser()
            >>> text = "Find medication [NOTE_TYPES: Medical Note, Prescription]"
            >>> note_types = parser._extract_note_types(text)
            >>> print(note_types)
            ['Medical Note', 'Prescription']
        """
        match = re.search(NOTE_TYPES_PATTERN, subsection_guidelines, re.IGNORECASE)

        if not match:
            logger.debug("No NOTE_TYPES found, defaulting to ALL")
            return None  # No restriction = search all

        note_types_str = match.group(1).strip()

        # Check for ALL keyword
        if note_types_str.upper() == 'ALL':
            logger.debug("NOTE_TYPES: ALL - no filtering")
            return None  # None = search all note types

        # Parse comma-separated list
        note_types = [nt.strip() for nt in note_types_str.split(',')]
        note_types = [nt for nt in note_types if nt]  # Remove empty strings

        if note_types:
            logger.debug(f"NOTE_TYPES extracted: {note_types}")
            return note_types
        else:
            logger.debug("Empty NOTE_TYPES list, defaulting to ALL")
            return None

    def _remove_note_types_marker(self, subsection_guidelines: str) -> str:
        """Remove [NOTE_TYPES: ...] markers from guideline text.

        These markers are used for hardcoded filtering and should not be
        sent to the LLM as they might confuse query generation.

        Args:
            subsection_guidelines: Guideline text with possible NOTE_TYPES markers

        Returns:
            Cleaned guideline text without NOTE_TYPES markers

        Example:
            >>> parser = GuidelineFactParser()
            >>> text = "Find meds [NOTE_TYPES: Medical] in records"
            >>> cleaned = parser._remove_note_types_marker(text)
            >>> print(cleaned)
            Find meds  in records
        """
        # Remove NOTE_TYPES markers
        cleaned_text = re.sub(
            NOTE_TYPES_PATTERN,
            '',
            subsection_guidelines,
            flags=re.IGNORECASE
        )

        # Clean up extra whitespace left behind
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    def _parse_with_llm(
        self,
        section_title: str,
        subsection_title: str,
        cleaned_guidelines: str,
        note_types: Optional[List[str]]
    ) -> Optional[SubsectionRequirements]:
        """Use LLM to parse guidelines into structured requirements.

        Args:
            section_title: Parent section title
            subsection_title: Subsection title
            cleaned_guidelines: Guidelines with NOTE_TYPES markers removed
            note_types: Extracted note types for filtering

        Returns:
            SubsectionRequirements if successful, None otherwise

        Example:
            >>> requirements = parser._parse_with_llm(
            ...     "Treatment", "Medications", "List current meds", None
            ... )
        """
        # Build parsing prompt
        parsing_prompt = GUIDELINE_PARSING_PROMPT_TEMPLATE.format(
            section_title=section_title,
            subsection_title=subsection_title,
            cleaned_guidelines=cleaned_guidelines
        )

        
        print(f"[INFO] Parsing requirements for '{subsection_title}'")

        # STEP 1: Extract note types (for hardcoded filtering)
        note_types = self._extract_note_types(subsection_guidelines)

        # STEP 2: Remove NOTE_TYPES markers from text before sending to LLM
        # The LLM doesn't need to see these markers as filtering is hardcoded
        cleaned_guidelines = self._remove_note_types_marker(subsection_guidelines)

        # STEP 3: Fetch facts with LLM using JSON output
        
        parsing_prompt = f"""
Du er en ekspert i at analysere medicinske retningslinjer.

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
        
        try:
            # Call LLM with retry logic
            response = safe_llm_invoke(
                prompt=parsing_prompt,
                llm=self.llm,
                max_retries=3,
                fallback_response=None,
                operation="guideline_parsing"
            )

            if not response or not response.strip():
                logger.error("Empty response from LLM")
                return None

            # Parse JSON response
            requirements = self._parse_json_response(
                response=response,
                section_title=section_title,
                subsection_title=subsection_title,
                note_types=note_types
            )

            return requirements

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            return None

    def _parse_json_response(
        self,
        response: str,
        section_title: str,
        subsection_title: str,
        note_types: Optional[List[str]]
    ) -> Optional[SubsectionRequirements]:
        """Parse JSON response from LLM into SubsectionRequirements.

        Args:
            response: Raw LLM response (should contain JSON)
            section_title: Parent section title
            subsection_title: Subsection title
            note_types: Note types for filtering

        Returns:
            SubsectionRequirements if parsing succeeds, None otherwise

        Example:
            >>> response = '{"required_facts": [...], "format_instructions": "..."}'
            >>> requirements = parser._parse_json_response(
            ...     response, "Treatment", "Medications", None
            ... )
        """
        # Extract JSON from response (in case LLM added extra text)
        response_clean = response.strip()

        # Find JSON block
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')

        if start_idx == -1 or end_idx == -1:
            logger.error("No JSON found in LLM response")
            logger.debug(f"Response preview: {response[:200]}")
            return None

        json_str = response_clean[start_idx:end_idx + 1]

        try:
            data = json.loads(json_str)

            # Validate required fields
            if 'required_facts' not in data:
                logger.error("Missing 'required_facts' in JSON response")
                return None

            # Parse required facts
            required_facts = []
            for fact_data in data.get('required_facts', []):
                fact = self._create_fact_from_json(
                    fact_data, section_title, subsection_title, note_types
                )
                if fact:
                    required_facts.append(fact)

            # Extract format instructions
            format_instructions = data.get('format_instructions', '')

            # Calculate complexity
            complexity = min(len(required_facts), 10)

            logger.info(f"Successfully parsed {len(required_facts)} required facts")

            return SubsectionRequirements(
                subsection_title=subsection_title,
                required_facts=required_facts,
                format_instructions=format_instructions,
                complexity_score=complexity,
                note_types=note_types
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"JSON string: {json_str[:300]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {e}", exc_info=True)
            return None

    def _create_fact_from_json(
        self,
        fact_data: Dict,
        section_title: str,
        subsection_title: str,
        note_types: Optional[List[str]]
    ) -> Optional[RequiredFact]:
        """Create RequiredFact object from JSON fact data.

        Args:
            fact_data: Dictionary with fact information from JSON
            section_title: Parent section title (for context)
            subsection_title: Subsection title (for context)
            note_types: Note types for this fact

        Returns:
            RequiredFact if creation succeeds, None otherwise

        Example:
            >>> fact_data = {
            ...     "description": "Current medications",
            ...     "search_query": "medications current treatment"
            ... }
            >>> fact = parser._create_fact_from_json(
            ...     fact_data, "Treatment", "Medications", None
            ... )
            >>> print(fact.description)
            Current medications
        """
        try:
            description = fact_data.get('description', '').strip()
            search_query = fact_data.get('search_query', description).strip()

            if not description:
                logger.warning("Skipping fact with empty description")
                return None

            return RequiredFact(
                description=description,
                priority=FactPriority.REQUIRED,
                search_query=search_query,
                note_types=note_types
            )

        except Exception as e:
            logger.error(f"Failed to create fact from JSON: {e}")
            return None

    def _create_fallback_requirements(
        self,
        subsection_title: str,
        subsection_guidelines: str,
        note_types: Optional[List[str]]
    ) -> SubsectionRequirements:
        """Create minimal fallback requirements when parsing fails.

        This provides a degraded but functional fallback that allows
        generation to proceed even when guideline parsing fails.

        Args:
            subsection_title: Subsection title
            subsection_guidelines: Raw guideline text
            note_types: Note types for filtering

        Returns:
            SubsectionRequirements with single generic fact

        Example:
            >>> requirements = parser._create_fallback_requirements(
            ...     "Medications", "List medications", None
            ... )
            >>> print(len(requirements.required_facts))
            1
        """
        logger.warning(f"Creating fallback requirements for '{subsection_title}'")

        # Create single generic fact
        fallback_fact = RequiredFact(
            description=f"Information relateret til {subsection_title}",
            priority=FactPriority.REQUIRED,
            search_query=subsection_title.lower(),
            note_types=note_types
        )

        return SubsectionRequirements(
            subsection_title=subsection_title,
            required_facts=[fallback_fact],
            format_instructions=subsection_guidelines[:200],  # First 200 chars
            complexity_score=1,
            note_types=note_types
        )
