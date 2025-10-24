"""
Fact parsing and requirement extraction from guidelines using JSON output.
Identifies what information needs to be retrieved from EHR.
"""

from typing import List, Dict
from dataclasses import dataclass
import json
import re

from config.llm_config import llm_config
from utils.error_handling import safe_llm_invoke


@dataclass
class RequiredFact:
    """Represents a single fact that needs to be found in EHR"""
    description: str  # What to look for
    priority: str  # "required" or "optional"
    search_query: str  # Optimized RAG query for this fact
    category: str  # Clinical category
    note_types: List[str] = None # Allowed note types for this fact


@dataclass
class SubsectionRequirements:
    """Structured requirements for a subsection"""
    subsection_title: str
    required_facts: List[RequiredFact]
    format_instructions: str
    complexity_score: int  # 0-10, based on number of facts
    note_types: List[str] = None  # Note types for entire subsection


class GuidelineFactParser:
    """Parses guidelines to extract factual requirements using JSON"""
    
    def __init__(self):
        self.llm = llm_config.llm_retrieve

    def _extract_note_types(self, subsection_guidelines: str) -> List[str]:
        """
        Extract NOTE_TYPES from guideline text.
        
        Args:
            subsection_guidelines: Guideline text
            
        Returns:
            List of note types, or None if [NOTE_TYPES: ALL]
        """
        # Pattern: [NOTE_TYPES: type1, type2, type3] or [NOTE_TYPES: ALL]
        pattern = r'\[NOTE_TYPES:\s*([^\]]+)\]'
        match = re.search(pattern, subsection_guidelines, re.IGNORECASE)
        
        if not match:
            print("[INFO] No NOTE_TYPES found, defaulting to ALL")
            return None  # No restriction = search all
        
        note_types_str = match.group(1).strip()
        
        # Check for ALL keyword
        if note_types_str.upper() == 'ALL':
            print("[INFO] NOTE_TYPES: ALL - no filtering")
            return None  # None = search all note types
        
        # Parse comma-separated list
        note_types = [nt.strip() for nt in note_types_str.split(',')]
        note_types = [nt for nt in note_types if nt]  # Remove empty strings
        
        print(f"[INFO] NOTE_TYPES extracted: {note_types}")
        return note_types if note_types else None
    
    def parse_subsection_requirements(self, 
                                     section_title: str,
                                     subsection_title: str,
                                     subsection_guidelines: str) -> SubsectionRequirements:
        """
        Parse guideline text to extract required facts using JSON.
        
        Args:
            section_title: Parent section title
            subsection_title: Subsection title
            subsection_guidelines: The guideline text to parse
            
        Returns:
            SubsectionRequirements with structured fact list
        """
        
        print(f"[INFO] Parsing requirements for '{subsection_title}'")

        # STEP 1: Extract note types
        note_types = self._extract_note_types(subsection_guidelines)

        # STEP 2: Fetch facts with LLM using JSON output
        
        parsing_prompt = f"""
Du er en ekspert i at analysere medicinske retningslinjer.

SEKTION: {section_title}
UNDERAFSNIT: {subsection_title}

RETNINGSLINJER:
{subsection_guidelines}

{"VIGTIGT: Denne subsection skal KUN bruge følgende note-typer: " + ", ".join(note_types) if note_types else "NOTE: Alle note-typer er tilladt"}

Din opgave er at identificere hvilke KONKRETE FAKTA der skal findes i patientjournalen, 
samt udtrække SPECIFIKKE FORMAT-INSTRUKTIONER for dette underafsnit.

VIGTIGT - Håndter følgende:
1. Hvis retningslinjen siger "Hvis ja...", udtræk BEGGE muligheder (ja og nej scenarios)
2. Når der står "Brug X notat", betyder det der skal søge i den notetype
3. FORMAT KRAV skal indeholde instruktioner om HVORDAN der skal svares (f.eks. "besvar kort", "pas på ikke at konkludere", "giv bedste bud")
4. Lister med "f.eks." betyder ALLE eksemplerne er potentielle fakta

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
            # Use safe LLM invoke with retry
            response = safe_llm_invoke(
                parsing_prompt,
                self.llm,
                max_retries=3,
                fallback_response=None
            )
            
            if not response or not response.strip():
                print(f"[ERROR] Empty response from LLM - using fallback")
                return self._create_fallback_requirements(subsection_title, subsection_guidelines, note_types)
            
            # Parse JSON response
            requirements = self._parse_json_response(response, section_title, subsection_title, note_types)
            
            if not requirements:
                print(f"[ERROR] Failed to parse JSON - using fallback")
                return self._create_fallback_requirements(subsection_title, subsection_guidelines, note_types)
            
            return requirements
        
        except Exception as e:
            print(f"[CRITICAL] Fact parsing completely failed: {str(e)}")
            return self._create_fallback_requirements(subsection_title, subsection_guidelines, note_types)

    def _parse_json_response(self, 
                            response: str, 
                            section_title: str,
                            subsection_title: str,
                            note_types: List[str]) -> SubsectionRequirements:
        """Parse JSON response from LLM"""
        
        # Extract JSON from response (in case LLM added extra text)
        response_clean = response.strip()
        
        # Find JSON block
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            print("[ERROR] No JSON found in response")
            print(f"[DEBUG] Response preview: {response[:200]}")
            return None
        
        json_str = response_clean[start_idx:end_idx+1]
        
        try:
            data = json.loads(json_str)
            
            # Validate required fields
            if 'required_facts' not in data:
                print("[ERROR] Missing 'required_facts' in JSON")
                return None
            
            # Parse required facts
            required_facts = []
            for fact_data in data.get('required_facts', []):
                fact = self._create_fact_from_json(fact_data, section_title, subsection_title, note_types)
                if fact:
                    required_facts.append(fact)

            # Extract format instructions
            format_instructions = data.get('format_instructions', '')

            # Calculate complexity
            complexity = len(required_facts)

            print(f"[SUCCESS] Parsed {len(required_facts)} required facts")

            return SubsectionRequirements(
                subsection_title=subsection_title,
                required_facts=required_facts,
                format_instructions=format_instructions,
                complexity_score=min(complexity, 10),
                note_types=note_types
            )
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON: {str(e)}")
            print(f"[DEBUG] JSON string: {json_str[:300]}")
            return None
    
    def _create_fact_from_json(self,
                               fact_data: Dict,
                               section_title: str,
                               subsection_title: str,
                               note_types: List[str]) -> RequiredFact:
        """Create RequiredFact from JSON data"""
        
        try:
            description = fact_data.get('description', '').strip()
            search_query = fact_data.get('search_query', description).strip()
            
            if not description:
                return None
            
            # Infer category
            category = self._infer_category(description)
            
            return RequiredFact(
                description=description,
                priority="required",
                search_query=search_query,
                category=category,
                note_types=note_types
            )
            
        except Exception as e:
            print(f"[ERROR] Failed to create fact from JSON: {e}")
            return None
    
    def _create_fallback_requirements(self, 
                                     subsection_title: str, 
                                     subsection_guidelines: str,
                                     note_types: List[str]) -> SubsectionRequirements:
        """Create minimal requirements when parsing fails"""
        
        print(f"[FALLBACK] Creating minimal requirements for '{subsection_title}'")
        
        # Create single generic fact
        fallback_fact = RequiredFact(
            description=f"Information relateret til {subsection_title}",
            priority="required",
            search_query=subsection_title.lower(),
            category="ukategoriseret",
            note_types=note_types
        )
        
        return SubsectionRequirements(
            subsection_title=subsection_title,
            required_facts=[fallback_fact],
            format_instructions=subsection_guidelines[:200],
            complexity_score=1,
            note_types=note_types
        )
     

    def _infer_category(self, description: str) -> str:
            """Infer clinical category from fact description"""
            
            from config.settings import DANISH_CLINICAL_CATEGORIES
            
            description_lower = description.lower()
            
            for category, keywords in DANISH_CLINICAL_CATEGORIES.items():
                if any(kw in description_lower for kw in keywords):
                    return category
            
            return "ukategoriseret"