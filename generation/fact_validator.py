"""
Simple fact-level validator for fact-by-fact generation.
Validates individual fact answers against their sources.
"""

from typing import Dict, List, Tuple
from generation.fact_parser import RequiredFact
from config.llm_config import llm_config
from utils.error_handling import safe_llm_invoke


class FactValidator:
    """Validates individual fact answers"""
    
    def __init__(self):
        self.llm = llm_config.llm_critique
    
    def validate_fact_answer(self, 
                            fact: RequiredFact,
                            answer: str,
                            sources: List[Dict],
                            max_retries: int = 2) -> Tuple[str, bool]:
        """
        Validate and optionally correct a fact answer.
        
        Args:
            fact: The fact that was answered
            answer: The LLM's answer to the fact
            sources: The sources used to answer
            max_retries: Maximum correction attempts
            
        Returns:
            (validated_answer, was_corrected)
        """
        
        if answer == "UNANSWERABLE" or not sources:
            return answer, False
        
        # Format sources for validation
        sources_text = self._format_sources(sources)
        
        validation_prompt = f"""
Du er en kvalitetskontrollør der skal validere et svar om en patient.

FAKTUM DER BLEV BESVARET:
{fact.description}

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
        
        try:
            validation_result = safe_llm_invoke(validation_prompt, self.llm, max_retries=1)
            
            if not validation_result:
                return answer, False
            
            validation_result = validation_result.strip()
            
            if validation_result.upper() == "VALID":
                print(f"  ✓ Fact valid: {fact.description[:50]}")
                return answer, False
            else:
                # LLM provided corrected answer
                print(f"  ⚠ Fact corrected: {fact.description[:50]}")
                return validation_result, True
                
        except Exception as e:
            print(f"[ERROR] Validation failed for fact '{fact.description[:50]}': {e}")
            return answer, False
    
    def _format_sources(self, sources: List[Dict]) -> str:
        """Format sources for validation prompt"""
        lines = []
        
        for i, source in enumerate(sources, 1):
            entry_type = source.get('entry_type', 'Note')
            timestamp = source.get('timestamp', 'Ukendt dato')
            content = source.get('full_content', source.get('snippet', ''))
            
            lines.append(f"Kilde {i}: {entry_type} ({timestamp})")
            lines.append(f"{content}")
            lines.append("")
        
        return "\n".join(lines)
