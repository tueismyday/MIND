"""
Generates subsections using fact-by-fact approach.
Each fact is answered independently, then assembled into coherent subsection.
"""

from typing import Dict, List, Tuple
from generation.fact_parser import RequiredFact
from config.llm_config import llm_config
from utils.error_handling import safe_llm_invoke


class FactBasedGenerator:
    """Generates text using fact-by-fact answering approach"""
    
    def __init__(self):
        self.llm = llm_config.llm_generate
        
        # Template string - will be formatted later with actual values
        self.fact_prompt_template = """
        Du er en sygeplejerske der skal besvare ét specifikt faktum baseret på patientjournalen.
        
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
        
        # Template string - will be formatted later with actual values
        self.assembly_prompt_template = """
        Du er en sygeplejerske der skal sammensætte underafsnittet '{subsection_title}' under sektionen '{section_title}'.
        
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

        
    def answer_single_fact(self, fact: RequiredFact, section_title: str, sources: List[Dict]) -> Tuple[str, bool]:
        """
        Answer a single fact using its retrieved sources.
        
        Args:
            fact: The fact to answer
            sources: Retrieved sources with full_content
            
        Returns:
            (answer_text, is_answerable)
        """
        
        if not sources:
            return "UNANSWERABLE", False
        
        # Format sources with FULL content
        sources_text = self._format_sources_full(sources)
        
        # Format the prompt with actual values
        prompt = self.fact_prompt_template.format(
            fact_description=fact.description,
            subsection_title=subsection_title,
            sources_text=sources_text
        )
        
        try:
            answer = safe_llm_invoke(prompt, self.llm, max_retries=2, operation="fact_answering")

            if not answer or "UNANSWERABLE" in answer.upper():
                return "UNANSWERABLE", False

            return answer.strip(), True

        except Exception as e:
            print(f"[ERROR] Failed to answer fact '{fact.description[:50]}': {e}")
            return "UNANSWERABLE", False
    
    def assemble_subsection_from_facts(self,
                                       subsection_title: str,
                                       section_title: str,
                                       section_intro: str,
                                       format_instructions: str,
                                       fact_answers: List[Dict]) -> Dict:
        """
        Assemble a coherent subsection from individual fact answers.
        
        Args:
            subsection_title: Title of subsection
            section_title: Parent section title
            section_intro: General instructions for entire section
            format_instructions: Specific instructions for this subsection
            fact_answers: List of {fact, answer, sources, answerable}
            
        Returns:
            Dict with 'answer' and 'unanswerable_items'
        """
        
        # Separate answerable and unanswerable
        answerable = [fa for fa in fact_answers if fa['answerable']]
        unanswerable = [fa['fact'].description for fa in fact_answers if not fa['answerable']]
        
        if not answerable:
            # Nothing could be answered
            return {
                'answer': '',
                'unanswerable_items': unanswerable
            }
        
        # Format the individual fact answers
        fact_texts = []
        for fa in answerable:
            fact_texts.append(f"• {fa['answer']}")
        
        combined_facts = "\n".join(fact_texts)
        
        # Format the assembly prompt with actual values
        assembly_prompt = self.assembly_prompt_template.format(
            subsection_title=subsection_title,
            section_title=section_title,
            section_intro=section_intro,
            format_instructions=format_instructions if format_instructions else "Ingen yderligere specifikke instruktioner",
            combined_facts=combined_facts
        )
        
        try:
            assembled_text = safe_llm_invoke(assembly_prompt, self.llm, max_retries=2, operation="fact_assembly")

            if not assembled_text:
                # Fallback: just concatenate
                assembled_text = "\n\n".join([fa['answer'] for fa in answerable])

            return {
                'answer': assembled_text.strip(),
                'unanswerable_items': unanswerable
            }

        except Exception as e:
            print(f"[ERROR] Assembly failed: {e}, using simple concatenation")
            return {
                'answer': "\n\n".join([fa['answer'] for fa in answerable]),
                'unanswerable_items': unanswerable
            }
    
    def _format_sources_full(self, sources: List[Dict]) -> str:
        """Format sources with FULL content (no truncation)"""
        lines = []
        
        for i, source in enumerate(sources, 1):
            entry_type = source.get('entry_type', 'Note')
            timestamp = source.get('timestamp', 'Ukendt dato')
            relevance = source.get('relevance', 0)
            
            # Use FULL content, not truncated
            content = source.get('full_content', source.get('snippet', ''))
            
            lines.append(f"--- KILDE {i}: {entry_type} ({timestamp}) - Relevans: {relevance}% ---")
            lines.append(content)
            lines.append("")
        
        return "\n".join(lines)
