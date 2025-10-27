"""
Enhanced section generation with fact-by-fact approach.
Each fact is answered independently, validated, then assembled.
"""

from typing import Tuple, List, Dict
from generation.fact_parser import GuidelineFactParser, SubsectionRequirements
from generation.fact_based_generator import FactBasedGenerator
from generation.fact_validator import FactValidator
from tools.patient_tools import get_patient_info_with_sources
from utils.error_handling import safe_rag_search
from utils.profiling import profile
from config.settings import DEFAULT_VALIDATION_CYCLES, MAX_VALIDATION_CYCLES, MIN_VALIDATION_CYCLES


@profile
def generate_subsection_with_hybrid_approach(
    section_title: str,
    subsection_title: str,
    section_intro: str,
    subsection_guidelines: str,
    patient_data: str,
    max_sources_per_fact: int = 2,
    enable_validation: bool = True,
    max_revision_cycles: int = None
) -> Tuple[str, List[Dict], Dict]:
    """
    Generate subsection using FACT-BY-FACT approach with per-fact validation.
    
    Pipeline:
    1. Parse guideline → identify required facts
    2. For each fact:
       a. RAG search for this fact
       b. LLM answers this fact
       c. Validate answer (if enabled)
    3. Assemble subsection from fact-answers
    
    Args:
        section_title: Parent section title
        subsection_title: Subsection title
        section_intro: Section introduction/instructions
        subsection_guidelines: Guidelines for this subsection
        patient_data: Patient data (unused in new approach, kept for compatibility)
        max_sources_per_fact: Max RAG sources per fact
        enable_validation: Whether to enable fact-level validation
        max_revision_cycles: Maximum validation/revision cycles (default: from settings)
        
    Returns:
        (generated_text, sources_list, validation_details)
    """
    
    # Set default cycles if not provided
    if max_revision_cycles is None:
        max_revision_cycles = DEFAULT_VALIDATION_CYCLES
    
    # Enforce limits
    max_revision_cycles = max(MIN_VALIDATION_CYCLES, min(max_revision_cycles, MAX_VALIDATION_CYCLES))
    
    print(f"\n[INFO] === Fact-by-Fact Generation: '{subsection_title}' ===")
    print(f"[CONFIG] Validation: {enable_validation}, Max cycles: {max_revision_cycles}\n")

    try:
    
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Parse Guidelines → Identify Required Facts
        # ═══════════════════════════════════════════════════════════════
        
        parser = GuidelineFactParser()
        requirements = parser.parse_subsection_requirements(
            section_title=section_title,
            subsection_title=subsection_title,
            subsection_guidelines=subsection_guidelines
        )
        
        print(f"\n[PHASE 1 COMPLETE] Identified {len(requirements.required_facts)} facts to retrieve\n")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: Answer Each Fact Independently
        # ═══════════════════════════════════════════════════════════════
        
        generator = FactBasedGenerator()
        validator = FactValidator() if enable_validation else None
        
        fact_answers = []
        all_sources = []
        validation_stats = {
            'total_facts': len(requirements.required_facts),
            'answered_facts': 0,
            'unanswered_facts': 0,
            'validated_facts': 0,
            'corrected_facts': 0
        }
        
        print(f"[PHASE 2] Answering {len(requirements.required_facts)} facts individually...\n")
        
        for i, fact in enumerate(requirements.required_facts, 1):
            print(f"[FACT {i}/{len(requirements.required_facts)}] {fact.description[:60]}...")
            
            # 2a. RAG retrieval for this specific fact
            note_types = fact.note_types or requirements.note_types
            
            if note_types:
                print(f"  → Filtering by: {note_types}")
            
            rag_result = safe_rag_search(
                query=fact.search_query,
                max_references=max_sources_per_fact,
                note_types=note_types
            )
            
            sources = rag_result.sources
            print(f"  → Retrieved {len(sources)} sources")
            
            if not sources:
                print(f"  ✗ No sources found - marking as unanswerable")
                fact_answers.append({
                    'fact': fact,
                    'answer': 'UNANSWERABLE',
                    'sources': [],
                    'answerable': False,
                    'validated': False,
                    'corrected': False
                })
                validation_stats['unanswered_facts'] += 1
                continue
            
            # 2b. LLM answers this fact
            answer, is_answerable = generator.answer_single_fact(fact, sources)
            
            if not is_answerable:
                print(f"  ✗ Could not answer from sources")
                fact_answers.append({
                    'fact': fact,
                    'answer': 'UNANSWERABLE',
                    'sources': sources,
                    'answerable': False,
                    'validated': False,
                    'corrected': False
                })
                validation_stats['unanswered_facts'] += 1
                continue
            
            print(f"  ✓ Answered: {answer[:60]}...")
            validation_stats['answered_facts'] += 1
            
            # 2c. Validate this fact answer (if enabled)
            was_corrected = False
            if enable_validation and validator:
                validated_answer, was_corrected = validator.validate_fact_answer(
                    fact, answer, sources, max_retries=max_revision_cycles
                )
                answer = validated_answer
                validation_stats['validated_facts'] += 1
                if was_corrected:
                    validation_stats['corrected_facts'] += 1
            
            # 2d. Store result
            fact_answers.append({
                'fact': fact,
                'answer': answer,
                'sources': sources,
                'answerable': True,
                'validated': enable_validation,
                'corrected': was_corrected
            })
            
            all_sources.extend(sources)
            print()
        
        print(f"[PHASE 2 COMPLETE] Answered {validation_stats['answered_facts']}/{validation_stats['total_facts']} facts")
        if enable_validation:
            print(f"  → Validated: {validation_stats['validated_facts']}, Corrected: {validation_stats['corrected_facts']}\n")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: Assemble Subsection from Fact-Answers
        # ═══════════════════════════════════════════════════════════════
        
        print(f"[PHASE 3] Assembling subsection...\n")
        
        result_json = generator.assemble_subsection_from_facts(
            subsection_title=subsection_title,
            section_title=section_title,
            section_intro=section_intro,
            format_instructions=requirements.format_instructions,
            fact_answers=fact_answers
        )
        
        assembled_text = result_json['answer']
        unanswerable_items = result_json['unanswerable_items']
        
        # Format final output
        final_output = f"SUBSECTION_TITLE: {subsection_title}\n\n"
        
        if assembled_text:
            final_output += assembled_text
        
        if unanswerable_items:
            final_output += f"\n\nKunne ikke besvares ud fra patientjournalen:\n"
            for item in unanswerable_items:
                final_output += f"• {item}\n"
        
        print(f"[PHASE 3 COMPLETE] Subsection assembled ({len(assembled_text)} chars)")
        
        # ═══════════════════════════════════════════════════════════════
        # Return Results
        # ═══════════════════════════════════════════════════════════════
        
        # Deduplicate sources
        unique_sources = _deduplicate_sources(all_sources)
        
        validation_details = {
            'enabled': enable_validation,
            'stats': validation_stats,
            'max_cycles': max_revision_cycles if enable_validation else 0
        }
        
        return final_output, unique_sources, validation_details
        
    except Exception as e:
        print(f"[ERROR] Fact-by-fact generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal fallback
        fallback = f"SUBSECTION_TITLE: {subsection_title}\n\n[Kunne ikke generere indhold for {subsection_title}]"
        return fallback, [], {'enabled': False, 'stats': {}, 'error': str(e)}


def _deduplicate_sources(sources: List[Dict]) -> List[Dict]:
    """Remove duplicate sources based on timestamp + entry_type"""
    seen = set()
    unique = []
    
    for source in sources:
        key = (source.get('timestamp', ''), source.get('entry_type', ''))
        if key not in seen:
            seen.add(key)
            unique.append(source)
    
    return unique


# ═══════════════════════════════════════════════════════════════
# Section-Level Function (Splits into Subsections)
# ═══════════════════════════════════════════════════════════════

def generate_section_with_hybrid_approach(
    section_title: str,
    section_guidelines: str,
    patient_data: str,
    max_sources_per_fact: int = 2,
    enable_validation: bool = True
) -> Tuple[str, List[Dict], Dict]:
    """
    Generate entire section by splitting into subsections.

    This function exists to maintain compatibility with document_generator.
    It splits the section into subsections and calls generate_subsection_with_hybrid_approach
    for each one.
    """

    from utils.text_processing import split_section_into_subsections

    # Type safety: ensure section_guidelines is a string
    if not isinstance(section_guidelines, str):
        print(f"[WARNING] section_guidelines is not a string (type: {type(section_guidelines)}). Converting to string.")
        section_guidelines = str(section_guidelines) if section_guidelines else ""

    subsections = split_section_into_subsections(section_guidelines)
    
    if not subsections:
        # Treat entire section as single subsection
        return generate_subsection_with_hybrid_approach(
            section_title=section_title,
            subsection_title=section_title,
            section_intro=section_guidelines[:500],
            subsection_guidelines=section_guidelines,
            patient_data=patient_data,
            max_sources_per_fact=max_sources_per_fact,
            enable_validation=enable_validation
        )
    
    # Generate each subsection
    all_subsection_outputs = []
    all_sources = []
    all_validation_details = {}

    # Safely extract section intro with type checking
    section_intro = ""
    if subsections and isinstance(subsections, list) and len(subsections) > 0:
        first_subsection = subsections[0]
        if isinstance(first_subsection, dict) and 'intro' in first_subsection:
            section_intro = first_subsection['intro']
        else:
            # Fallback: use first 500 chars of section_guidelines
            section_intro = str(section_guidelines)[:500] if section_guidelines else ""
    else:
        section_intro = str(section_guidelines)[:500] if section_guidelines else ""
    
    for subsection in subsections:
        # Type safety: ensure subsection is a dict with required keys
        if not isinstance(subsection, dict):
            print(f"[WARNING] Skipping invalid subsection (not a dict): {type(subsection)}")
            continue

        if 'title' not in subsection or 'content' not in subsection:
            print(f"[WARNING] Skipping subsection missing required keys: {subsection.keys()}")
            continue

        output, sources, validation = generate_subsection_with_hybrid_approach(
            section_title=section_title,
            subsection_title=subsection['title'],
            section_intro=section_intro,
            subsection_guidelines=subsection['content'],
            patient_data=patient_data,
            max_sources_per_fact=max_sources_per_fact,
            enable_validation=enable_validation
        )

        all_subsection_outputs.append(output)
        all_sources.extend(sources)
        all_validation_details[subsection['title']] = validation
    
    # Combine outputs
    final_output = "\n\n".join(all_subsection_outputs)
    unique_sources = _deduplicate_sources(all_sources)
    
    combined_validation = {
        'enabled': enable_validation,
        'subsections': all_validation_details
    }
    
    return final_output, unique_sources, combined_validation
