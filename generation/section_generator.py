"""
Section generation with fact-by-fact approach
Facts are extracted, answered, validated and concatinated for each subsection
before the assembly of a section

- Batched cross-encoder predictions
- Batched LLM fact answering calls
- Batched LLM validation calls
"""

from typing import Tuple, List, Dict
from generation.fact_parser import GuidelineFactParser
from generation.fact_based_generator import FactBasedGenerator
from generation.fact_validator import FactValidator
from utils.text_processing import split_section_into_subsections
from utils.error_handling import safe_rag_search
from utils.profiling import profile
from config.settings import DEFAULT_VALIDATION_CYCLES, MAX_VALIDATION_CYCLES, MIN_VALIDATION_CYCLES, MAX_SOURCES_PER_FACT
import time


def _batch_llm_calls(prompts: List[str], llm_instance, operation: str = "batch", max_retries: int = 2) -> List[str]:
    """
    Batch multiple LLM calls into one request.
    Falls back to sequential if batching fails.
    
    Args:
        prompts: List of prompts to process
        llm_instance: LLM instance (vLLM client)
        operation: Operation name for logging
        max_retries: Max retry attempts for batch call
        
    Returns:
        List of responses (same length as prompts)
    """
    if not prompts:
        return []
    
    # Try batched call
    for attempt in range(max_retries):
        try:
            responses = []
            
            if hasattr(llm_instance, 'client') and hasattr(llm_instance.client, 'chat'):
                for prompt in prompts:
                    response = llm_instance.client.chat.completions.create(
                        model=llm_instance.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=getattr(llm_instance, 'temperature', 0.1),
                        max_tokens=6056,
                    )
                    responses.append(response.choices[0].message.content.strip())
                
                return responses
            
            else:
                # Fallback: sequential calls using invoke
                for prompt in prompts:
                    response = llm_instance.invoke(prompt)
                    responses.append(response.strip())
                
                return responses
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Batch {operation} attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"[ERROR] Batch {operation} failed after {max_retries} attempts, using sequential fallback")
                # Final fallback: sequential with error handling
                responses = []
                for prompt in prompts:
                    try:
                        response = llm_instance.invoke(prompt)
                        responses.append(response.strip())
                    except:
                        responses.append("UNANSWERABLE")
                return responses
    
    return ["UNANSWERABLE"] * len(prompts)


@profile
def generate_subsection_with_hybrid_approach(
    section_title: str,
    subsection_title: str,
    section_intro: str,
    subsection_guidelines: str,
    patient_data: str,
    max_sources_per_fact: MAX_SOURCES_PER_FACT,
    enable_validation: bool = True,
    max_revision_cycles: int = None
) -> Tuple[str, List[Dict], Dict]:
    """
    Generate subsection using FACT-BY-FACT approach with batched LLM calls.
    
    Pipeline:
    1. Parse guideline --> identify required facts
    2. For each fact: RAG search
    3. BATCH: Answer all facts together
    4. BATCH: Validate all facts together (if enabled)
    5. Assemble subsection from fact-answers
    
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
    
        # =============================================================================================================================================================================================
        # PHASE 1: Parse Guidelines --> Identify Required Facts
        # =============================================================================================================================================================================================
        
        parser = GuidelineFactParser()
        requirements = parser.parse_subsection_requirements(
            section_title=section_title,
            subsection_title=subsection_title,
            subsection_guidelines=subsection_guidelines
        )
        
        print(f"\n[PHASE 1 COMPLETE] Identified {len(requirements.required_facts)} facts to retrieve\n")
        
        # =============================================================================================================================================================================================
        # PHASE 2: Answer Each Fact (with BATCHED LLM calls)
        # =============================================================================================================================================================================================
        
        generator = FactBasedGenerator()
        validator = FactValidator() if enable_validation else None
        
        fact_sources_pairs = []  # (fact, sources) tuples
        all_sources = []
        validation_stats = {
            'total_facts': len(requirements.required_facts),
            'answered_facts': 0,
            'unanswered_facts': 0,
            'validated_facts': 0,
            'corrected_facts': 0
        }
        
        print(f"[PHASE 2] Retrieving sources for {len(requirements.required_facts)} facts...\n")
        
        # Step 2a: RAG retrieval for all facts (sequential, but faster than original)
        for i, fact in enumerate(requirements.required_facts, 1):
            print(f"[FACT {i}/{len(requirements.required_facts)}] {fact.description[:60]}...")
            
            note_types = fact.note_types or requirements.note_types
            
            if note_types:
                print(f"  --> Filtering by: {note_types}")
            
            rag_result = safe_rag_search(
                query=fact.search_query,
                max_references=max_sources_per_fact,
                note_types=note_types
            )
            
            sources = rag_result.sources
            print(f"  --> Retrieved {len(sources)} sources")
            
            fact_sources_pairs.append((fact, sources))
            all_sources.extend(sources)
        
        print(f"\n[PHASE 2A COMPLETE] Retrieved sources for all facts\n")
        
        # Step 2b: BATCH answer all facts
        print(f"[PHASE 2B] Batching LLM calls to answer {len(fact_sources_pairs)} facts...\n")
        
        answering_prompts = []
        answerable_indices = []  # Track which facts have sources
        
        for idx, (fact, sources) in enumerate(fact_sources_pairs):
            if not sources:
                answering_prompts.append(None)
                continue
            
            sources_text = generator._format_sources_full(sources)
            
            # Format the prompt template with actual values
            prompt = generator.fact_prompt_template.format(
                fact_description=fact.description,
                subsection_title=subsection_title,
                sources_text=sources_text
            )
            answering_prompts.append(prompt)
            answerable_indices.append(idx)
        
        # Batch call for answering
        valid_prompts = [p for p in answering_prompts if p is not None]
        
        if valid_prompts:
            batch_answers = _batch_llm_calls(valid_prompts, generator.llm, operation="fact_answering")
        else:
            batch_answers = []
        
        # Map answers back to facts
        fact_answers_raw = []
        batch_idx = 0
        
        for idx, (fact, sources) in enumerate(fact_sources_pairs):
            if idx in answerable_indices:
                answer = batch_answers[batch_idx]
                batch_idx += 1
                
                is_answerable = answer and "UNANSWERABLE" not in answer.upper()
                
                if is_answerable:
                    print(f"  --> [{idx+1}] Answered: {answer[:60]}...")
                    validation_stats['answered_facts'] += 1
                else:
                    print(f"  --> [{idx+1}] Could not answer from sources")
                    validation_stats['unanswered_facts'] += 1
                
                fact_answers_raw.append((fact, sources, answer, is_answerable))
            else:
                print(f"  --> [{idx+1}] No sources found")
                validation_stats['unanswered_facts'] += 1
                fact_answers_raw.append((fact, sources, "UNANSWERABLE", False))
        
        print(f"\n[PHASE 2B COMPLETE] Answered {validation_stats['answered_facts']}/{validation_stats['total_facts']} facts\n")
        
        # Step 2c: BATCH validate all facts (if enabled)
        if enable_validation and validator:
            print(f"[PHASE 2C] Batching validation for {len(fact_answers_raw)} facts...\n")
            
            # Build validation prompts using EXACT original formatting
            validation_prompts = []
            validatable_indices = []
            
            for idx, (fact, sources, answer, is_answerable) in enumerate(fact_answers_raw):
                if answer == "UNANSWERABLE" or not sources or not is_answerable:
                    validation_prompts.append(None)
                    continue
                
                # Format sources for validation
                sources_text = validator._format_sources(sources)
                
                # Format the validation prompt template with actual values
                prompt = validator.validation_prompt_template.format(
                    fact_description=fact.description,
                    answer=answer,
                    sources_text=sources_text
                )
                validation_prompts.append(prompt)
                validatable_indices.append(idx)
            
            # Batch validation calls
            valid_validation_prompts = [p for p in validation_prompts if p is not None]
            
            if valid_validation_prompts:
                batch_validations = _batch_llm_calls(valid_validation_prompts, validator.llm, operation="fact_validation", max_retries=1)
            else:
                batch_validations = []
            
            # Apply validation results
            fact_answers_final = []
            validation_idx = 0
            
            for idx, (fact, sources, answer, is_answerable) in enumerate(fact_answers_raw):
                if idx in validatable_indices:
                    validation_result = batch_validations[validation_idx]
                    validation_idx += 1
                    
                    validation_stats['validated_facts'] += 1
                    
                    if validation_result.strip().upper() == "VALID":
                        print(f"  --> Fact valid: {fact.description[:50]}")
                        fact_answers_final.append((fact, sources, answer, False))
                    else:
                        print(f"  -->  Fact corrected: {fact.description[:50]}")
                        validation_stats['corrected_facts'] += 1
                        fact_answers_final.append((fact, sources, validation_result, True))
                else:
                    fact_answers_final.append((fact, sources, answer, False))
            
            print(f"\n[PHASE 2C COMPLETE] Validated {validation_stats['validated_facts']} facts, corrected {validation_stats['corrected_facts']}\n")
        else:
            fact_answers_final = [(f, s, a, False) for f, s, a, _ in fact_answers_raw]
        
        # Build fact_answers structure for assembly
        fact_answers = []
        for fact, sources, answer, was_corrected in fact_answers_final:
            is_answerable = answer and "UNANSWERABLE" not in answer.upper()
            
            fact_answers.append({
                'fact': fact,
                'answer': answer if is_answerable else 'UNANSWERABLE',
                'sources': sources,
                'answerable': is_answerable,
                'validated': enable_validation,
                'corrected': was_corrected
            })
        
        print(f"[PHASE 2 COMPLETE] Answered {validation_stats['answered_facts']}/{validation_stats['total_facts']} facts")
        if enable_validation:
            print(f"  --> Validated: {validation_stats['validated_facts']}, Corrected: {validation_stats['corrected_facts']}\n")
        
        # =============================================================================================================================================================================================
        # PHASE 3: Assemble Subsection from Fact-Answers
        # =============================================================================================================================================================================================
        
        print(f"[PHASE 3] Assembling subsection...\n")
        
        # Use EXACT original assembly method
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
                final_output += f"{item}\n"
        
        print(f"[PHASE 3 COMPLETE] Subsection assembled ({len(assembled_text)} chars)")
        
        # =============================================================================================================================================================================================
        # Return Results
        # =
        
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


# =============================================================================================================================================================================================
# Section-Level Function (Splits into Subsections)
# =============================================================================================================================================================================================

def generate_section_with_hybrid_approach(
    section_title: str,
    section_guidelines: str,
    patient_data: str,
    max_sources_per_fact: MAX_SOURCES_PER_FACT,
    enable_validation: bool = True
) -> Tuple[str, List[Dict], Dict]:
    """
    Generate entire section by splitting into subsections.

    This function exists to maintain compatibility with document_generator.
    It splits the section into subsections and calls generate_subsection_with_hybrid_approach
    for each one.
    """

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
