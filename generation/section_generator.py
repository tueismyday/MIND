"""
Section and subsection generation with fact-by-fact approach.

This module orchestrates the complete process of generating document sections
using a fact-by-fact approach with batched LLM calls for performance optimization.

The generation pipeline consists of three main phases:
1. Parse guidelines to identify required facts (GuidelineFactParser)
2. Retrieve sources and answer all facts (batched for performance)
3. Validate answers and assemble into coherent subsections

The module supports:
- Batch LLM processing for improved throughput
- Optional two-stage validation
- Note type filtering for focused retrieval
- Graceful fallback handling

Example:
    >>> output, sources, validation = generate_subsection_with_hybrid_approach(
    ...     section_title="Medical History",
    ...     subsection_title="Chronic Conditions",
    ...     section_intro="Document patient's medical history",
    ...     subsection_guidelines="List all chronic conditions...",
    ...     patient_data=None,
    ...     max_sources_per_fact=5,
    ...     enable_validation=True
    ... )
    >>> print(f"Generated: {len(output)} chars")
"""

import logging
import time
from typing import Tuple, List, Dict, Any, Optional

from generation.fact_parser import GuidelineFactParser
from generation.fact_based_generator import FactBasedGenerator
from generation.fact_validator import FactValidator
from generation.models import RequiredFact, ValidationStatistics
from generation.constants import (
    SUBSECTION_TITLE_FORMAT,
    UNANSWERABLE_SECTION_FORMAT,
    UNANSWERABLE_MARKER,
    VALID_MARKER,
    DEFAULT_MAX_WORD_COUNT
)
from generation.exceptions import GenerationError
from utils.text_processing import split_section_into_subsections
from utils.error_handling import safe_rag_search
from utils.profiling import profile
from config.settings import (
    DEFAULT_VALIDATION_CYCLES,
    MAX_VALIDATION_CYCLES,
    MIN_VALIDATION_CYCLES,
    MAX_SOURCES_PER_FACT
)

# Configure module logger
logger = logging.getLogger(__name__)


def _batch_llm_calls(
    prompts: List[str],
    llm_instance: Any,
    operation: str = "batch",
    max_retries: int = 2
) -> List[str]:
    """Batch multiple LLM calls into one request for better performance.

    This function attempts to process multiple prompts in a single batch
    request to the LLM. If batching fails, it falls back to sequential
    processing to ensure reliability.

    Args:
        prompts: List of prompt strings to process
        llm_instance: LLM instance with invoke() or client.chat interface
        operation: Operation name for logging purposes
        max_retries: Maximum retry attempts for batch call

    Returns:
        List of response strings (same length as prompts)

    Note:
        Falls back to sequential processing if batching fails after retries.
        Returns "UNANSWERABLE" for any individual failures in fallback mode.

    Example:
        >>> prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        >>> responses = _batch_llm_calls(prompts, llm_instance, "answering")
        >>> print(f"Processed {len(responses)} prompts")
    """
    if not prompts:
        logger.debug("Empty prompts list, returning empty results")
        return []

    logger.debug(f"Batching {len(prompts)} {operation} prompts")

    # Try batched call with retries
    for attempt in range(max_retries):
        try:
            responses = []

            # Check if LLM has client.chat interface (vLLM style)
            if hasattr(llm_instance, 'client') and hasattr(llm_instance.client, 'chat'):
                for prompt in prompts:
                    response = llm_instance.client.chat.completions.create(
                        model=llm_instance.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=getattr(llm_instance, 'temperature', 0.1),
                        max_tokens=6056,
                    )
                    responses.append(response.choices[0].message.content.strip())

                logger.debug(f"Batch {operation} completed successfully")
                return responses

            else:
                # Fallback: sequential calls using invoke
                logger.debug(f"Using sequential invoke for {operation}")
                for prompt in prompts:
                    response = llm_instance.invoke(prompt)
                    responses.append(response.strip())

                return responses

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Batch {operation} attempt {attempt + 1} failed: {e}, "
                    f"retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Batch {operation} failed after {max_retries} attempts, "
                    "using sequential fallback"
                )
                # Final fallback: sequential with error handling
                responses = []
                for prompt in prompts:
                    try:
                        response = llm_instance.invoke(prompt)
                        responses.append(response.strip())
                    except Exception as e:
                        logger.error(f"Individual prompt failed: {e}")
                        responses.append(UNANSWERABLE_MARKER)
                return responses

    # Should never reach here, but return safe fallback
    logger.error(f"Unexpected state in batch processing, returning UNANSWERABLE")
    return [UNANSWERABLE_MARKER] * len(prompts)


@profile
def generate_subsection_with_hybrid_approach(
    section_title: str,
    subsection_title: str,
    section_intro: str,
    subsection_guidelines: str,
    patient_data: Optional[str],
    max_sources_per_fact: int = MAX_SOURCES_PER_FACT,
    enable_validation: bool = True,
    max_revision_cycles: Optional[int] = None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Generate a subsection using fact-by-fact approach with batched LLM calls.

    This is the main function for generating a single subsection. It implements
    a three-phase pipeline:
    1. Parse guidelines to identify required facts
    2. Retrieve sources and answer all facts (with batching)
    3. Optionally validate answers (with batching)
    4. Assemble facts into coherent narrative

    Args:
        section_title: Parent section title for context
        subsection_title: Title of the subsection to generate
        section_intro: Section introduction/instructions for context
        subsection_guidelines: Specific guidelines for this subsection
        patient_data: Patient data (kept for compatibility, unused in new approach)
        max_sources_per_fact: Maximum RAG sources to retrieve per fact
        enable_validation: Whether to enable fact-level validation
        max_revision_cycles: Maximum validation cycles (default: from settings)

    Returns:
        Tuple of (generated_text, sources_list, validation_details) where:
            - generated_text: Complete subsection text with title
            - sources_list: List of all source documents used
            - validation_details: Dictionary with validation statistics

    Raises:
        GenerationError: If generation fails critically

    Example:
        >>> output, sources, validation = generate_subsection_with_hybrid_approach(
        ...     section_title="Treatment Plan",
        ...     subsection_title="Current Medications",
        ...     section_intro="Document current treatment",
        ...     subsection_guidelines="List all medications with dosages",
        ...     patient_data=None,
        ...     max_sources_per_fact=5,
        ...     enable_validation=True
        ... )
        >>> print(f"Generated: {len(output)} chars")
        >>> print(f"Sources used: {len(sources)}")
        >>> print(f"Validated: {validation['stats']['validated_facts']} facts")
    """
    # Set and validate cycles configuration
    if max_revision_cycles is None:
        max_revision_cycles = DEFAULT_VALIDATION_CYCLES

    max_revision_cycles = max(
        MIN_VALIDATION_CYCLES,
        min(max_revision_cycles, MAX_VALIDATION_CYCLES)
    )

    logger.info(
        f"=== Fact-by-Fact Generation: '{subsection_title}' ==="
    )
    logger.info(
        f"Config: Validation={enable_validation}, Max cycles={max_revision_cycles}"
    )

    try:
        # ================================================================
        # PHASE 1: Parse Guidelines → Identify Required Facts
        # ================================================================

        logger.info("PHASE 1: Parsing guidelines to identify required facts")

        parser = GuidelineFactParser()
        requirements = parser.parse_subsection_requirements(
            section_title=section_title,
            subsection_title=subsection_title,
            subsection_guidelines=subsection_guidelines
        )

        logger.info(
            f"PHASE 1 COMPLETE: Identified {len(requirements.required_facts)} facts"
        )

        # ================================================================
        # PHASE 2: Answer Each Fact (with BATCHED LLM calls)
        # ================================================================

        logger.info(f"PHASE 2: Processing {len(requirements.required_facts)} facts")

        generator = FactBasedGenerator()
        validator = FactValidator() if enable_validation else None

        fact_sources_pairs = []  # (fact, sources) tuples
        all_sources = []
        validation_stats = ValidationStatistics(
            total_facts=len(requirements.required_facts)
        )

        # Step 2a: RAG retrieval for all facts
        logger.info("PHASE 2A: Retrieving sources for all facts")

        for i, fact in enumerate(requirements.required_facts, 1):
            logger.debug(f"Fact {i}/{len(requirements.required_facts)}: {fact.description[:60]}...")

            note_types = fact.note_types or requirements.note_types

            if note_types:
                logger.debug(f"  Filtering by note types: {note_types}")

            rag_result = safe_rag_search(
                query=fact.search_query,
                max_references=max_sources_per_fact,
                note_types=note_types
            )

            sources = rag_result.sources
            logger.debug(f"  Retrieved {len(sources)} sources")

            fact_sources_pairs.append((fact, sources))
            all_sources.extend(sources)

        logger.info("PHASE 2A COMPLETE: Retrieved sources for all facts")

        # Step 2b: BATCH answer all facts
        logger.info(f"PHASE 2B: Batching LLM calls to answer {len(fact_sources_pairs)} facts")

        answering_prompts = []
        answerable_indices = []  # Track which facts have sources

        for idx, (fact, sources) in enumerate(fact_sources_pairs):
            if not sources:
                answering_prompts.append(None)
                continue

            sources_text = generator._format_sources_full(sources)

            # Format the prompt template with actual values
            from generation.constants import FACT_ANSWERING_PROMPT_TEMPLATE
            prompt = FACT_ANSWERING_PROMPT_TEMPLATE.format(
                fact_description=fact.description,
                subsection_title=subsection_title,
                sources_text=sources_text
            )
            answering_prompts.append(prompt)
            answerable_indices.append(idx)

        # Batch call for answering
        valid_prompts = [p for p in answering_prompts if p is not None]

        if valid_prompts:
            batch_answers = _batch_llm_calls(
                valid_prompts, generator.llm, operation="fact_answering"
            )
        else:
            batch_answers = []

        # Map answers back to facts
        fact_answers_raw = []
        batch_idx = 0

        for idx, (fact, sources) in enumerate(fact_sources_pairs):
            if idx in answerable_indices:
                answer = batch_answers[batch_idx]
                batch_idx += 1

                is_answerable = answer and UNANSWERABLE_MARKER not in answer.upper()

                if is_answerable:
                    logger.debug(f"  [{idx+1}] Answered: {answer[:60]}...")
                    validation_stats.answered_facts += 1
                else:
                    logger.debug(f"  [{idx+1}] Could not answer from sources")
                    validation_stats.unanswered_facts += 1

                fact_answers_raw.append((fact, sources, answer, is_answerable))
            else:
                logger.debug(f"  [{idx+1}] No sources found")
                validation_stats.unanswered_facts += 1
                fact_answers_raw.append((fact, sources, UNANSWERABLE_MARKER, False))

        logger.info(
            f"PHASE 2B COMPLETE: Answered {validation_stats.answered_facts}/"
            f"{validation_stats.total_facts} facts"
        )

        # Step 2c: BATCH validate all facts (if enabled)
        if enable_validation and validator:
            logger.info(f"PHASE 2C: Batching validation for {len(fact_answers_raw)} facts")

            # Build validation prompts
            validation_prompts = []
            validatable_indices = []

            for idx, (fact, sources, answer, is_answerable) in enumerate(fact_answers_raw):
                if answer == UNANSWERABLE_MARKER or not sources or not is_answerable:
                    validation_prompts.append(None)
                    continue

                # Format sources for validation
                sources_text = validator._format_sources(sources)

                # Format the validation prompt template
                from generation.constants import FACT_VALIDATION_PROMPT_TEMPLATE
                prompt = FACT_VALIDATION_PROMPT_TEMPLATE.format(
                    fact_description=fact.description,
                    answer=answer,
                    sources_text=sources_text
                )
                validation_prompts.append(prompt)
                validatable_indices.append(idx)

            # Batch validation calls
            valid_validation_prompts = [p for p in validation_prompts if p is not None]

            if valid_validation_prompts:
                batch_validations = _batch_llm_calls(
                    valid_validation_prompts,
                    validator.llm,
                    operation="fact_validation",
                    max_retries=1
                )
            else:
                batch_validations = []

            # Apply validation results
            fact_answers_final = []
            validation_idx = 0

            for idx, (fact, sources, answer, is_answerable) in enumerate(fact_answers_raw):
                if idx in validatable_indices:
                    validation_result = batch_validations[validation_idx]
                    validation_idx += 1

                    validation_stats.validated_facts += 1

                    if validation_result.strip().upper() == VALID_MARKER:
                        logger.debug(f"  Fact valid: {fact.description[:50]}")
                        fact_answers_final.append((fact, sources, answer, False))
                    else:
                        logger.info(f"  Fact corrected: {fact.description[:50]}")
                        validation_stats.corrected_facts += 1
                        fact_answers_final.append((fact, sources, validation_result, True))
                else:
                    fact_answers_final.append((fact, sources, answer, False))

            logger.info(
                f"PHASE 2C COMPLETE: Validated {validation_stats.validated_facts} facts, "
                f"corrected {validation_stats.corrected_facts}"
            )
        else:
            fact_answers_final = [(f, s, a, False) for f, s, a, _ in fact_answers_raw]

        # Build fact_answers structure for assembly
        fact_answers = []
        for fact, sources, answer, was_corrected in fact_answers_final:
            is_answerable = answer and UNANSWERABLE_MARKER not in answer.upper()

            fact_answers.append({
                'fact': fact,
                'answer': answer if is_answerable else UNANSWERABLE_MARKER,
                'sources': sources,
                'answerable': is_answerable,
                'validated': enable_validation,
                'corrected': was_corrected
            })

        logger.info(
            f"PHASE 2 COMPLETE: Answered {validation_stats.answered_facts}/"
            f"{validation_stats.total_facts} facts"
        )
        if enable_validation:
            logger.info(
                f"  Validated: {validation_stats.validated_facts}, "
                f"Corrected: {validation_stats.corrected_facts}"
            )

        # ================================================================
        # PHASE 3: Assemble Subsection from Fact-Answers
        # ================================================================

        logger.info("PHASE 3: Assembling subsection from fact answers")

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
        final_output = SUBSECTION_TITLE_FORMAT.format(title=subsection_title)

        if assembled_text:
            final_output += assembled_text

        if unanswerable_items:
            final_output += UNANSWERABLE_SECTION_FORMAT
            for item in unanswerable_items:
                final_output += f"{item}\n"

        logger.info(f"PHASE 3 COMPLETE: Subsection assembled ({len(assembled_text)} chars)")

        # ================================================================
        # Return Results
        # ================================================================

        # Deduplicate sources
        unique_sources = _deduplicate_sources(all_sources)

        validation_details = {
            'enabled': enable_validation,
            'stats': validation_stats.to_dict(),
            'max_cycles': max_revision_cycles if enable_validation else 0
        }

        logger.info(f"Generation complete for '{subsection_title}'")

        return final_output, unique_sources, validation_details

    except Exception as e:
        logger.error(
            f"Fact-by-fact generation failed for '{subsection_title}': {e}",
            exc_info=True
        )

        # Return minimal fallback
        fallback = (
            f"{SUBSECTION_TITLE_FORMAT.format(title=subsection_title)}"
            f"[Kunne ikke generere indhold for {subsection_title}]"
        )
        return fallback, [], {'enabled': False, 'stats': {}, 'error': str(e)}


def _deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate sources based on timestamp + entry_type.

    Args:
        sources: List of source document dictionaries

    Returns:
        List of unique source documents

    Example:
        >>> sources = [
        ...     {"timestamp": "15.03.2024", "entry_type": "Medical Note", "content": "..."},
        ...     {"timestamp": "15.03.2024", "entry_type": "Medical Note", "content": "..."},
        ...     {"timestamp": "16.03.2024", "entry_type": "Nursing Note", "content": "..."}
        ... ]
        >>> unique = _deduplicate_sources(sources)
        >>> print(len(unique))
        2
    """
    seen = set()
    unique = []

    for source in sources:
        key = (source.get('timestamp', ''), source.get('entry_type', ''))
        if key not in seen:
            seen.add(key)
            unique.append(source)

    logger.debug(f"Deduplicated sources: {len(sources)} → {len(unique)}")
    return unique


def generate_section_with_hybrid_approach(
    section_title: str,
    section_guidelines: str,
    patient_data: Optional[str],
    max_sources_per_fact: int = MAX_SOURCES_PER_FACT,
    enable_validation: bool = True
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Generate entire section by splitting into subsections.

    This function exists to maintain compatibility with document_generator.
    It splits the section into subsections and calls generate_subsection_with_hybrid_approach
    for each one.

    Args:
        section_title: Title of the section to generate
        section_guidelines: Complete guidelines for the section
        patient_data: Patient data (kept for compatibility, unused)
        max_sources_per_fact: Maximum RAG sources per fact
        enable_validation: Whether to enable validation

    Returns:
        Tuple of (section_text, sources_list, validation_details)

    Example:
        >>> output, sources, validation = generate_section_with_hybrid_approach(
        ...     section_title="Treatment Plan",
        ...     section_guidelines="Document current treatment...",
        ...     patient_data=None,
        ...     max_sources_per_fact=5,
        ...     enable_validation=True
        ... )
    """
    logger.info(f"Generating section: '{section_title}'")

    subsections = split_section_into_subsections(section_guidelines)

    if not subsections:
        # Treat entire section as single subsection
        logger.info("No subsections found, treating entire section as one subsection")
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

    logger.info(f"Generating {len(subsections)} subsections")

    for subsection in subsections:
        # Type safety: ensure subsection is a dict with required keys
        if not isinstance(subsection, dict):
            logger.warning(f"Skipping invalid subsection (not a dict): {type(subsection)}")
            continue

        if 'title' not in subsection or 'content' not in subsection:
            logger.warning(f"Skipping subsection missing required keys: {subsection.keys()}")
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

    logger.info(
        f"Section generation complete: {len(all_subsection_outputs)} subsections, "
        f"{len(unique_sources)} unique sources"
    )

    return final_output, unique_sources, combined_validation
