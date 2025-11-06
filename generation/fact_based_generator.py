"""
Fact-by-fact document generation.

This module implements a fact-by-fact approach to generating medical documentation.
Each required fact is answered independently using RAG-retrieved sources,
then all fact answers are assembled into a coherent subsection.

The process follows these steps:
1. Answer each fact independently using its specific sources
2. Validate each answer (optional)
3. Assemble all fact answers into a coherent narrative

Example:
    >>> generator = FactBasedGenerator()
    >>> answer, answerable = generator.answer_single_fact(
    ...     fact=required_fact,
    ...     subsection_title="Current Medications",
    ...     sources=sources
    ... )
    >>> if answerable:
    ...     print(f"Fact answered: {answer}")
"""

import logging
from typing import List, Dict, Tuple, Any

from generation.models import RequiredFact
from generation.constants import (
    FACT_ANSWERING_PROMPT_TEMPLATE,
    FACT_ASSEMBLY_PROMPT_TEMPLATE,
    UNANSWERABLE_MARKER,
    DEFAULT_MAX_WORD_COUNT
)
from generation.exceptions import AssemblyError, LLMInvocationError
from config.llm_config import llm_config
from utils.error_handling import safe_llm_invoke

# Configure module logger
logger = logging.getLogger(__name__)


class FactBasedGenerator:
    """Generates medical documentation using fact-by-fact approach.

    This generator answers each required fact independently, then assembles
    the answers into coherent subsections. This approach provides better
    traceability and more accurate source attribution than monolithic generation.

    The generator uses two LLM calls per subsection:
    1. Answer individual facts (can be batched)
    2. Assemble facts into narrative

    Attributes:
        llm: Language model instance for generation

    Example:
        >>> generator = FactBasedGenerator()
        >>> # Answer individual facts
        >>> answers = []
        >>> for fact in required_facts:
        ...     answer, answerable = generator.answer_single_fact(
        ...         fact=fact,
        ...         subsection_title="Medications",
        ...         sources=fact_sources
        ...     )
        ...     answers.append({"fact": fact, "answer": answer, "answerable": answerable})
        >>>
        >>> # Assemble into subsection
        >>> result = generator.assemble_subsection_from_facts(
        ...     subsection_title="Medications",
        ...     section_title="Treatment Plan",
        ...     section_intro="Document current treatment",
        ...     format_instructions="List each medication",
        ...     fact_answers=answers
        ... )
    """

    def __init__(self):
        """Initialize the fact-based generator with generation LLM."""
        self.llm = llm_config.llm_generate
        logger.debug("FactBasedGenerator initialized with generation LLM")

    def answer_single_fact(
        self,
        fact: RequiredFact,
        subsection_title: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[str, bool]:
        """Answer a single fact using its retrieved sources.

        This method generates an answer to a specific fact question
        based on the provided source documents. The answer includes
        proper source citations.

        Args:
            fact: The RequiredFact to answer
            subsection_title: Title of the subsection being generated
            sources: List of source documents with full_content field

        Returns:
            Tuple of (answer_text, is_answerable) where:
                - answer_text is the generated answer or "UNANSWERABLE"
                - is_answerable indicates if the fact could be answered

        Example:
            >>> fact = RequiredFact(
            ...     description="Patient's current medications",
            ...     search_query="medications current"
            ... )
            >>> sources = [
            ...     {"entry_type": "Medical Note", "timestamp": "15.03.2024",
            ...      "full_content": "Patient takes metformin 1000mg"}
            ... ]
            >>> answer, answerable = generator.answer_single_fact(
            ...     fact, "Medications", sources
            ... )
            >>> print(f"Answer: {answer}")
            Patient takes metformin 1000mg [Kilde: Medical Note - 15.03.2024]
        """
        if not sources:
            logger.debug(f"No sources for fact: {fact.description[:50]}...")
            return UNANSWERABLE_MARKER, False

        logger.debug(f"Answering fact with {len(sources)} sources: {fact.description[:50]}...")

        try:
            # Format sources with FULL content
            sources_text = self._format_sources_full(sources)

            # Build answering prompt
            prompt = FACT_ANSWERING_PROMPT_TEMPLATE.format(
                fact_description=fact.description,
                subsection_title=subsection_title,
                sources_text=sources_text
            )

            # Call LLM to answer the fact
            answer = safe_llm_invoke(
                prompt=prompt,
                llm=self.llm,
                max_retries=2,
                operation="fact_answering"
            )

            # Check if answer is valid
            if not answer or UNANSWERABLE_MARKER in answer.upper():
                logger.debug(f"Fact could not be answered: {fact.description[:50]}...")
                return UNANSWERABLE_MARKER, False

            logger.debug(f"Fact answered: {answer[:60]}...")
            return answer.strip(), True

        except Exception as e:
            logger.error(
                f"Failed to answer fact '{fact.description[:50]}...': {e}",
                exc_info=True
            )
            return UNANSWERABLE_MARKER, False

    def assemble_subsection_from_facts(
        self,
        subsection_title: str,
        section_title: str,
        section_intro: str,
        format_instructions: str,
        fact_answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assemble a coherent subsection from individual fact answers.

        This method takes independently answered facts and combines them
        into a natural, flowing narrative that follows the specified
        formatting instructions.

        Args:
            subsection_title: Title of the subsection
            section_title: Parent section title
            section_intro: General instructions for the entire section
            format_instructions: Specific formatting instructions for this subsection
            fact_answers: List of dicts with keys: 'fact', 'answer', 'answerable', 'sources'

        Returns:
            Dictionary with keys:
                - 'answer': Assembled subsection text
                - 'unanswerable_items': List of fact descriptions that couldn't be answered

        Raises:
            AssemblyError: If assembly fails and no fallback is possible

        Example:
            >>> fact_answers = [
            ...     {"fact": fact1, "answer": "Patient has diabetes", "answerable": True},
            ...     {"fact": fact2, "answer": "UNANSWERABLE", "answerable": False}
            ... ]
            >>> result = generator.assemble_subsection_from_facts(
            ...     subsection_title="Diagnosis",
            ...     section_title="Medical History",
            ...     section_intro="Document medical conditions",
            ...     format_instructions="List diagnoses chronologically",
            ...     fact_answers=fact_answers
            ... )
            >>> print(result['answer'])
            Patient has diabetes [Kilde: Medical Note - 15.03.2024]
            >>> print(result['unanswerable_items'])
            ['Medication allergies']
        """
        logger.debug(
            f"Assembling subsection '{subsection_title}' from "
            f"{len(fact_answers)} fact answers"
        )

        # Separate answerable and unanswerable facts
        answerable = [fa for fa in fact_answers if fa.get('answerable', False)]
        unanswerable = [
            fa['fact'].description
            for fa in fact_answers
            if not fa.get('answerable', False)
        ]

        if not answerable:
            # Nothing could be answered
            logger.warning(
                f"No facts could be answered for subsection '{subsection_title}'"
            )
            return {
                'answer': '',
                'unanswerable_items': unanswerable
            }

        # Format the individual fact answers as bullet points
        fact_texts = []
        for fa in answerable:
            answer_text = fa.get('answer', '')
            if answer_text and answer_text != UNANSWERABLE_MARKER:
                fact_texts.append(f"• {answer_text}")

        combined_facts = "\n".join(fact_texts)
        logger.debug(f"Combined {len(fact_texts)} answerable facts")

        # Build assembly prompt
        assembly_prompt = FACT_ASSEMBLY_PROMPT_TEMPLATE.format(
            subsection_title=subsection_title,
            section_title=section_title,
            section_intro=section_intro,
            format_instructions=format_instructions or "Ingen yderligere specifikke instruktioner",
            combined_facts=combined_facts
        )

        try:
            # Call LLM to assemble the subsection
            assembled_text = safe_llm_invoke(
                prompt=assembly_prompt,
                llm=self.llm,
                max_retries=2,
                operation="fact_assembly"
            )

            if not assembled_text:
                logger.warning(
                    f"Assembly returned empty text for '{subsection_title}', "
                    "using simple concatenation"
                )
                # Fallback: just concatenate the answers
                assembled_text = "\n\n".join([fa['answer'] for fa in answerable])

            logger.info(
                f"Subsection assembled: {len(assembled_text)} chars, "
                f"{len(unanswerable)} unanswerable"
            )

            return {
                'answer': assembled_text.strip(),
                'unanswerable_items': unanswerable
            }

        except Exception as e:
            logger.error(
                f"Assembly failed for '{subsection_title}': {e}, "
                "using simple concatenation",
                exc_info=True
            )
            # Fallback: simple concatenation
            return {
                'answer': "\n\n".join([fa['answer'] for fa in answerable]),
                'unanswerable_items': unanswerable
            }

    def _format_sources_full(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources with full content for fact answering prompt.

        Args:
            sources: List of source document dictionaries

        Returns:
            Formatted string with full source content

        Note:
            Uses full_content field, not truncated snippets, to provide
            maximum context for accurate answering.

        Example:
            >>> sources = [
            ...     {"entry_type": "Medical Note", "timestamp": "15.03.2024",
            ...      "full_content": "Patient diagnosed...", "relevance": 95}
            ... ]
            >>> formatted = generator._format_sources_full(sources)
            >>> print(formatted)
            --- KILDE 1: Medical Note (15.03.2024) - Relevans: 95% ---
            Patient diagnosed...
        """
        lines = []

        for i, source in enumerate(sources, 1):
            entry_type = source.get('entry_type', 'Note')
            timestamp = source.get('timestamp', 'Ukendt dato')
            relevance = source.get('relevance', 0)

            # Use FULL content, not truncated snippet
            content = source.get('full_content', source.get('snippet', ''))

            lines.append(
                f"--- KILDE {i}: {entry_type} ({timestamp}) - "
                f"Relevans: {relevance}% ---"
            )
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    def answer_facts_batch(
        self,
        facts_with_sources: List[Tuple[RequiredFact, str, List[Dict[str, Any]]]]
    ) -> List[Tuple[str, bool]]:
        """Answer multiple facts in sequence.

        This is a convenience method for answering multiple facts.
        Note: This performs sequential answering, not parallel.

        Args:
            facts_with_sources: List of (fact, subsection_title, sources) tuples

        Returns:
            List of (answer, is_answerable) tuples

        Example:
            >>> facts_batch = [
            ...     (fact1, "Medications", sources1),
            ...     (fact2, "Allergies", sources2)
            ... ]
            >>> answers = generator.answer_facts_batch(facts_batch)
            >>> for answer, answerable in answers:
            ...     if answerable:
            ...         print(f"Answered: {answer[:50]}...")
        """
        results = []

        for fact, subsection_title, sources in facts_with_sources:
            answer, answerable = self.answer_single_fact(
                fact=fact,
                subsection_title=subsection_title,
                sources=sources
            )
            results.append((answer, answerable))

        answerable_count = sum(1 for _, a in results if a)
        logger.info(
            f"Batch answering complete: {answerable_count}/{len(facts_with_sources)} "
            "facts answered"
        )

        return results
