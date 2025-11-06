"""
Fact-level validation for the fact-by-fact generation approach.

This module provides validation functionality for individual fact answers,
checking them against their source documents for accuracy and completeness.
The validator uses an LLM to critique and optionally correct fact answers.

Example:
    >>> validator = FactValidator()
    >>> validated_answer, was_corrected = validator.validate_fact_answer(
    ...     fact=required_fact,
    ...     answer="Patient has diabetes",
    ...     sources=[source1, source2],
    ...     max_retries=2
    ... )
"""

import logging
from typing import List, Tuple, Dict, Any

from generation.models import RequiredFact, SourceDocument
from generation.constants import (
    FACT_VALIDATION_PROMPT_TEMPLATE,
    UNANSWERABLE_MARKER,
    VALID_MARKER
)
from generation.exceptions import ValidationError, LLMInvocationError
from config.llm_config import llm_config
from utils.error_handling import safe_llm_invoke

# Configure module logger
logger = logging.getLogger(__name__)


class FactValidator:
    """Validates individual fact answers against their source documents.

    The FactValidator uses an LLM to check if fact answers are:
    - Accurate based on the provided sources
    - Properly cited with correct source references
    - Using the most recent information available

    If validation finds issues, the validator can optionally correct
    the answer and return the corrected version.

    Attributes:
        llm: Language model instance for validation (critique model)

    Example:
        >>> validator = FactValidator()
        >>> answer, corrected = validator.validate_fact_answer(
        ...     fact=fact,
        ...     answer="Patient diagnosed with diabetes type 2",
        ...     sources=sources
        ... )
        >>> if corrected:
        ...     logger.info("Answer was corrected during validation")
    """

    def __init__(self):
        """Initialize the fact validator with the critique LLM."""
        self.llm = llm_config.llm_critique
        logger.debug("FactValidator initialized with critique LLM")

    def validate_fact_answer(
        self,
        fact: RequiredFact,
        answer: str,
        sources: List[Dict[str, Any]],
        max_retries: int = 2
    ) -> Tuple[str, bool]:
        """Validate and optionally correct a fact answer.

        This method checks if an answer is accurate based on the provided
        sources. If the answer is correct, it returns the original answer.
        If issues are found, it returns a corrected version.

        Args:
            fact: The RequiredFact that was answered
            answer: The LLM-generated answer to validate
            sources: List of source documents (dicts) used to answer the fact
            max_retries: Maximum number of correction attempts (default: 2)

        Returns:
            Tuple of (validated_answer, was_corrected) where:
                - validated_answer is either the original or corrected answer
                - was_corrected is True if the answer was modified

        Raises:
            ValidationError: If validation fails critically

        Example:
            >>> fact = RequiredFact(
            ...     description="Patient's diabetes diagnosis",
            ...     search_query="diabetes diagnosis"
            ... )
            >>> answer = "Patient has type 2 diabetes"
            >>> validated, corrected = validator.validate_fact_answer(
            ...     fact, answer, sources
            ... )
        """
        # Skip validation for unanswerable facts or empty sources
        if answer == UNANSWERABLE_MARKER or not sources:
            logger.debug(
                f"Skipping validation for unanswerable or sourceless fact: "
                f"{fact.description[:50]}..."
            )
            return answer, False

        logger.debug(f"Validating fact answer: {fact.description[:50]}...")

        try:
            # Format sources for validation prompt
            sources_text = self._format_sources(sources)

            # Build validation prompt
            validation_prompt = FACT_VALIDATION_PROMPT_TEMPLATE.format(
                fact_description=fact.description,
                answer=answer,
                sources_text=sources_text
            )

            # Call LLM for validation
            validation_result = safe_llm_invoke(
                prompt=validation_prompt,
                llm=self.llm,
                max_retries=1,
                operation="fact_validation"
            )

            if not validation_result:
                logger.warning(
                    f"Empty validation result for fact: {fact.description[:50]}..., "
                    "accepting original answer"
                )
                return answer, False

            validation_result = validation_result.strip()

            # Check if validation passed
            if validation_result.upper() == VALID_MARKER:
                logger.info(f"✓ Fact valid: {fact.description[:50]}...")
                return answer, False

            # LLM provided corrected answer
            logger.warning(f"⚠ Fact corrected: {fact.description[:50]}...")
            return validation_result, True

        except Exception as e:
            logger.error(
                f"Validation failed for fact '{fact.description[:50]}...': {e}",
                exc_info=True
            )
            # Return original answer on validation failure
            return answer, False

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format source documents for inclusion in validation prompt.

        Args:
            sources: List of source document dictionaries

        Returns:
            Formatted string with all sources

        Example:
            >>> sources = [
            ...     {"entry_type": "Medical Note", "timestamp": "15.03.2024",
            ...      "full_content": "Patient has diabetes"}
            ... ]
            >>> formatted = validator._format_sources(sources)
            >>> print(formatted)
            Kilde 1: Medical Note (15.03.2024)
            Patient has diabetes
        """
        lines = []

        for i, source in enumerate(sources, 1):
            entry_type = source.get('entry_type', 'Note')
            timestamp = source.get('timestamp', 'Ukendt dato')
            content = source.get('full_content', source.get('snippet', ''))

            lines.append(f"Kilde {i}: {entry_type} ({timestamp})")
            lines.append(f"{content}")
            lines.append("")

        return "\n".join(lines)

    def validate_batch(
        self,
        fact_answers: List[Tuple[RequiredFact, str, List[Dict[str, Any]]]],
        max_retries: int = 2
    ) -> List[Tuple[str, bool]]:
        """Validate multiple fact answers in sequence.

        This is a convenience method for validating multiple facts.
        Note: This performs sequential validation, not parallel.

        Args:
            fact_answers: List of (fact, answer, sources) tuples
            max_retries: Maximum retries per validation

        Returns:
            List of (validated_answer, was_corrected) tuples

        Example:
            >>> fact_answers = [
            ...     (fact1, answer1, sources1),
            ...     (fact2, answer2, sources2)
            ... ]
            >>> results = validator.validate_batch(fact_answers)
            >>> for validated, corrected in results:
            ...     print(f"Corrected: {corrected}")
        """
        results = []

        for fact, answer, sources in fact_answers:
            validated, corrected = self.validate_fact_answer(
                fact=fact,
                answer=answer,
                sources=sources,
                max_retries=max_retries
            )
            results.append((validated, corrected))

        logger.info(
            f"Batch validation complete: {len(fact_answers)} facts validated, "
            f"{sum(1 for _, c in results if c)} corrected"
        )

        return results
