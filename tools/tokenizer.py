"""
Medical text tokenization for Danish clinical documents.

This module provides specialized tokenization for Danish medical text,
preserving medical terminology, dosages, measurements, and clinical
abbreviations while filtering common stopwords.
"""

import re
import logging
from typing import List, Set, Pattern
from tools.constants import DANISH_STOPWORDS, MEDICAL_TERM_MIN_LENGTH


logger = logging.getLogger(__name__)


class MedicalTextTokenizer:
    """
    Tokenizer optimized for Danish medical text.

    This tokenizer preserves important medical patterns such as dosages,
    measurements, and clinical abbreviations while filtering Danish stopwords
    for improved keyword search relevance.

    Features:
        - Preserves dosages (e.g., "500mg", "2.5ml")
        - Preserves measurements (e.g., "37.5°C", "120mm")
        - Preserves medical units (mg, g, ml, l, %, mm, cm, kg)
        - Preserves temporal expressions (år, måned, dag, timer, min)
        - Filters Danish stopwords
        - Keeps numbers and tokens containing numbers
        - Case-insensitive processing

    Example:
        >>> tokenizer = MedicalTextTokenizer()
        >>> text = "Patient fik 500mg paracetamol og 2.5l væske"
        >>> tokens = tokenizer.tokenize(text)
        >>> print(tokens)
        ['patient', 'fik', '500mg', 'paracetamol', '2.5l', 'væske']
    """

    def __init__(
        self,
        stopwords: Set[str] = None,
        min_term_length: int = MEDICAL_TERM_MIN_LENGTH,
        preserve_numbers: bool = True
    ):
        """
        Initialize medical text tokenizer.

        Args:
            stopwords: Set of stopwords to filter (defaults to Danish stopwords)
            min_term_length: Minimum length to keep a term even if it's a stopword
            preserve_numbers: Whether to keep numeric tokens
        """
        self.stopwords = stopwords if stopwords is not None else DANISH_STOPWORDS
        self.min_term_length = min_term_length
        self.preserve_numbers = preserve_numbers

        # Pre-compile regex pattern for better performance
        self._token_pattern = self._build_token_pattern()

        logger.debug(
            f"Initialized MedicalTextTokenizer with {len(self.stopwords)} stopwords, "
            f"min_term_length={min_term_length}"
        )

    def _build_token_pattern(self) -> Pattern:
        """
        Build compiled regex pattern for tokenization.

        The pattern matches:
        - Numbers with optional decimal points and medical units
        - Medical abbreviations and terms
        - General word tokens

        Returns:
            Compiled regex pattern
        """
        # Pattern explanation:
        # \d+(?:[.,]\d+)? - numbers with optional decimal part (comma or period)
        # (?:mg|g|ml|l|%|mm|cm|kg|år|måned|dag|timer|min)? - optional medical units
        # \b - word boundary
        # \w+ - one or more word characters
        pattern = r'\b(?:\d+(?:[.,]\d+)?(?:mg|g|ml|l|%|mm|cm|kg|år|måned|dag|timer|min)?\b|\w+)\b'
        return re.compile(pattern, re.IGNORECASE)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Danish medical text.

        Converts text to lowercase, extracts tokens using medical-aware regex,
        and filters stopwords while preserving important medical terms.

        Args:
            text: Input text to tokenize

        Returns:
            List of filtered tokens

        Example:
            >>> tokenizer = MedicalTextTokenizer()
            >>> tokenizer.tokenize("Patienten har diabetes og fik 500mg metformin")
            ['patienten', 'diabetes', 'fik', '500mg', 'metformin']
        """
        if not text:
            return []

        # Convert to lowercase for consistent processing
        text = text.lower()

        # Extract tokens using compiled pattern
        tokens = self._token_pattern.findall(text)

        # Filter tokens based on criteria
        filtered_tokens = self._filter_tokens(tokens)

        logger.debug(f"Tokenized text: {len(tokens)} tokens -> {len(filtered_tokens)} after filtering")

        return filtered_tokens

    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on stopwords and medical term criteria.

        Keeps a token if:
        - It's not a stopword, OR
        - It's longer than min_term_length (likely a medical term), OR
        - It's a number or contains numbers, OR
        - It contains medical units

        Args:
            tokens: List of raw tokens

        Returns:
            List of filtered tokens
        """
        filtered = []

        for token in tokens:
            # Keep medical/clinical terms even if they might be stopwords
            if self._should_keep_token(token):
                filtered.append(token)

        return filtered

    def _should_keep_token(self, token: str) -> bool:
        """
        Determine if a token should be kept.

        Args:
            token: Token to evaluate

        Returns:
            True if token should be kept, False otherwise
        """
        # Keep if not a stopword
        if token not in self.stopwords:
            return True

        # Keep if longer than minimum length (likely important term)
        if len(token) > self.min_term_length:
            return True

        # Keep if it's a number
        if self.preserve_numbers and token.isdigit():
            return True

        # Keep if it contains numbers (e.g., "37.5", "500mg")
        if self.preserve_numbers and any(char.isdigit() for char in token):
            return True

        return False

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize multiple texts in batch.

        Args:
            texts: List of text strings to tokenize

        Returns:
            List of token lists, one for each input text

        Example:
            >>> tokenizer = MedicalTextTokenizer()
            >>> texts = ["Patient fik insulin", "Blodsukker var 8.5"]
            >>> tokenizer.tokenize_batch(texts)
            [['patient', 'fik', 'insulin'], ['blodsukker', '8.5']]
        """
        return [self.tokenize(text) for text in texts]

    def add_stopwords(self, words: Set[str]) -> None:
        """
        Add additional stopwords to the filter list.

        Args:
            words: Set of stopwords to add
        """
        self.stopwords.update(words)
        logger.debug(f"Added {len(words)} stopwords, total: {len(self.stopwords)}")

    def remove_stopwords(self, words: Set[str]) -> None:
        """
        Remove stopwords from the filter list.

        Args:
            words: Set of stopwords to remove
        """
        self.stopwords.difference_update(words)
        logger.debug(f"Removed {len(words)} stopwords, total: {len(self.stopwords)}")

    def get_stats(self) -> dict:
        """
        Get tokenizer statistics.

        Returns:
            Dictionary with tokenizer configuration stats
        """
        return {
            "num_stopwords": len(self.stopwords),
            "min_term_length": self.min_term_length,
            "preserve_numbers": self.preserve_numbers
        }


# Convenience function for simple tokenization
def tokenize_danish_medical(text: str) -> List[str]:
    """
    Convenience function for tokenizing Danish medical text.

    Creates a default MedicalTextTokenizer and tokenizes the input text.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens

    Example:
        >>> tokens = tokenize_danish_medical("Patient fik 500mg paracetamol")
        >>> print(tokens)
        ['patient', 'fik', '500mg', 'paracetamol']
    """
    tokenizer = MedicalTextTokenizer()
    return tokenizer.tokenize(text)
