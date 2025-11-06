"""
Pre-compiled regular expression patterns for text processing.

This module contains all regex patterns used in the MIND system for
text parsing, citation extraction, and document structure analysis.
Pre-compiling patterns improves performance for repeated operations.

Pattern Categories:
    - Document Structure: Section and subsection markers
    - Citations: Source reference patterns
    - Dates: Medical record date formats
    - Content Detection: Warning boxes and status indicators

Example:
    >>> from utils.text_patterns import SUBSECTION_PATTERN
    >>> matches = SUBSECTION_PATTERN.findall(text)
"""

import re
from typing import Pattern


# ============================================================================
# Document Structure Patterns
# ============================================================================

# Section header marker: "Overskrift: Title"
SECTION_HEADER_PATTERN: Pattern = re.compile(
    r'Overskrift:\s*(.+?)(?:\n|$)',
    re.IGNORECASE
)

# Subsection marker: "Sub_section: Title"
SUBSECTION_PATTERN: Pattern = re.compile(
    r'(Sub_section:\s*[^\n]+)',
    re.IGNORECASE
)

# Subsection split pattern with capturing group for title
SUBSECTION_SPLIT_PATTERN: Pattern = re.compile(
    r'Sub_section:\s*(.+)',
    re.IGNORECASE
)

# Alternative subsection marker: "SUBSECTION_TITLE: Title"
SUBSECTION_TITLE_PATTERN: Pattern = re.compile(
    r'SUBSECTION_TITLE:\s*(.+?)(?:\n|$)',
    re.IGNORECASE
)


# ============================================================================
# Citation and Source Patterns
# ============================================================================

# Citation pattern: [Kilde: NoteType - Date]
# Matches formats:
#   - [Kilde: NotType - DD.MM.YYYY]
#   - [Kilde: NotType - DD.MM.YY]
#   - [Kilde: NotType - YYYY-MM-DD HH:MM]
CITATION_PATTERN: Pattern = re.compile(
    r'\[Kilde:\s*([^\]]+?)\s*-\s*(\d{2}\.\d{2}\.(?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)\]'
)

# Simplified source extraction: "Source: text"
SOURCE_REFERENCE_PATTERN: Pattern = re.compile(
    r'(?:Source|Kilde):\s*([^\n\]]+)',
    re.IGNORECASE
)


# ============================================================================
# Date Patterns
# ============================================================================

# Danish date format with time: YY.MM.DD HH:MM
DATE_YMD_HMS_PATTERN: Pattern = re.compile(
    r'\d{2}\.\d{2}\.\d{2}\s+\d{2}:\d{2}'
)

# Danish date format: YY.MM.DD or DD.MM.YYYY
DATE_YMD_PATTERN: Pattern = re.compile(
    r'\d{2}\.\d{2}\.(?:\d{2}|\d{4})'
)

# ISO date format: YYYY-MM-DD
DATE_ISO_PATTERN: Pattern = re.compile(
    r'\d{4}-\d{2}-\d{2}'
)

# ISO datetime format: YYYY-MM-DD HH:MM
DATETIME_ISO_PATTERN: Pattern = re.compile(
    r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}'
)


# ============================================================================
# Content Detection Patterns
# ============================================================================

# Warning box indicator: "Kunne ikke besvares"
WARNING_BOX_PATTERN: Pattern = re.compile(
    r'kunne\s+ikke\s+besvares',
    re.IGNORECASE
)

# Status indicator: "Status X: text" (removed from PDF rendering but kept for compatibility)
STATUS_INDICATOR_PATTERN: Pattern = re.compile(
    r'Status\s+\d+:\s*',
    re.IGNORECASE
)

# Empty line or whitespace-only line
EMPTY_LINE_PATTERN: Pattern = re.compile(
    r'^\s*$'
)

# Multiple consecutive newlines (for normalization)
MULTIPLE_NEWLINES_PATTERN: Pattern = re.compile(
    r'\n{3,}'
)


# ============================================================================
# Danish Medical Text Patterns (for tokenization)
# ============================================================================

# Medical dosage units: mg, ml, g, etc.
MEDICAL_DOSAGE_PATTERN: Pattern = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:mg|ml|g|l|mcg|µg|IU|IE)\b',
    re.IGNORECASE
)

# Medical abbreviations to preserve
MEDICAL_ABBREVIATION_PATTERN: Pattern = re.compile(
    r'\b(?:EKG|MR|CT|RTG|BMI|HbA1c|TSH|CRP|INR|ALAT|eGFR)\b',
    re.IGNORECASE
)

# ICD-10 diagnosis codes
ICD10_CODE_PATTERN: Pattern = re.compile(
    r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b'
)

# ATC medication codes
ATC_CODE_PATTERN: Pattern = re.compile(
    r'\b[A-Z]\d{2}[A-Z]{2}\d{2}\b'
)


# ============================================================================
# Superscript and Formatting Patterns
# ============================================================================

# ReportLab superscript markup
SUPERSCRIPT_PATTERN: Pattern = re.compile(
    r'<super>\[(\d+)\]</super>'
)

# HTML/XML-like tags (for cleaning)
HTML_TAG_PATTERN: Pattern = re.compile(
    r'<[^>]+>'
)


# ============================================================================
# Pattern Validation Functions
# ============================================================================

def is_section_header(line: str) -> bool:
    """
    Check if a line is a section header.

    Args:
        line: Text line to check

    Returns:
        True if line starts with "Overskrift:", False otherwise

    Example:
        >>> is_section_header("Overskrift: Diagnose")
        True
        >>> is_section_header("Regular text")
        False
    """
    return SECTION_HEADER_PATTERN.match(line.strip()) is not None


def is_subsection_header(line: str) -> bool:
    """
    Check if a line is a subsection header.

    Args:
        line: Text line to check

    Returns:
        True if line starts with "Sub_section:" or "SUBSECTION_TITLE:", False otherwise

    Example:
        >>> is_subsection_header("Sub_section: Behandling")
        True
        >>> is_subsection_header("SUBSECTION_TITLE: Medicin")
        True
    """
    line_stripped = line.strip()
    return (SUBSECTION_PATTERN.match(line_stripped) is not None or
            SUBSECTION_TITLE_PATTERN.match(line_stripped) is not None)


def contains_citation(text: str) -> bool:
    """
    Check if text contains citations.

    Args:
        text: Text to check

    Returns:
        True if text contains citation markers, False otherwise

    Example:
        >>> contains_citation("[Kilde: Laboratoriesvar - 01.01.2024]")
        True
    """
    return CITATION_PATTERN.search(text) is not None


def contains_warning(text: str) -> bool:
    """
    Check if text contains warning indicators.

    Args:
        text: Text to check

    Returns:
        True if text contains "Kunne ikke besvares", False otherwise

    Example:
        >>> contains_warning("Kunne ikke besvares ud fra journalen")
        True
    """
    return WARNING_BOX_PATTERN.search(text) is not None


def extract_section_title(line: str) -> str:
    """
    Extract title from section header.

    Args:
        line: Section header line

    Returns:
        Extracted title, or empty string if not a valid section header

    Example:
        >>> extract_section_title("Overskrift: Diagnose")
        'Diagnose'
    """
    match = SECTION_HEADER_PATTERN.match(line.strip())
    return match.group(1).strip() if match else ""


def extract_subsection_title(line: str) -> str:
    """
    Extract title from subsection header.

    Args:
        line: Subsection header line

    Returns:
        Extracted title, or empty string if not a valid subsection header

    Example:
        >>> extract_subsection_title("Sub_section: Behandling")
        'Behandling'
    """
    line_stripped = line.strip()

    # Try Sub_section pattern first
    match = SUBSECTION_SPLIT_PATTERN.match(line_stripped)
    if match:
        return match.group(1).strip()

    # Try SUBSECTION_TITLE pattern
    match = SUBSECTION_TITLE_PATTERN.match(line_stripped)
    if match:
        return match.group(1).strip()

    return ""


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Replaces multiple consecutive newlines with double newlines
    and removes trailing whitespace from lines.

    Args:
        text: Text to normalize

    Returns:
        Normalized text

    Example:
        >>> normalize_whitespace("Line1\\n\\n\\n\\nLine2")
        'Line1\\n\\nLine2'
    """
    # Replace multiple newlines with double newlines
    text = MULTIPLE_NEWLINES_PATTERN.sub('\n\n', text)
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines)
