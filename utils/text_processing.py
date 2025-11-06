"""
Text processing utilities for medical document generation.

This module provides functions for parsing and manipulating medical text,
including section extraction, date parsing, and document assembly. It uses
pre-compiled regex patterns for optimal performance.

Key Functions:
    - parse_section_subsections: Split section into subsections by markers
    - parse_medical_record_date: Parse dates in various medical record formats
    - extract_final_section: Extract final section from LLM response
    - assemble_document_from_sections: Assemble final document from section parts

Section Format:
    Documents use these markers:
    - Overskrift: Main section title marker
    - Sub_section: Subsection title marker
    - SUBSECTION_TITLE: Alternative subsection marker

Date Formats:
    Supports multiple Danish medical date formats:
    - "YY.MM.DD HH:MM" (e.g., "24.01.15 14:30")
    - "YY.MM.DD" (e.g., "24.01.15")
    - "YYYY-MM-DD" (ISO format)

Dependencies:
    - utils.text_patterns: Pre-compiled regex patterns
    - utils.exceptions: Custom exception types
    - config.settings: System configuration

Example:
    >>> from utils.text_processing import parse_section_subsections
    >>> subsections = parse_section_subsections(section_text)
    >>> for sub in subsections:
    ...     print(sub['title'], sub['content'])
"""

import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from config.settings import DATE_FORMATS
from .text_patterns import (
    SUBSECTION_PATTERN,
    SUBSECTION_SPLIT_PATTERN,
    normalize_whitespace
)
from .exceptions import (
    TextParsingError,
    DateParsingError,
    InvalidSectionFormatError
)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class Subsection:
    """
    Represents a document subsection with title, content, and optional intro.

    Attributes:
        title: Subsection title/heading
        content: Main content text
        intro: Optional introductory text before subsections
    """
    title: str
    content: str
    intro: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for backward compatibility."""
        result = {
            "title": self.title,
            "content": self.content
        }
        if self.intro:
            result["intro"] = self.intro
        return result

def parse_section_subsections(section_text: str) -> List[Dict[str, str]]:
    """
    Split section text into subsections based on 'Sub_section' markers.

    This function parses medical document sections, extracting subsections
    marked with "Sub_section: Title" format. The introductory text before
    the first subsection is attached to that subsection.

    Args:
        section_text: The text to split into subsections

    Returns:
        List of subsection dictionaries with 'title', 'content', and optionally 'intro'

    Raises:
        InvalidSectionFormatError: If section format is invalid
        TextParsingError: If parsing fails

    Example:
        >>> text = "Intro text\\nSub_section: Diagnosis\\nDiabetes Type 2"
        >>> subsections = parse_section_subsections(text)
        >>> subsections[0]['title']
        'Diagnosis'
    """
    # Type safety: ensure section_text is a string
    if not isinstance(section_text, str):
        logger.warning(
            f"parse_section_subsections received non-string input (type: {type(section_text)}). "
            f"Converting to string."
        )
        section_text = str(section_text) if section_text else ""

    # Check if there are any subsections
    if "Sub_section" not in section_text:
        logger.debug("No subsection markers found, returning entire text as single section")
        return [{"title": "Main Content", "content": section_text}]

    try:
        # Split the text by "Sub_section" markers
        parts = SUBSECTION_PATTERN.split(section_text)

        # The first part (before any Sub_section) is the introduction
        intro_text = parts[0].strip()
        logger.debug(f"Found intro text: {len(intro_text)} characters")

        subsections = _parse_subsection_parts(parts, intro_text)

        if not subsections:
            logger.warning("No subsections extracted, returning entire text")
            return [{"title": "Main Content", "content": section_text}]

        logger.debug(f"Successfully parsed {len(subsections)} subsections")
        return subsections

    except Exception as e:
        logger.error(f"Failed to parse subsections: {e}", exc_info=True)
        raise TextParsingError(
            "Failed to parse section subsections",
            details=str(e)
        )


def _parse_subsection_parts(
    parts: List[str],
    intro_text: str
) -> List[Dict[str, str]]:
    """
    Parse split section parts into subsection dictionaries.

    Args:
        parts: Split section parts from regex split
        intro_text: Introductory text before first subsection

    Returns:
        List of subsection dictionaries
    """
    subsections = []

    for i in range(1, len(parts), 2):
        if i >= len(parts) - 1:
            break

        subsection_header = parts[i].strip()  # "Sub_section: Title"
        subsection_content = parts[i + 1].strip()

        # Extract title from "Sub_section: Title"
        title_match = SUBSECTION_SPLIT_PATTERN.match(subsection_header)
        if title_match:
            clean_title = title_match.group(1).strip()
        else:
            clean_title = f"Subsection {i//2 + 1}"
            logger.warning(
                f"Failed to extract subsection title from: '{subsection_header}', "
                f"using default: '{clean_title}'"
            )

        subsection_dict = {
            "title": clean_title,
            "content": subsection_content
        }

        # Add intro only to the first subsection
        if i == 1 and intro_text:
            subsection_dict["intro"] = intro_text

        subsections.append(subsection_dict)

    return subsections


# Backward compatibility alias
split_section_into_subsections = parse_section_subsections

def parse_medical_record_date(date_str: str) -> datetime:
    """
    Parse date string from medical records with multiple format support.

    Attempts to parse dates using all known medical record date formats.
    Returns datetime.min if all parsing attempts fail.

    Args:
        date_str: Date string to parse (e.g., "24.01.15 14:30", "2024-01-15")

    Returns:
        Parsed datetime object, or datetime.min if parsing fails

    Example:
        >>> date = parse_medical_record_date("24.01.15 14:30")
        >>> date.year
        2024
    """
    if not date_str or not isinstance(date_str, str):
        logger.warning(f"Invalid date string provided: {date_str}")
        return datetime.min

    date_str_stripped = date_str.strip()

    for fmt in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_str_stripped, fmt)
            logger.debug(f"Successfully parsed date '{date_str}' using format '{fmt}'")
            return parsed_date
        except ValueError:
            continue

    logger.warning(
        f"Failed to parse date '{date_str}' with any known format. "
        f"Tried {len(DATE_FORMATS)} format(s)."
    )
    return datetime.min


# Backward compatibility alias
parse_date_safe = parse_medical_record_date


def extract_final_section(response: Union[Dict[str, Any], str, Any]) -> str:
    """
    Extract the final improved section from LLM response.

    Handles different response formats from the LLM, including dict responses
    with 'output' key and direct string responses.

    Args:
        response: The response from the LLM containing the final section.
                 Can be dict, str, or other types.

    Returns:
        Extracted section text, or empty string if extraction fails

    Raises:
        TextParsingError: If response type is completely unsupported

    Example:
        >>> response = {"output": "Section text here"}
        >>> text = extract_final_section(response)
        >>> print(text)
        'Section text here'
    """
    if isinstance(response, dict):
        if 'output' in response:
            logger.debug("Extracted section from dict response (key: 'output')")
            return response['output']
        elif 'content' in response:
            logger.debug("Extracted section from dict response (key: 'content')")
            return response['content']
        elif 'text' in response:
            logger.debug("Extracted section from dict response (key: 'text')")
            return response['text']
        else:
            logger.warning(f"Dict response has no recognized content key: {response.keys()}")
            return ""

    elif isinstance(response, str):
        logger.debug("Response is already a string")
        return response

    else:
        logger.error(
            f"Unable to extract final section from response of type: {type(response)}"
        )
        return ""


def assemble_document_from_sections(
    section_outputs: Dict[str, str],
    validation_details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Assemble final document from section outputs.

    Combines multiple section outputs into a single document with proper
    section headers (Overskrift: markers). Section content is assumed to
    already contain subsection markers (SUBSECTION_TITLE:).

    Args:
        section_outputs: Dictionary mapping section titles to section content
        validation_details: Optional validation details (currently unused, for future use)

    Returns:
        Assembled document text with section markers

    Example:
        >>> sections = {
        ...     "Diagnose": "SUBSECTION_TITLE: Primær diagnose\\nDiabetes Type 2",
        ...     "Behandling": "SUBSECTION_TITLE: Medicin\\nMetformin 500mg"
        ... }
        >>> doc = assemble_document_from_sections(sections)
        >>> print(doc)
    """
    logger.info(f"Assembling final document from {len(section_outputs)} sections")

    assembled_parts = []

    for title, text in section_outputs.items():
        # Add section header
        section_part = f"Overskrift: {title}\n\n"

        # Add section content AS-IS (already has SUBSECTION_TITLE: markers)
        section_part += text

        assembled_parts.append(section_part)

    assembled_document = "\n\n".join(assembled_parts)

    logger.info(
        f"Document assembled successfully. "
        f"Total length: {len(assembled_document)} characters"
    )

    return assembled_document


# Backward compatibility alias
assemble_final_document = assemble_document_from_sections
