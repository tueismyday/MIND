"""
Text processing utilities for the Agentic RAG Medical Documentation System.
Handles text manipulation, parsing, and formatting operations.
"""

import re
from typing import Tuple, List
from datetime import datetime
from config.settings import DATE_FORMATS

def split_section_into_subsections(section_text: str) -> List[dict]:
    """
    Splits section text into subsections based on 'Sub_section' markers.

    Args:
        section_text (str): The text to split into subsections.

    Returns:
        List[dict]: List of subsection dictionaries with 'title', 'content', and optionally 'intro'
    """
    # Type safety: ensure section_text is a string
    if not isinstance(section_text, str):
        print(f"[WARNING] split_section_into_subsections received non-string input (type: {type(section_text)}). Converting to string.")
        section_text = str(section_text) if section_text else ""

    # First, check if there are any subsections
    if "Sub_section" not in section_text:
        # No subsections found, return the entire section as one
        return [{"title": "Main Content", "content": section_text}]

    # Split the text by "Sub_section" markers
    parts = re.split(r'(Sub_section:\s*[^\n]+)', section_text)

    # The first part (before any Sub_section) is the introduction
    intro_text = parts[0].strip()

    subsections = []
    for i in range(1, len(parts), 2):
        if i < len(parts) - 1:
            subsection_header = parts[i].strip()  # "Sub_section: Title"
            subsection_content = parts[i + 1].strip()

            # Extract title from "Sub_section: Title"
            title_match = re.match(r'Sub_section:\s*(.+)', subsection_header)
            if title_match:
                clean_title = title_match.group(1).strip()
            else:
                clean_title = f"Subsection {i//2 + 1}"

            subsection_dict = {
                "title": clean_title,
                "content": subsection_content
            }

            # Add intro only to the first subsection
            if i == 1 and intro_text:
                subsection_dict["intro"] = intro_text

            subsections.append(subsection_dict)

    if not subsections:
        return [{"title": "Main Content", "content": section_text}]

    return subsections

def parse_date_safe(date_str: str) -> datetime:
    """
    Safely parse a date string with multiple format attempts.
    
    Args:
        date_str (str): The date string to parse.
        
    Returns:
        datetime: Parsed datetime object, or datetime.min if parsing fails.
    """
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.min

def extract_final_section(response) -> str:
    """
    Extracts the final improved section from the response object.
    Handles different response formats from the LLM.
    
    Args:
        response: The response from the LLM containing the final section.
        
    Returns:
        str: The extracted section text.
    """
    if isinstance(response, dict) and 'output' in response:
        return response['output']
    elif isinstance(response, str):
        return response
    else:
        print("[ERROR] Unable to extract final section from response.")
        return ""

def assemble_final_document(section_outputs: dict, validation_details: dict = None) -> str:
    """Assemble final document from section outputs"""
    
    print("[RESULT] Final document has been created from guidelines!")
    
    assembled_parts = []
    
    for title, text in section_outputs.items():
        # Add section header
        section_part = f"Overskrift: {title}\n\n"
        
        # Add section content AS-IS (already has SUBSECTION_TITLE: markers)
        section_part += text
        
        assembled_parts.append(section_part)
    
    return "\n\n".join(assembled_parts)