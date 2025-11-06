"""
Enhanced PDF generation with beautiful formatting and footnoted citations.

This module provides professional PDF generation for medical documents using
ReportLab. It handles citation extraction, formatting, and document assembly
with customizable styling.

Key Classes:
    - CitationExtractor: Extracts and manages citations from text
    - EnhancedPDFBuilder: Main PDF builder with formatting capabilities

Features:
    - Automatic citation numbering and footnoting
    - Beautiful visual hierarchy with custom styles
    - Section and subsection formatting
    - Warning box rendering
    - Source citation management

Dependencies:
    - reportlab: PDF generation library
    - utils.pdf_styles: Styling configuration
    - utils.text_patterns: Pre-compiled regex patterns
    - utils.exceptions: Custom exception types
    - config.settings: System configuration

Example:
    >>> from utils.pdf_utils import save_to_pdf
    >>> document_text = "Overskrift: Diagnose\\n\\nContent here"
    >>> save_to_pdf(document_text, "patient_report.pdf")
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, KeepTogether
)

from config.settings import GENERATED_DOCS_DIR
from .pdf_styles import (
    StyleFactory, PDFFonts, COLORS, LAYOUT
)
from .text_patterns import (
    CITATION_PATTERN, WARNING_BOX_PATTERN
)
from .exceptions import (
    PDFGenerationError, PDFRenderingError
)

# Configure module logger
logger = logging.getLogger(__name__)

class CitationExtractor:
    """
    Extracts and manages citations from medical document text.

    This class handles extraction of source citations from text,
    assigns unique numbers, and manages citation deduplication.

    Citation Format:
        [Kilde: NoteType - DD.MM.YYYY] or [Kilde: NoteType - YYYY-MM-DD HH:MM]

    Example:
        >>> extractor = CitationExtractor()
        >>> text = "Patient data [Kilde: Laboratoriesvar - 01.01.2024]"
        >>> clean_text, sources = extractor.extract_citations(text)
        >>> print(clean_text)
        'Patient data <super>[1]</super>'
    """

    def __init__(self):
        """Initialize citation extractor with empty state."""
        self.citations: List[Tuple[str, Dict]] = []  # List of (citation_text, source_dict)
        self.citation_map: Dict[str, int] = {}  # citation_key -> number
        logger.debug("CitationExtractor initialized")

    def extract_citations(self, text: str) -> Tuple[str, List[Dict[str, any]]]:
        """
        Extract citations from text and replace with superscript numbers.

        Args:
            text: Text containing citation markers

        Returns:
            Tuple of (cleaned_text, list_of_source_dicts)

        Example:
            >>> extractor = CitationExtractor()
            >>> clean, sources = extractor.extract_citations(text_with_citations)
            >>> len(sources)
            3
        """
        # Find all citation matches using pre-compiled pattern
        matches = list(CITATION_PATTERN.finditer(text))

        if not matches:
            logger.debug("No citations found in text")
            return text, []

        logger.debug(f"Found {len(matches)} citation(s) in text")

        # Process matches in reverse to maintain string positions
        cleaned_text = text
        sources = []

        for match in reversed(matches):
            note_type = match.group(1).strip()
            timestamp = match.group(2).strip()

            # Create unique citation key
            citation_key = f"{note_type}|{timestamp}"

            # Get or assign citation number
            if citation_key not in self.citation_map:
                self.citation_map[citation_key] = len(self.citation_map) + 1
                sources.insert(0, {
                    'number': self.citation_map[citation_key],
                    'note_type': note_type,
                    'timestamp': timestamp
                })
                logger.debug(
                    f"New citation #{self.citation_map[citation_key]}: "
                    f"{note_type} - {timestamp}"
                )

            citation_num = self.citation_map[citation_key]

            # Replace with superscript reference
            superscript = f'<super>[{citation_num}]</super>'
            cleaned_text = (
                cleaned_text[:match.start()] +
                superscript +
                cleaned_text[match.end():]
            )

        # Sort sources by number
        sources.sort(key=lambda x: x['number'])

        return cleaned_text, sources

    def reset_for_new_section(self) -> None:
        """
        Reset citation counter for new major section.

        Call this when starting a new major section to reset
        citation numbering.
        """
        logger.debug("Resetting citation counter for new section")
        self.citations = []
        self.citation_map = {}


class EnhancedPDFBuilder:
    """
    Enhanced PDF builder with beautiful formatting and footnoted citations.

    This class orchestrates PDF generation with automatic citation handling,
    section formatting, and customizable styling.

    Attributes:
        output_path: Path to output PDF file
        doc: ReportLab SimpleDocTemplate instance
        story: List of PDF flowables (content elements)
        citation_extractor: Citation extraction and numbering manager
        pending_sources: Sources waiting to be rendered as footnotes

    Example:
        >>> builder = EnhancedPDFBuilder("output.pdf")
        >>> builder.add_document_header("Care Plan")
        >>> builder.add_section_heading("Diagnosis")
        >>> builder.add_body_text("Patient has Type 2 Diabetes")
        >>> builder.build()
    """

    def __init__(self, output_path: str):
        """
        Initialize PDF builder.

        Args:
            output_path: Path where PDF will be saved
        """
        self.output_path = Path(output_path)
        logger.info(f"Initializing PDF builder for output: {self.output_path}")

        # Load fonts
        try:
            PDFFonts.load_custom_font()
        except Exception as e:
            logger.warning(f"Font loading failed, using defaults: {e}")

        # Create document with margins
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=LAYOUT.LEFT_MARGIN_CM * cm,
            rightMargin=LAYOUT.RIGHT_MARGIN_CM * cm,
            topMargin=LAYOUT.TOP_MARGIN_CM * cm,
            bottomMargin=LAYOUT.BOTTOM_MARGIN_CM * cm
        )

        # Initialize content and state
        self.story: List = []
        self.citation_extractor = CitationExtractor()
        self.pending_sources: List[Dict] = []

        # Create styles using StyleFactory
        self._create_styles()

        logger.debug("PDF builder initialized successfully")

    def _create_styles(self) -> None:
        """
        Create paragraph styles using StyleFactory.

        This method creates all necessary styles for document rendering
        using the centralized StyleFactory.
        """
        logger.debug("Creating PDF paragraph styles")

        styles = StyleFactory.create_all_styles()

        self.title_style = styles['title']
        self.section_style = styles['section']
        self.subsection_style = styles['subsection']
        self.body_style = styles['body']
        self.warning_style = styles['warning']
        self.footnote_style = styles['footnote']
        self.footnote_header_style = styles['footnote_header']

        logger.debug("PDF styles created successfully")
    
    def add_document_header(self, title: str = "Plejeforløbsplan") -> None:
        """
        Add styled document header with colored box.

        Args:
            title: Document title text (default: "Plejeforløbsplan")
        """
        logger.debug(f"Adding document header: '{title}'")

        # Create colored header box
        header_table = Table(
            [[Paragraph(title, self.title_style)]],
            colWidths=[self.doc.width]
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS.PRIMARY_BLUE),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), LAYOUT.HEADER_TOP_PADDING),
            ('BOTTOMPADDING', (0, 0), (-1, -1), LAYOUT.HEADER_BOTTOM_PADDING),
            ('ROUNDEDCORNERS', [LAYOUT.HEADER_BORDER_RADIUS] * 4),
        ]))

        self.story.append(header_table)
        self.story.append(Spacer(1, 20))
    
    def add_section_heading(self, text: str) -> None:
        """
        Add section heading (Overskrift).

        Args:
            text: Section heading text
        """
        logger.debug(f"Adding section heading: '{text}'")

        # Reset citations for new major section
        self.citation_extractor.reset_for_new_section()
        self.pending_sources = []

        self.story.append(Spacer(1, 15))
        self.story.append(Paragraph(text, self.section_style))

    def add_subsection_heading(self, text: str) -> None:
        """
        Add subsection heading (SUBSECTION_TITLE).

        Args:
            text: Subsection heading text
        """
        logger.debug(f"Adding subsection heading: '{text}'")
        self.story.append(Paragraph(text, self.subsection_style))

    def add_body_text(self, text: str, collect_citations: bool = True) -> None:
        """
        Add body text with optional citation extraction.

        Args:
            text: Body text content
            collect_citations: Whether to extract and number citations (default: True)
        """
        if not text.strip():
            return

        logger.debug(f"Adding body text ({len(text)} chars, citations={collect_citations})")

        if collect_citations:
            # Extract citations and collect sources
            cleaned_text, sources = self.citation_extractor.extract_citations(text)

            if sources:
                logger.debug(f"Collected {len(sources)} citation(s) from body text")
                self.pending_sources.extend(sources)

            # Add cleaned text
            paragraphs = cleaned_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    self.story.append(Paragraph(para.strip(), self.body_style))
        else:
            # Add text without citation processing
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    self.story.append(Paragraph(para.strip(), self.body_style))

    def add_warning_box(self, text: str) -> None:
        """
        Add warning box for unanswerable questions.

        Args:
            text: Warning text (will be cleaned of standard prefixes)
        """
        if not text.strip():
            return

        logger.debug(f"Adding warning box: '{text[:50]}...'")

        # Clean the text
        display_text = text.replace("Kunne ikke besvares ud fra patientjournalen:", "").strip()
        display_text = display_text.replace("Kunne ikke besvares:", "").strip()

        if display_text:
            full_text = f"<b>Kunne ikke besvares:</b> {display_text}"
            self.story.append(Paragraph(full_text, self.warning_style))
    
    def flush_pending_sources(self) -> None:
        """
        Add all pending sources as footnotes at end of subsection.

        Deduplicates sources while preserving order, then renders
        them as a footnote section.
        """
        if not self.pending_sources:
            return

        logger.debug(f"Flushing {len(self.pending_sources)} pending source(s)")

        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in self.pending_sources:
            key = (source['number'], source['note_type'], source['timestamp'])
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)

        logger.debug(f"After deduplication: {len(unique_sources)} unique source(s)")

        self._add_footnotes(unique_sources)
        self.pending_sources = []

    def _add_footnotes(self, sources: List[Dict[str, any]]) -> None:
        """
        Add footnotes section with source citations.

        Args:
            sources: List of source dictionaries with 'number', 'note_type', 'timestamp'
        """
        self.story.append(Spacer(1, 8))
        self.story.append(Paragraph("<b>Kilder:</b>", self.footnote_header_style))

        for source in sources:
            footnote_text = f"[{source['number']}] {source['note_type']} - {source['timestamp']}"
            self.story.append(Paragraph(footnote_text, self.footnote_style))

        self.story.append(Spacer(1, 8))

    def add_page_break(self) -> None:
        """Add page break between sections."""
        logger.debug("Adding page break")
        self.story.append(PageBreak())

    def build(self) -> None:
        """
        Build and save the PDF document.

        Flushes any remaining pending sources, builds the document,
        and saves it to the output path.

        Raises:
            PDFGenerationError: If PDF building fails
        """
        try:
            logger.info(f"Building PDF document: {self.output_path}")

            # Flush any remaining sources
            self.flush_pending_sources()

            # Build the PDF
            self.doc.build(self.story)

            logger.info(f"PDF created successfully: {self.output_path}")

        except Exception as e:
            logger.error(f"PDF creation failed: {e}", exc_info=True)
            raise PDFGenerationError(
                f"Failed to build PDF",
                details=f"Output path: {self.output_path}, Error: {str(e)}"
            )


def detect_content_type(content: str) -> str:
    """
    Detect content type for special rendering.

    Analyzes content to determine which style to apply during PDF rendering.
    Currently distinguishes between warning boxes and regular body text.

    Args:
        content: Text content to analyze

    Returns:
        Content type string: 'warning' or 'body'

    Example:
        >>> detect_content_type("Kunne ikke besvares ud fra journalen")
        'warning'
        >>> detect_content_type("Patient has diabetes")
        'body'
    """
    content_lower = content.lower().strip()

    # Warning box detection (checks first 50 characters for efficiency)
    if WARNING_BOX_PATTERN.search(content_lower[:50]):
        logger.debug("Detected warning box content")
        return 'warning'

    return 'body'


def parse_enhanced_document(text: str) -> List[Dict]:
    """
    Parse document text into structured sections for enhanced PDF generation.
    
    Returns list of sections with metadata for rendering.
    """
    sections = []
    current_section = None
    current_subsection = None
    buffer = []
    
    lines = text.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        
        # Detect section header (Overskrift:)
        if line_stripped.startswith('Overskrift:'):
            # Save previous subsection
            if current_subsection:
                current_subsection['content'] = '\n'.join(buffer).strip()
                current_section['subsections'].append(current_subsection)
                buffer = []
            
            # Save previous section
            if current_section:
                sections.append(current_section)
            
            # Start new section
            section_title = line_stripped.replace('Overskrift:', '').strip().rstrip(':')
            current_section = {
                'type': 'section',
                'title': section_title,
                'subsections': []
            }
            current_subsection = None
            continue
        
        # Detect subsection header (SUBSECTION_TITLE:)
        if line_stripped.startswith('SUBSECTION_TITLE:'):
            # Save previous subsection
            if current_subsection:
                current_subsection['content'] = '\n'.join(buffer).strip()
                current_section['subsections'].append(current_subsection)
                buffer = []
            
            # Start new subsection
            subsection_title = line_stripped.replace('SUBSECTION_TITLE:', '').strip()
            current_subsection = {
                'type': 'subsection',
                'title': subsection_title,
                'content': ''
            }
            continue
        
        # Accumulate content
        if current_subsection is not None:
            buffer.append(line)
        elif current_section is not None:
            # Content before any subsection (intro text)
            buffer.append(line)
    
    # Save final subsection
    if current_subsection:
        current_subsection['content'] = '\n'.join(buffer).strip()
        current_section['subsections'].append(current_subsection)
    
    # Save final section
    if current_section:
        sections.append(current_section)
    
    return sections


def save_to_pdf(text: str, output_name: str) -> None:
    """
    Save document to enhanced PDF with beautiful formatting.

    Main entry point for PDF generation. Parses document text,
    extracts structure, and renders to PDF with citations and formatting.

    Args:
        text: Document text with markers (Overskrift:, SUBSECTION_TITLE:)
        output_name: Output filename (will be saved in GENERATED_DOCS_DIR)

    Raises:
        PDFGenerationError: If PDF generation fails

    Example:
        >>> document = "Overskrift: Diagnose\\n\\nContent here..."
        >>> save_to_pdf(document, "patient_123_report.pdf")
    """
    try:
        logger.info(f"Starting PDF generation for: {output_name}")

        # Ensure output directory exists
        os.makedirs(GENERATED_DOCS_DIR, exist_ok=True)
        output_path = GENERATED_DOCS_DIR / output_name

        # Create PDF builder
        pdf = EnhancedPDFBuilder(output_path)

        # Add document header
        pdf.add_document_header("Plejeforløbsplan")

        # Parse document structure
        sections = parse_enhanced_document(text)
        logger.info(f"Parsed {len(sections)} section(s) from document")

        # Render each section
        for section_idx, section in enumerate(sections):
            logger.debug(f"Rendering section {section_idx + 1}/{len(sections)}: {section['title']}")

            # Add section heading
            pdf.add_section_heading(section['title'])

            # Process subsections
            for subsection in section['subsections']:
                logger.debug(f"  Rendering subsection: {subsection['title']}")

                # Add subsection heading
                pdf.add_subsection_heading(subsection['title'])

                # Split content into paragraphs
                content = subsection['content']
                paragraphs = content.split('\n\n')

                for para in paragraphs:
                    if not para.strip():
                        continue

                    content_type = detect_content_type(para)

                    if content_type == 'warning':
                        pdf.add_warning_box(para)
                    else:
                        # All body text (including citations)
                        pdf.add_body_text(para, collect_citations=True)

            # Flush sources at end of section
            pdf.flush_pending_sources()

            # Page break after each major section (except last)
            if section_idx < len(sections) - 1:
                pdf.add_page_break()

        # Build PDF
        pdf.build()

        logger.info(f"PDF generation completed successfully: {output_name}")

    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise PDFGenerationError(
            f"Failed to generate PDF: {output_name}",
            details=str(e)
        )
        
