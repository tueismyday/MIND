"""
Enhanced PDF generation with beautiful formatting and footnoted citations.
Uses ReportLab for professional medical document output with visual hierarchy.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
    Table, TableStyle, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError
import gc

from config.settings import GENERATED_DOCS_DIR, PDF_FONT_PATH

class CitationExtractor:
    """Extracts and manages citations from text."""
    
    def __init__(self):
        self.citations = []  # List of (citation_text, source_dict)
        self.citation_map = {}  # citation_text -> number
        
    def extract_citations(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Extract citations from text and replace with superscript numbers.
        
        Returns:
            (cleaned_text, list_of_sources)
        """
        # Pattern: [Kilde: NotType - DD.MM.YYYY] or [Kilde: NotType - YYYY-MM-DD HH:MM]
        pattern = r'\[Kilde:\s*([^\]]+?)\s*-\s*(\d{2}\.\d{2}\.(?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)\]'
        
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return text, []
        
        # Process matches in reverse to maintain string positions
        cleaned_text = text
        sources = []
        
        for match in reversed(matches):
            citation_full = match.group(0)
            note_type = match.group(1).strip()
            timestamp = match.group(2).strip()
            
            # Create citation key
            citation_key = f"{note_type}|{timestamp}"
            
            # Get or assign citation number
            if citation_key not in self.citation_map:
                self.citation_map[citation_key] = len(self.citation_map) + 1
                sources.insert(0, {
                    'number': self.citation_map[citation_key],
                    'note_type': note_type,
                    'timestamp': timestamp
                })
            
            citation_num = self.citation_map[citation_key]
            
            # Replace with superscript reference
            superscript = f'<super>[{citation_num}]</super>'
            cleaned_text = cleaned_text[:match.start()] + superscript + cleaned_text[match.end():]
        
        # Sort sources by number
        sources.sort(key=lambda x: x['number'])
        
        return cleaned_text, sources
    
    def reset_for_new_section(self):
        """Reset citation counter for new major section."""
        self.citations = []
        self.citation_map = {}


class EnhancedPDFBuilder:
    """Enhanced PDF builder with beautiful formatting and footnoted citations."""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=2*cm,
            rightMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        self.story = []
        self.citation_extractor = CitationExtractor()
        self.pending_sources = []  # Collect sources for entire subsection
        
        # Setup fonts
        self.has_custom_font = self._setup_fonts()
        
        # Create styles
        self._create_styles()
    
    def _setup_fonts(self) -> bool:
        """Setup fonts with fallback."""
        if os.path.isfile(PDF_FONT_PATH):
            try:
                pdfmetrics.registerFont(TTFont('DejaVu', PDF_FONT_PATH))
                return True
            except:
                pass
        return False
    
    def _create_styles(self):
        """Create comprehensive paragraph styles."""
        styles = getSampleStyleSheet()
        font = 'DejaVu' if self.has_custom_font else 'Helvetica'
        font_bold = font if self.has_custom_font else 'Helvetica-Bold'
        
        # Document title (center header)
        self.title_style = ParagraphStyle(
            'DocumentTitle',
            parent=styles['Title'],
            fontSize=18,
            textColor=white,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName=font_bold
        )
        
        # Section heading (Overskrift)
        self.section_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=HexColor('#2c3e50'),
            spaceBefore=25,
            spaceAfter=18,  # Increased from 15 for more space
            fontName=font_bold,
            borderWidth=0,
            borderPadding=8,
            borderColor=HexColor('#3498db'),
            backColor=HexColor('#ecf0f1')
        )
        
        # Subsection heading (SUBSECTION_TITLE)
        self.subsection_style = ParagraphStyle(
            'SubsectionHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=HexColor('#34495e'),
            spaceBefore=15,
            spaceAfter=12,  # Increased from 10 for more space
            fontName=font_bold,
            leftIndent=10,
            borderWidth=0,
            borderPadding=5,
            borderColor=HexColor('#3498db'),
            backColor=HexColor('#f8f9fa')
        )
        
        # Body text
        self.body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=black,
            spaceAfter=8,
            fontName=font
        )
        
        # Warning box (Kunne ikke besvares) - REMOVED status_style
        self.warning_style = ParagraphStyle(
            'WarningBox',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=HexColor('#856404'),
            fontName=font,
            backColor=HexColor('#fff3cd'),
            borderWidth=1,
            borderColor=HexColor('#ffc107'),
            borderPadding=8,
            leftIndent=5,
            spaceBefore=10,  # Added spacing above
            spaceAfter=10    # Added spacing below
        )
        
        # Footnote style
        self.footnote_style = ParagraphStyle(
            'Footnote',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=HexColor('#7f8c8d'),
            fontName=font,
            leftIndent=15,
            spaceAfter=4
        )
        
        # Footnote header
        self.footnote_header_style = ParagraphStyle(
            'FootnoteHeader',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            textColor=HexColor('#34495e'),
            fontName=font_bold,
            spaceBefore=10,
            spaceAfter=6
        )
    
    def add_document_header(self, title: str = "Plejeforløbsplan"):
        """Add styled document header."""
        # Create colored header box
        header_table = Table(
            [[Paragraph(title, self.title_style)]],
            colWidths=[self.doc.width]
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#667eea')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
        ]))
        
        self.story.append(header_table)
        self.story.append(Spacer(1, 20))
    
    def add_section_heading(self, text: str):
        """Add section heading (Overskrift)."""
        # Reset citations for new major section
        self.citation_extractor.reset_for_new_section()
        self.pending_sources = []
        
        self.story.append(Spacer(1, 15))
        self.story.append(Paragraph(text, self.section_style))
    
    def add_subsection_heading(self, text: str):
        """Add subsection heading and reset pending sources."""
        # Flush any pending sources from previous subsection
        
        self.story.append(Paragraph(text, self.subsection_style))
    
    def add_body_text(self, text: str, collect_citations: bool = True):
        """Add body text with citation extraction - defers footnote addition."""
        if not text.strip():
            return
        
        if collect_citations:
            # Extract citations and collect sources
            cleaned_text, sources = self.citation_extractor.extract_citations(text)
            
            # Collect sources for later (don't add footnotes yet)
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
    
    def add_warning_box(self, text: str):
        """Add warning box (Kunne ikke besvares) with spacing."""
        if not text.strip():
            return
        
        # Clean the text
        display_text = text.replace("Kunne ikke besvares ud fra patientjournalen:", "").strip()
        display_text = display_text.replace("Kunne ikke besvares:", "").strip()
        
        if display_text:
            full_text = f"<b>Kunne ikke besvares:</b> {display_text}"
            self.story.append(Paragraph(full_text, self.warning_style))
    
    def flush_pending_sources(self):
        """Add all pending sources as footnotes at end of subsection."""
        if self.pending_sources:
            # Remove duplicates while preserving order
            seen = set()
            unique_sources = []
            for source in self.pending_sources:
                key = (source['number'], source['note_type'], source['timestamp'])
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(source)
            
            self._add_footnotes(unique_sources)
            self.pending_sources = []
    
    def _add_footnotes(self, sources: List[Dict]):
        """Add footnotes section."""
        self.story.append(Spacer(1, 8))
        self.story.append(Paragraph("<b>Kilder:</b>", self.footnote_header_style))
        
        for source in sources:
            footnote_text = f"[{source['number']}] {source['note_type']} - {source['timestamp']}"
            self.story.append(Paragraph(footnote_text, self.footnote_style))
        
        self.story.append(Spacer(1, 8))
    
    def add_page_break(self):
        """Add page break."""
        self.story.append(PageBreak())
    
    def build(self):
        """Build the PDF."""
        try:
            # Flush any remaining sources
            self.flush_pending_sources()
            
            self.doc.build(self.story)
            print(f"[SUCCESS] Enhanced PDF created: {self.output_path}")
        except Exception as e:
            print(f"[ERROR] PDF creation failed: {e}")
            raise


def detect_content_type(content: str) -> str:
    """
    Detect content type for special rendering.
    
    Returns: 'warning', 'body'
    """
    content_lower = content.lower().strip()
    
    # REMOVED status box detection - now treat as body text
    
    # Warning box detection
    if 'kunne ikke besvares' in content_lower[:50]:
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


def detect_content_type(content: str) -> str:
    """
    Detect content type for special rendering.
    
    Returns: 'warning', 'body'
    """
    content_lower = content.lower().strip()
    
    # REMOVED status box detection - now treat as body text
    
    # Warning box detection
    if 'kunne ikke besvares' in content_lower[:50]:
        return 'warning'
    
    return 'body'


def save_to_pdf(text: str, output_name: str) -> None:
    """
    Save document to enhanced PDF with beautiful formatting.
    
    Args:
        text: Document text with markers (Overskrift:, SUBSECTION_TITLE:)
        output_name: Output filename
    """
    try:
        # Ensure output directory exists
        os.makedirs(GENERATED_DOCS_DIR, exist_ok=True)
        output_path = GENERATED_DOCS_DIR / output_name
        
        # Create PDF builder
        pdf = EnhancedPDFBuilder(output_path)
        
        # Add document header
        pdf.add_document_header("Plejeforløbsplan")
        
        # Parse document structure
        sections = parse_enhanced_document(text)
        
        print(f"[INFO] Rendering {len(sections)} sections...")
        
        for section_idx, section in enumerate(sections):
            # Add section heading
            pdf.add_section_heading(section['title'])
            
            # Process subsections
            for subsection in section['subsections']:
                # Add subsection heading (this also flushes previous subsection's sources)
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
                        # All body text (including Status X: lines)
                        pdf.add_body_text(para, collect_citations=True)
                
            # Flush sources at end of section
            pdf.flush_pending_sources()
            
            # Page break after each major section (except last)
            if section_idx < len(sections) - 1:
                pdf.add_page_break()
        
        # Build PDF
        pdf.build()
        
        print(f"[SUCCESS] Enhanced PDF saved: {output_name}")
        
    except Exception as e:
        print(f"[ERROR] Enhanced PDF generation failed: {e}")
        raise
        
