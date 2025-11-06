"""
PDF styling configuration for medical document generation.

This module contains all styling definitions for PDF generation including
fonts, colors, paragraph styles, and layout constants. Separating styling
from rendering logic makes the system more maintainable and customizable.

Key Components:
    - PDFColors: Color palette for document elements
    - PDFFonts: Font configuration and loading
    - StyleFactory: Creates ReportLab paragraph styles

Dependencies:
    - reportlab: PDF generation library
    - config.settings: Font paths and configuration

Example:
    >>> from utils.pdf_styles import StyleFactory
    >>> styles = StyleFactory.create_all_styles()
    >>> title_style = styles['title']
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from config.settings import PDF_FONT_PATH
from .exceptions import PDFStyleError

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Color Palette
# ============================================================================

@dataclass(frozen=True)
class PDFColors:
    """
    Color palette for PDF documents.

    All colors are defined as HexColor objects for consistency.
    Using a dataclass ensures immutability and clear organization.
    """
    # Primary colors
    PRIMARY_BLUE: HexColor = HexColor('#667eea')
    DARK_BLUE: HexColor = HexColor('#3498db')
    LIGHT_BLUE: HexColor = HexColor('#ecf0f1')

    # Text colors
    TEXT_PRIMARY: HexColor = HexColor('#2c3e50')
    TEXT_SECONDARY: HexColor = HexColor('#34495e')
    TEXT_TERTIARY: HexColor = HexColor('#7f8c8d')

    # Background colors
    BG_LIGHT_GRAY: HexColor = HexColor('#f8f9fa')
    BG_MEDIUM_GRAY: HexColor = HexColor('#ecf0f1')
    BG_DARK_GRAY: HexColor = HexColor('#95a5a6')

    # Status colors
    WARNING_BG: HexColor = HexColor('#fff3cd')
    WARNING_BORDER: HexColor = HexColor('#ffc107')
    WARNING_TEXT: HexColor = HexColor('#856404')
    ERROR_RED: HexColor = HexColor('#e74c3c')
    SUCCESS_GREEN: HexColor = HexColor('#27ae60')

    # Standard colors
    BLACK: HexColor = black
    WHITE: HexColor = white

    # Border colors
    BORDER_GRAY: HexColor = HexColor('#bdc3c7')


# Global color palette instance
COLORS = PDFColors()


# ============================================================================
# Font Configuration
# ============================================================================

class PDFFonts:
    """
    Font configuration and loading for PDF documents.

    Handles font registration with ReportLab, including fallback
    to default fonts if custom fonts are unavailable.
    """

    DEFAULT_FONT = 'Helvetica'
    DEFAULT_FONT_BOLD = 'Helvetica-Bold'
    CUSTOM_FONT_NAME = 'DejaVu'

    _custom_font_loaded = False

    @classmethod
    def load_custom_font(cls, font_path: Optional[str] = None) -> bool:
        """
        Load custom font for PDF generation.

        Args:
            font_path: Path to TrueType font file (default: from settings)

        Returns:
            True if custom font loaded successfully, False if using fallback

        Example:
            >>> PDFFonts.load_custom_font()
            True
        """
        if cls._custom_font_loaded:
            logger.debug(f"Custom font '{cls.CUSTOM_FONT_NAME}' already loaded")
            return True

        if font_path is None:
            font_path = PDF_FONT_PATH

        if not os.path.isfile(font_path):
            logger.warning(
                f"Custom font not found at {font_path}. "
                f"Using default font: {cls.DEFAULT_FONT}"
            )
            return False

        try:
            pdfmetrics.registerFont(TTFont(cls.CUSTOM_FONT_NAME, font_path))
            cls._custom_font_loaded = True
            logger.info(f"Custom font '{cls.CUSTOM_FONT_NAME}' loaded successfully from {font_path}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to load custom font from {font_path}: {e}",
                exc_info=True
            )
            raise PDFStyleError(
                f"Failed to load PDF font",
                details=f"Font path: {font_path}, Error: {str(e)}"
            )

    @classmethod
    def get_font_name(cls, bold: bool = False) -> str:
        """
        Get font name for use in styles.

        Args:
            bold: Whether to return bold font variant

        Returns:
            Font name string for ReportLab

        Example:
            >>> font = PDFFonts.get_font_name(bold=True)
            >>> print(font)
            'DejaVu' or 'Helvetica-Bold'
        """
        if cls._custom_font_loaded:
            # Custom font (DejaVu handles bold via style)
            return cls.CUSTOM_FONT_NAME
        else:
            # Default font
            return cls.DEFAULT_FONT_BOLD if bold else cls.DEFAULT_FONT

    @classmethod
    def has_custom_font(cls) -> bool:
        """Check if custom font is available."""
        return cls._custom_font_loaded


# ============================================================================
# Style Factory
# ============================================================================

class StyleFactory:
    """
    Factory for creating ReportLab paragraph styles.

    Provides methods to create all standard styles used in medical
    document PDF generation.
    """

    @staticmethod
    def create_title_style() -> ParagraphStyle:
        """
        Create document title style (center header).

        Returns:
            ParagraphStyle for document title
        """
        styles = getSampleStyleSheet()
        font = PDFFonts.get_font_name(bold=True)

        return ParagraphStyle(
            'DocumentTitle',
            parent=styles['Title'],
            fontSize=18,
            textColor=COLORS.WHITE,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName=font
        )

    @staticmethod
    def create_section_style() -> ParagraphStyle:
        """
        Create section heading style (Overskrift).

        Returns:
            ParagraphStyle for main section headings
        """
        styles = getSampleStyleSheet()
        font = PDFFonts.get_font_name(bold=True)

        return ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=COLORS.TEXT_PRIMARY,
            spaceBefore=25,
            spaceAfter=18,
            fontName=font,
            borderWidth=0,
            borderPadding=8,
            borderColor=COLORS.DARK_BLUE,
            backColor=COLORS.BG_MEDIUM_GRAY
        )

    @staticmethod
    def create_subsection_style() -> ParagraphStyle:
        """
        Create subsection heading style (SUBSECTION_TITLE).

        Returns:
            ParagraphStyle for subsection headings
        """
        styles = getSampleStyleSheet()
        font = PDFFonts.get_font_name(bold=True)

        return ParagraphStyle(
            'SubsectionHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=COLORS.TEXT_SECONDARY,
            spaceBefore=15,
            spaceAfter=12,
            fontName=font,
            leftIndent=10,
            borderWidth=0,
            borderPadding=5,
            borderColor=COLORS.DARK_BLUE,
            backColor=COLORS.BG_LIGHT_GRAY
        )

    @staticmethod
    def create_body_style() -> ParagraphStyle:
        """
        Create body text style.

        Returns:
            ParagraphStyle for body text
        """
        styles = getSampleStyleSheet()
        font = PDFFonts.get_font_name()

        return ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=COLORS.BLACK,
            spaceAfter=8,
            fontName=font,
            alignment=TA_LEFT
        )

    @staticmethod
    def create_warning_style() -> ParagraphStyle:
        """
        Create warning box style (Kunne ikke besvares).

        Returns:
            ParagraphStyle for warning boxes
        """
        styles = getSampleStyleSheet()
        font = PDFFonts.get_font_name()

        return ParagraphStyle(
            'WarningBox',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=COLORS.WARNING_TEXT,
            fontName=font,
            backColor=COLORS.WARNING_BG,
            borderWidth=1,
            borderColor=COLORS.WARNING_BORDER,
            borderPadding=8,
            leftIndent=5,
            spaceBefore=10,
            spaceAfter=10
        )

    @staticmethod
    def create_footnote_style() -> ParagraphStyle:
        """
        Create footnote text style.

        Returns:
            ParagraphStyle for footnote text
        """
        styles = getSampleStyleSheet()
        font = PDFFonts.get_font_name()

        return ParagraphStyle(
            'Footnote',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=COLORS.TEXT_TERTIARY,
            fontName=font,
            leftIndent=15,
            spaceAfter=4
        )

    @staticmethod
    def create_footnote_header_style() -> ParagraphStyle:
        """
        Create footnote header style ("Kilder:").

        Returns:
            ParagraphStyle for footnote section header
        """
        styles = getSampleStyleSheet()
        font = PDFFonts.get_font_name(bold=True)

        return ParagraphStyle(
            'FootnoteHeader',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            textColor=COLORS.TEXT_SECONDARY,
            fontName=font,
            spaceBefore=10,
            spaceAfter=6
        )

    @classmethod
    def create_all_styles(cls) -> Dict[str, ParagraphStyle]:
        """
        Create all standard PDF styles.

        Returns:
            Dictionary mapping style names to ParagraphStyle objects

        Example:
            >>> styles = StyleFactory.create_all_styles()
            >>> title_style = styles['title']
            >>> body_style = styles['body']
        """
        logger.debug("Creating all PDF styles")

        styles = {
            'title': cls.create_title_style(),
            'section': cls.create_section_style(),
            'subsection': cls.create_subsection_style(),
            'body': cls.create_body_style(),
            'warning': cls.create_warning_style(),
            'footnote': cls.create_footnote_style(),
            'footnote_header': cls.create_footnote_header_style(),
        }

        logger.debug(f"Created {len(styles)} PDF styles")
        return styles


# ============================================================================
# Layout Constants
# ============================================================================

@dataclass(frozen=True)
class PDFLayout:
    """
    Layout constants for PDF generation.

    Defines margins, spacings, and other layout parameters.
    """
    # Page margins (in cm)
    LEFT_MARGIN_CM: float = 2.0
    RIGHT_MARGIN_CM: float = 2.0
    TOP_MARGIN_CM: float = 2.0
    BOTTOM_MARGIN_CM: float = 2.0

    # Spacings (in points)
    SECTION_SPACING_BEFORE: int = 25
    SECTION_SPACING_AFTER: int = 18
    SUBSECTION_SPACING_BEFORE: int = 15
    SUBSECTION_SPACING_AFTER: int = 12
    PARAGRAPH_SPACING: int = 8
    FOOTNOTE_SPACING: int = 4

    # Header dimensions
    HEADER_TOP_PADDING: int = 15
    HEADER_BOTTOM_PADDING: int = 15
    HEADER_BORDER_RADIUS: int = 10


# Global layout constants
LAYOUT = PDFLayout()
