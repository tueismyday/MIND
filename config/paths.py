"""
Path configuration for the MIND medical documentation system.

This module centralizes all directory and file path settings,
making it easy to manage where data is stored and accessed.
"""

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .exceptions import InvalidPathError

logger = logging.getLogger(__name__)


class PathConfig(BaseModel):
    """
    Configuration for all file system paths used by the MIND system.

    Attributes:
        base_dir: Base project directory (auto-detected from this file's location)
        data_dir: Root data directory containing all data files
        guideline_dir: Directory containing clinical guideline documents
        patient_record_dir: Directory containing patient record files
        generated_docs_dir: Directory for storing generated documents
        guideline_db_dir: ChromaDB vector database for guidelines
        patient_db_dir: ChromaDB vector database for patient records
        generated_docs_db_dir: ChromaDB vector database for generated documents
        cache_dir: Cache directory for sentence transformers and models
        pdf_font_path: Path to font file for PDF generation
        default_patient_file: Optional path to default patient file
    """

    # Base directories
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Base project directory"
    )

    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data",
        description="Root data directory"
    )

    # Data subdirectories
    guideline_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing clinical guideline documents"
    )

    patient_record_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing patient record files"
    )

    generated_docs_dir: Optional[Path] = Field(
        default=None,
        description="Directory for storing generated documents"
    )

    # Vector database directories
    guideline_db_dir: Optional[Path] = Field(
        default=None,
        description="ChromaDB vector database for guidelines"
    )

    patient_db_dir: Optional[Path] = Field(
        default=None,
        description="ChromaDB vector database for patient records"
    )

    generated_docs_db_dir: Optional[Path] = Field(
        default=None,
        description="ChromaDB vector database for generated documents"
    )

    # Cache directory
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Cache directory for models"
    )

    # PDF configuration
    pdf_font_path: str = Field(
        default="DejaVuSans.ttf",
        description="Font file for PDF generation"
    )

    default_output_name: str = Field(
        default="generated_medical_document.pdf",
        description="Default name for generated PDF documents"
    )

    default_patient_file: Optional[Path] = Field(
        default=None,
        description="Default patient file if available"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

    def model_post_init(self, __context):
        """Initialize derived paths after model creation."""
        # Set subdirectories relative to data_dir if not explicitly set
        if self.guideline_dir is None:
            self.guideline_dir = self.data_dir / "hospital_guidelines"

        if self.patient_record_dir is None:
            self.patient_record_dir = self.data_dir / "patient_record"

        if self.generated_docs_dir is None:
            self.generated_docs_dir = self.data_dir / "generated_documents"

        if self.guideline_db_dir is None:
            self.guideline_db_dir = self.data_dir / "hospital_guidelines_db"

        if self.patient_db_dir is None:
            self.patient_db_dir = self.data_dir / "patient_record_db"

        if self.generated_docs_db_dir is None:
            self.generated_docs_db_dir = self.data_dir / "generated_documents_db"

        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "sentence_transformers_cache"

        if self.default_patient_file is None:
            self.default_patient_file = (
                self.patient_record_dir / "Patient_journal_Geriatrisk_patient.pdf"
            )

    @field_validator('base_dir', 'data_dir')
    @classmethod
    def validate_base_paths(cls, v: Path) -> Path:
        """Ensure base paths are valid Path objects."""
        if not isinstance(v, Path):
            v = Path(v)
        return v

    def ensure_directories(self) -> None:
        """
        Create all necessary directories if they don't exist.

        Creates the complete directory structure needed for the MIND system,
        including data directories and vector database directories.

        Raises:
            InvalidPathError: If directory creation fails
        """
        directories = [
            self.data_dir,
            self.guideline_dir,
            self.patient_record_dir,
            self.generated_docs_dir,
            self.guideline_db_dir,
            self.patient_db_dir,
            self.generated_docs_db_dir,
            self.cache_dir,
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                raise InvalidPathError(
                    f"Failed to create directory {directory}: {e}"
                ) from e

    def get_patient_file_path(self) -> Optional[str]:
        """
        Get the path to the patient file if it exists.

        Returns:
            str path if the default patient file exists, None otherwise
        """
        if self.default_patient_file and self.default_patient_file.exists():
            return str(self.default_patient_file)
        return None

    def validate_paths(self) -> bool:
        """
        Validate that critical paths exist and are accessible.

        Returns:
            True if all critical paths are valid

        Raises:
            InvalidPathError: If any critical path is invalid
        """
        # Base and data directories must exist or be creatable
        try:
            self.ensure_directories()
        except Exception as e:
            raise InvalidPathError(f"Failed to validate paths: {e}") from e

        return True


# Date parsing formats for patient records
DATE_FORMATS = ["%y.%m.%d %H:%M", "%y.%m.%d"]
