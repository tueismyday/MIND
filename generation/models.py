"""
Data models for the document generation pipeline.

This module defines structured data classes using Python's dataclasses
for type safety and better code organization throughout the generation process.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class ValidationStage(Enum):
    """Enumeration of validation stages in the two-stage validation process."""

    FACT_CHECK = "fact_check"
    GUIDELINE_CHECK = "guideline_check"
    DISABLED = "disabled"


class FactPriority(Enum):
    """Priority level for a required fact."""

    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass
class RequiredFact:
    """Represents a single fact that needs to be found in patient records.

    This class encapsulates all information needed to retrieve and answer
    a specific factual question from the patient's electronic health records.

    Attributes:
        description: Human-readable description of what information to find
        priority: Whether this fact is required or optional (default: REQUIRED)
        search_query: Optimized query string for RAG retrieval
        note_types: List of note types to search in, or None for all types

    Example:
        >>> fact = RequiredFact(
        ...     description="Patient's current diabetes medications",
        ...     priority=FactPriority.REQUIRED,
        ...     search_query="diabetes medication current treatment",
        ...     note_types=["Medical Note", "Prescription"]
        ... )
    """

    description: str
    priority: FactPriority = FactPriority.REQUIRED
    search_query: str = ""
    note_types: Optional[List[str]] = None

    def __post_init__(self):
        """Set default search query to description if not provided."""
        if not self.search_query:
            self.search_query = self.description

        # Convert string priority to enum if needed
        if isinstance(self.priority, str):
            self.priority = FactPriority(self.priority.lower())


@dataclass
class SubsectionRequirements:
    """Structured requirements for generating a subsection.

    This class contains all the information extracted from guidelines
    needed to generate a single subsection of the document.

    Attributes:
        subsection_title: Title of the subsection
        required_facts: List of facts that need to be answered
        format_instructions: Specific formatting instructions for this subsection
        complexity_score: Complexity rating 0-10 based on number of facts
        note_types: Note types for the entire subsection, or None for all types

    Example:
        >>> requirements = SubsectionRequirements(
        ...     subsection_title="Current Medications",
        ...     required_facts=[fact1, fact2, fact3],
        ...     format_instructions="List medications with dosages",
        ...     complexity_score=7,
        ...     note_types=["Prescription", "Medical Note"]
        ... )
    """

    subsection_title: str
    required_facts: List[RequiredFact]
    format_instructions: str
    complexity_score: int = 0
    note_types: Optional[List[str]] = None

    def __post_init__(self):
        """Calculate complexity score if not provided."""
        if self.complexity_score == 0:
            self.complexity_score = min(len(self.required_facts), 10)


@dataclass
class SourceDocument:
    """Represents a source document retrieved from patient records.

    This class encapsulates all metadata and content for a source document
    used to answer facts or generate content.

    Attributes:
        entry_type: Type of medical note (e.g., "Medical Note", "Nursing Note")
        timestamp: Date and time of the note in format DD.MM.YYYY
        content: Full text content of the source
        snippet: Short excerpt for display purposes
        relevance: Relevance score 0-100 from reranking
        metadata: Additional metadata (author, department, etc.)

    Example:
        >>> source = SourceDocument(
        ...     entry_type="Medical Note",
        ...     timestamp="15.03.2024",
        ...     content="Patient diagnosed with type 2 diabetes...",
        ...     snippet="Patient diagnosed with type 2 diabetes",
        ...     relevance=95,
        ...     metadata={"author": "Dr. Smith"}
        ... )
    """

    entry_type: str
    timestamp: str
    content: str
    snippet: str = ""
    relevance: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set snippet to truncated content if not provided."""
        if not self.snippet and self.content:
            self.snippet = self.content[:80] + "..." if len(self.content) > 80 else self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility.

        Returns:
            Dictionary representation of the source document
        """
        return {
            "entry_type": self.entry_type,
            "timestamp": self.timestamp,
            "full_content": self.content,
            "snippet": self.snippet,
            "relevance": self.relevance,
            **self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceDocument':
        """Create SourceDocument from dictionary.

        Args:
            data: Dictionary with source data

        Returns:
            SourceDocument instance
        """
        return cls(
            entry_type=data.get('entry_type', 'Unknown'),
            timestamp=data.get('timestamp', ''),
            content=data.get('full_content', data.get('content', '')),
            snippet=data.get('snippet', ''),
            relevance=data.get('relevance', 0),
            metadata={k: v for k, v in data.items()
                     if k not in ['entry_type', 'timestamp', 'full_content', 'content', 'snippet', 'relevance']}
        )


@dataclass
class FactAnswer:
    """Represents an answered fact with its sources and validation status.

    This class contains the result of attempting to answer a single fact,
    including the answer text, sources used, and validation information.

    Attributes:
        fact: The original RequiredFact that was answered
        answer: The generated answer text (or "UNANSWERABLE")
        sources: List of source documents used to answer the fact
        is_answerable: Whether the fact could be answered from sources
        is_validated: Whether the answer has been validated
        was_corrected: Whether the answer was corrected during validation

    Example:
        >>> fact_answer = FactAnswer(
        ...     fact=required_fact,
        ...     answer="Patient has type 2 diabetes [Kilde: Medical Note - 15.03.2024]",
        ...     sources=[source1, source2],
        ...     is_answerable=True,
        ...     is_validated=True,
        ...     was_corrected=False
        ... )
    """

    fact: RequiredFact
    answer: str
    sources: List[SourceDocument]
    is_answerable: bool = True
    is_validated: bool = False
    was_corrected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility.

        Returns:
            Dictionary representation of the fact answer
        """
        return {
            "fact": self.fact,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "answerable": self.is_answerable,
            "validated": self.is_validated,
            "corrected": self.was_corrected
        }


@dataclass
class ValidationIssue:
    """Represents a single validation issue found during quality checking.

    Attributes:
        issue_type: Type of issue (e.g., "factual_error", "missing_source")
        description: Human-readable description of the issue
        severity: Severity level (e.g., "critical", "warning", "info")
        suggestion: Suggested fix or improvement
        location: Optional location information (line, fact, etc.)
    """

    issue_type: str
    description: str
    severity: str = "warning"
    suggestion: str = ""
    location: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating a fact answer or subsection.

    This class contains the outcome of a validation check, including
    whether validation passed and any issues found.

    Attributes:
        passed: Whether validation passed without critical issues
        issues: List of validation issues found
        corrected_text: Corrected text if validation made changes
        stage: Which validation stage produced this result
        metadata: Additional validation metadata

    Example:
        >>> result = ValidationResult(
        ...     passed=False,
        ...     issues=[issue1, issue2],
        ...     corrected_text="Corrected answer...",
        ...     stage=ValidationStage.FACT_CHECK,
        ...     metadata={"attempt": 2}
        ... )
    """

    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    corrected_text: Optional[str] = None
    stage: ValidationStage = ValidationStage.FACT_CHECK
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_critical_issues(self) -> bool:
        """Check if any issues are critical severity.

        Returns:
            True if any issue has severity "critical"
        """
        return any(issue.severity == "critical" for issue in self.issues)


@dataclass
class SubsectionOutput:
    """Complete output for a generated subsection.

    This class contains all information about a generated subsection,
    including the text, sources, validation details, and metadata.

    Attributes:
        title: Subsection title
        content: Generated text content
        sources: List of sources used
        fact_answers: List of individual fact answers
        unanswerable_items: List of facts that couldn't be answered
        validation_result: Validation result if validation was performed
        metadata: Additional metadata (generation time, word count, etc.)

    Example:
        >>> output = SubsectionOutput(
        ...     title="Current Medications",
        ...     content="Patient takes metformin 1000mg twice daily...",
        ...     sources=[source1, source2],
        ...     fact_answers=[answer1, answer2],
        ...     unanswerable_items=["Medication allergies"],
        ...     validation_result=validation_result
        ... )
    """

    title: str
    content: str
    sources: List[SourceDocument] = field(default_factory=list)
    fact_answers: List[FactAnswer] = field(default_factory=list)
    unanswerable_items: List[str] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_word_count(self) -> int:
        """Get word count of the generated content.

        Returns:
            Number of words in content
        """
        return len(self.content.split()) if self.content else 0

    def get_source_count(self) -> int:
        """Get number of unique sources used.

        Returns:
            Number of unique sources
        """
        unique_sources = set()
        for source in self.sources:
            unique_sources.add((source.entry_type, source.timestamp))
        return len(unique_sources)


@dataclass
class GenerationContext:
    """Context information for document generation.

    This class contains all the contextual information needed throughout
    the generation process for a section or document.

    Attributes:
        patient_id: Patient identifier
        section_title: Current section being generated
        subsection_title: Current subsection being generated
        guidelines: Guidelines text
        enable_validation: Whether validation is enabled
        max_validation_cycles: Maximum number of validation cycles
        max_sources_per_fact: Maximum sources to retrieve per fact
        include_references: Whether to include source references
        metadata: Additional context metadata

    Example:
        >>> context = GenerationContext(
        ...     patient_id="12345",
        ...     section_title="Medical History",
        ...     subsection_title="Chronic Conditions",
        ...     guidelines="Document all chronic conditions...",
        ...     enable_validation=True,
        ...     max_validation_cycles=2
        ... )
    """

    patient_id: Optional[str] = None
    section_title: str = ""
    subsection_title: str = ""
    guidelines: str = ""
    enable_validation: bool = True
    max_validation_cycles: int = 2
    max_sources_per_fact: int = 5
    include_references: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationStatistics:
    """Statistics about the validation process.

    Attributes:
        total_facts: Total number of facts processed
        answered_facts: Number of facts successfully answered
        unanswered_facts: Number of facts that couldn't be answered
        validated_facts: Number of facts that went through validation
        corrected_facts: Number of facts that were corrected
        validation_cycles: Number of validation cycles performed
    """

    total_facts: int = 0
    answered_facts: int = 0
    unanswered_facts: int = 0
    validated_facts: int = 0
    corrected_facts: int = 0
    validation_cycles: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "total_facts": self.total_facts,
            "answered_facts": self.answered_facts,
            "unanswered_facts": self.unanswered_facts,
            "validated_facts": self.validated_facts,
            "corrected_facts": self.corrected_facts,
            "validation_cycles": self.validation_cycles
        }
