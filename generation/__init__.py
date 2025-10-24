"""
Generation package for medical document creation.
Contains fact parsing, RAG retrieval, generation, and validation components.
"""

# Fact parsing
from generation.fact_parser import (
    GuidelineFactParser,
    SubsectionRequirements,
    RequiredFact
)

# Fact-by-fact generation
from generation.fact_based_generator import FactBasedGenerator

# Validation
from generation.fact_validator import FactValidator

# Section generation
from generation.section_generator import (
    generate_subsection_with_hybrid_approach,
    generate_section_with_hybrid_approach
)

# Document generation
from generation.document_generator import (
    EnhancedDocumentGenerator,
    DocumentGenerator
)

__all__ = [
    # Parsers
    'GuidelineFactParser',
    'SubsectionRequirements',
    'RequiredFact',
    
    # Generators
    'FactBasedGenerator',
    'EnhancedDocumentGenerator',
    'DocumentGenerator',
    
    # Validators
    'FactValidator',
    'SmartValidator',
    'ValidationResult',
    'ValidationIssue',
    
    # Retrievers
    'MultiFactRetriever',
    'SubsectionContext',
    'FactEvidence',
    
    # Functions
    'generate_subsection_with_hybrid_approach',
    'generate_section_with_hybrid_approach',
]