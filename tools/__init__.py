"""
Tools package for the Agentic RAG Medical Documentation System.
Provides retrieval tools for patients, guidelines, and documents.
"""

# Patient tools
from .patient_tools import (
    # Main retrieval functions
    retrieve_patient_info,
    retrieve_patient_info_with_rrf_analysis,
    get_patient_info_with_sources,
    
    # Result classes
    PatientInfoResult,
    
    # Retriever classes
    RRFPatientRetriever,
    get_patient_retriever,
)

# Guideline tools
from .guideline_tools import (
    retrieve_guideline_knowledge,
    retrieve_guidelines_by_section,
)

# Document tools
from .document_tools import (
    start_document_generation,
    retrieve_generated_document_info,
)

# Hybrid search
from .hybrid_search import (
    RRFHybridSearch,
    RRFHybridRetriever,
)

__all__ = [
    # Patient tools
    'retrieve_patient_info',
    'retrieve_patient_info_with_rrf_analysis',
    'get_patient_info_with_sources',
    'PatientInfoResult',
    'RRFPatientRetriever',
    'get_patient_retriever',
    
    # Guideline tools
    'retrieve_guideline_knowledge',
    'retrieve_guidelines_by_section',
    
    # Document tools
    'start_document_generation',
    'retrieve_generated_document_info',
    
    # Hybrid search
    'RRFHybridSearch',
    'RRFHybridRetriever',
]