# MIND Codebase Refactoring Plan

## Overview
This document outlines the refactoring plan for the MIND (Medical Documentation Generation System) codebase. The goal is to improve code quality, readability, maintainability, and professionalism across all modules.

## General Refactoring Principles
1. **Code Organization**: Logical structure, clear separation of concerns
2. **Naming Conventions**: Descriptive, consistent, PEP 8 compliant
3. **Documentation**: Comprehensive docstrings (Google/NumPy style), inline comments for complex logic
4. **Type Hints**: Full type annotations for all functions and methods
5. **Error Handling**: Robust exception handling with specific error types
6. **Constants**: Extract magic numbers/strings to named constants
7. **DRY Principle**: Eliminate code duplication
8. **Logging**: Consistent, structured logging throughout
9. **Code Complexity**: Break down large functions into smaller, testable units
10. **Configuration**: Centralize configuration values

---

## 1. Generation Folder - Document Generation Pipeline

**Priority: HIGH** (Core business logic)

### Current State Analysis
The generation folder contains 5 modules implementing a three-phase document generation pipeline:
- `document_generator.py` (EnhancedDocumentGenerator - main orchestration)
- `section_generator.py` (Section and subsection generation)
- `fact_parser.py` (GuidelineFactParser - extracts facts from guidelines)
- `fact_based_generator.py` (Fact answering and subsection assembly)
- `fact_validator.py` (Two-stage validation system)

### Refactoring Tasks

#### 1.1 Code Structure & Organization
- [ ] **Extract constants**: Move all magic strings (prompts, markers, delimiters) to a `generation/constants.py` file
  - Section markers: `"Overskrift:"`, `"SUBSECTION_TITLE:"`, etc.
  - Prompt templates and instruction text
  - Validation thresholds and parameters
  - Format strings and delimiters

- [ ] **Create data classes**: Define structured data models using `dataclasses` or `pydantic`
  - `Fact` class (question, answer, sources, answerability, validation status)
  - `Section` class (title, content, subsections, metadata)
  - `ValidationResult` class (passed, issues, suggestions, stage)
  - `GenerationContext` class (patient_id, section_type, guidelines, settings)

- [ ] **Separate concerns**: Split large classes into focused components
  - Extract prompt building logic into `PromptBuilder` class
  - Extract validation report formatting into separate module
  - Create `SourceTracker` class for citation management

#### 1.2 Naming Conventions
- [ ] **Standardize function names**: Ensure all functions use clear, verb-based names
  - `generate_section()` → Keep (clear)
  - Review all method names for consistency with PEP 8

- [ ] **Improve variable names**: Replace abbreviations and unclear names
  - `llm` → `language_model` or keep `llm` (industry standard)
  - Single letter variables in loops → descriptive names
  - `temp_` prefixes → more descriptive contextual names

- [ ] **Class naming**: Ensure consistency
  - `EnhancedDocumentGenerator` → Consider `DocumentGenerationOrchestrator` or `DocumentGenerator`
  - `GuidelineFactParser` → Good, keep
  - Review all class names for clarity

#### 1.3 Documentation
- [ ] **Add module docstrings**: Each file needs a comprehensive module-level docstring
  - Purpose of the module
  - Key classes and functions
  - Usage examples
  - Dependencies

- [ ] **Complete function docstrings**: Every public function/method needs:
  ```python
  def function_name(param1: Type1, param2: Type2) -> ReturnType:
      """Brief one-line description.

      Longer description explaining the function's purpose,
      behavior, and any important details.

      Args:
          param1: Description of param1
          param2: Description of param2

      Returns:
          Description of return value

      Raises:
          ValueError: When invalid input is provided
          RuntimeError: When operation fails

      Example:
          >>> result = function_name(value1, value2)
          >>> print(result)
      """
  ```

- [ ] **Add inline comments**: Document complex algorithms and business logic
  - Validation scoring logic
  - Fact extraction heuristics
  - Multi-stage processing flows

#### 1.4 Type Hints
- [ ] **Add complete type annotations**: All functions need proper typing
  ```python
  from typing import List, Dict, Optional, Tuple, Any

  def process_facts(
      facts: List[Dict[str, Any]],
      context: GenerationContext
  ) -> Tuple[List[Fact], ValidationResult]:
      ...
  ```

- [ ] **Use modern type syntax**: Leverage Python 3.9+ features
  - `list[str]` instead of `List[str]` (if Python 3.9+)
  - `dict[str, Any]` instead of `Dict[str, Any]`

- [ ] **Create type aliases**: For complex types used multiple times
  ```python
  FactList = List[Dict[str, Any]]
  SourceMetadata = Dict[str, Union[str, int, float]]
  ```

#### 1.5 Error Handling
- [ ] **Define custom exceptions**: Create domain-specific exception classes
  ```python
  # generation/exceptions.py
  class GenerationError(Exception):
      """Base exception for generation errors"""

  class FactParsingError(GenerationError):
      """Raised when fact parsing fails"""

  class ValidationError(GenerationError):
      """Raised when validation fails critically"""

  class InsufficientDataError(GenerationError):
      """Raised when patient data is insufficient"""
  ```

- [ ] **Improve error messages**: Make them actionable and informative
  - Include context (which section, which fact, etc.)
  - Suggest remediation steps
  - Log full stack traces at appropriate levels

- [ ] **Add graceful degradation**: Handle partial failures
  - Continue processing when individual facts fail
  - Collect and report all errors at the end
  - Provide partial results when possible

#### 1.6 Logging
- [ ] **Standardize logging**: Use Python's `logging` module consistently
  ```python
  import logging

  logger = logging.getLogger(__name__)

  logger.debug("Detailed diagnostic information")
  logger.info("General informational messages")
  logger.warning("Warning messages")
  logger.error("Error messages", exc_info=True)
  ```

- [ ] **Add structured logging**: Include context in log messages
  - Patient ID
  - Section being processed
  - Phase of generation
  - Timing information

- [ ] **Remove debug prints**: Replace all `print()` statements with proper logging

#### 1.7 Code Duplication
- [ ] **Extract common patterns**: Identify repeated code blocks
  - LLM invocation patterns → Create `LLMClient` wrapper
  - Prompt formatting → Create `PromptFormatter` utility
  - Result parsing → Create `ResponseParser` utility
  - Retry logic → Use decorator or utility function

- [ ] **Create helper functions**: For repeated operations
  - Source extraction and formatting
  - Citation formatting
  - Validation report generation

#### 1.8 Function Complexity
- [ ] **Break down large functions**: Split functions > 50 lines
  - Extract helper methods
  - Create pipeline stages as separate functions
  - Use composition over long sequential code

- [ ] **Reduce nesting**: Flatten deeply nested conditionals
  - Use early returns
  - Extract conditions into named variables
  - Consider strategy pattern for complex branching

- [ ] **Single Responsibility**: Each function should do one thing well
  - Separate data retrieval from processing
  - Separate validation from generation
  - Separate formatting from content creation

#### 1.9 Testing Considerations
- [ ] **Make functions testable**: Design for unit testing
  - Inject dependencies (don't create them inside functions)
  - Return explicit values (avoid side effects)
  - Separate I/O from logic

- [ ] **Add example test stubs**: Create test file structure
  ```python
  # tests/generation/test_fact_parser.py
  # tests/generation/test_fact_validator.py
  # etc.
  ```

#### 1.10 Performance
- [ ] **Add profiling decorators**: Measure critical paths
- [ ] **Cache expensive operations**: LLM calls, embeddings, etc.
- [ ] **Optimize data structures**: Use appropriate data types
- [ ] **Add progress tracking**: For long-running operations

### Success Criteria for Generation Folder
- [ ] All modules have comprehensive docstrings
- [ ] All functions have type hints
- [ ] No `print()` statements (use logging)
- [ ] Constants extracted to dedicated file
- [ ] Custom exceptions defined and used
- [ ] No functions exceed 50 lines (except where justified)
- [ ] Code passes `pylint` with score > 8.5
- [ ] Code passes `mypy` type checking
- [ ] All magic numbers/strings are named constants

---

## 2. Core Folder - System Fundamentals

**Priority: HIGH** (Foundation for all other modules)

### Current State Analysis
The core folder contains 4 modules providing fundamental infrastructure:
- `database.py` (DatabaseManager - ChromaDB connection management)
- `embeddings.py` (Embedding model loading with GPU/CPU fallback)
- `reranker.py` (Cross-encoder reranker model loading)
- `memory.py` (Session memory management)

These modules use singleton patterns, have good GPU memory detection, but need standardization.

### Refactoring Tasks

#### 2.1 Code Structure & Organization
- [ ] **Create base classes**: Establish common patterns
  - `BaseModelLoader` abstract class for embeddings/reranker
  - `BaseResourceManager` for database lifecycle management
  - Shared initialization, error handling, and logging patterns

- [ ] **Extract device selection logic**: Consolidate GPU/CPU selection
  - Create `core/device_manager.py` with unified device selection
  - Remove duplicate GPU memory checking code from embeddings.py and reranker.py
  - Centralize torch.cuda.mem_get_info() calls
  - Single source of truth for MIN_FREE_MEMORY thresholds

- [ ] **Database connection pooling**: Improve resource management
  - Consider implementing connection pooling if needed
  - Add database health checks
  - Implement proper cleanup/disconnect methods

- [ ] **Separate concerns in DatabaseManager**:
  - Split `get_database_info()` formatting from data retrieval
  - Extract statistics gathering into separate module
  - Consider lazy loading optimization for database properties

#### 2.2 Naming Conventions
- [ ] **Standardize variable names**:
  - `_embedding_model_cache` → `_cached_embedding_model` (more readable)
  - `_embedding_device_cache` → `_cached_embedding_device`
  - Review all private variable naming for consistency

- [ ] **Improve function names**:
  - `print_database_info()` → `display_database_statistics()` (more descriptive)
  - `get_embeddings()` → `get_embedding_model()` (clearer)
  - `get_reranker()` → `get_reranker_model()` (parallel to embedding)

- [ ] **Consistent prefixes**: Use clear prefixes for different types
  - `_load_*` for loading operations
  - `_validate_*` for validation operations
  - `_check_*` for checking operations

#### 2.3 Documentation
- [ ] **Add comprehensive module docstrings**:
  ```python
  """
  Vector database management for the MIND system.

  This module provides a centralized database manager that handles connections
  to three ChromaDB vector databases: patient records, clinical guidelines, and
  generated documents. Uses lazy loading for efficient resource management.

  Key Classes:
      DatabaseManager: Singleton manager for all vector database instances

  Dependencies:
      - langchain_chroma: Vector database interface
      - core.embeddings: Embedding model for vectorization
      - core.reranker: Cross-encoder for relevance scoring

  Example:
      >>> from core.database import db_manager
      >>> patient_db = db_manager.patient_db
      >>> results = patient_db.similarity_search("diabetes", k=5)
  """
  ```

- [ ] **Document singleton patterns**: Explain caching behavior
  - Why singleton is used
  - Thread-safety considerations
  - Memory implications

- [ ] **Add device selection documentation**: Explain fallback logic
  - Document GPU memory requirements
  - Explain when CPU fallback occurs
  - Provide troubleshooting guidance

#### 2.4 Type Hints
- [ ] **Add complete type annotations**: Especially for vector operations
  ```python
  from typing import List, Dict, Any, Optional, Tuple
  from numpy.typing import NDArray
  import numpy as np

  def embed_query(text: str) -> NDArray[np.float32]:
      """Generate embedding vector for query text."""
      ...

  def get_database_info(self) -> Dict[str, int]:
      """Get database statistics."""
      ...
  ```

- [ ] **Create type aliases**: For common complex types
  ```python
  # core/types.py
  EmbeddingVector = NDArray[np.float32]
  DeviceType = Literal["cpu", "cuda", "cuda:0", "cuda:1"]
  DatabaseStats = Dict[str, int]
  ```

- [ ] **Type hint GPU memory values**: Make units explicit
  ```python
  free_memory_gb: float  # In gigabytes
  min_required_memory_gb: float = 1.5  # Minimum GB needed
  ```

#### 2.5 Error Handling
- [ ] **Define custom exceptions**:
  ```python
  # core/exceptions.py
  class CoreError(Exception):
      """Base exception for core module errors"""

  class DatabaseConnectionError(CoreError):
      """Failed to connect to vector database"""

  class ModelLoadingError(CoreError):
      """Failed to load ML model"""

  class InsufficientGPUMemoryError(ModelLoadingError):
      """Insufficient GPU memory for model loading"""

  class DeviceNotAvailableError(CoreError):
      """Requested device not available"""
  ```

- [ ] **Improve error handling in embeddings.py**:
  - Replace bare `except:` with specific exception types
  - Provide actionable error messages
  - Log full context (device attempted, memory available, error details)

- [ ] **Add validation for database operations**:
  - Check database exists before operations
  - Validate collection is not empty
  - Handle ChromaDB-specific errors gracefully

- [ ] **Graceful degradation**: Handle partial failures
  - If one database fails, others should still work
  - Document which operations require which databases

#### 2.6 Logging
- [ ] **Replace print statements**: Use proper logging
  ```python
  import logging

  logger = logging.getLogger(__name__)

  # Before:
  print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL_NAME}")

  # After:
  logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
  ```

- [ ] **Structured logging for GPU operations**:
  ```python
  logger.debug(
      "GPU memory check",
      extra={
          'total_gb': total_memory,
          'used_gb': used_memory,
          'free_gb': free_memory,
          'device': device,
          'operation': 'embedding_load'
      }
  )
  ```

- [ ] **Add performance logging**: Track initialization times
  - Log how long each model takes to load
  - Log database connection times
  - Help diagnose performance issues

#### 2.7 Code Duplication
- [ ] **Extract GPU memory checking**: Create shared utility
  ```python
  # core/device_manager.py
  def check_gpu_memory(
      device_id: int = 0,
      min_required_gb: float = 1.5
  ) -> Tuple[bool, Dict[str, float]]:
      """
      Check if GPU has sufficient free memory.

      Returns:
          Tuple of (is_sufficient, memory_info_dict)
      """
  ```

- [ ] **Consolidate model loading patterns**: Both embeddings and reranker have similar code
  - Create `BaseModelLoader` with common loading logic
  - Share retry logic
  - Share device fallback logic
  - Share cache management

- [ ] **Extract singleton pattern**: Create reusable decorator
  ```python
  # core/singleton.py
  def singleton(cls):
      """Decorator to make a class a singleton."""
      instances = {}
      def get_instance(*args, **kwargs):
          if cls not in instances:
              instances[cls] = cls(*args, **kwargs)
          return instances[cls]
      return get_instance
  ```

#### 2.8 Function Complexity
- [ ] **Break down `get_embeddings()`**: Currently 182 lines
  - Extract `_determine_device_strategy()` helper
  - Extract `_attempt_model_load()` helper
  - Extract `_verify_model_functionality()` helper
  - Main function should be high-level orchestration

- [ ] **Simplify device selection logic**: Reduce nesting
  - Use early returns for invalid cases
  - Extract device validation to separate function
  - Create device fallback chain as data structure

- [ ] **Refactor DatabaseManager properties**: Reduce duplication
  - Create generic `_get_or_create_db()` method
  - Properties call generic method with specific paths

#### 2.9 Testing Considerations
- [ ] **Make models mockable**: Allow test doubles
  - Accept model instances via dependency injection
  - Don't hard-code model creation
  - Allow overriding model paths for testing

- [ ] **Add database fixtures**: For testing
  - Create test database utilities
  - Mock ChromaDB for unit tests
  - Provide sample data loaders

- [ ] **Test GPU fallback logic**: Ensure CPU fallback works
  - Mock GPU availability
  - Mock insufficient memory scenarios
  - Verify graceful degradation

#### 2.10 Performance
- [ ] **Optimize model loading**: Reduce startup time
  - Consider lazy loading reranker (only when needed)
  - Profile embedding generation time
  - Cache embeddings for repeated queries

- [ ] **Database query optimization**:
  - Add query result caching
  - Batch operations where possible
  - Monitor ChromaDB performance

- [ ] **Memory management**:
  - Implement proper CUDA cache cleanup
  - Monitor memory leaks in long-running processes
  - Add memory usage tracking

### Success Criteria for Core Folder
- [ ] All modules have comprehensive docstrings
- [ ] All functions have complete type hints
- [ ] All print statements replaced with logging
- [ ] Custom exceptions defined and used consistently
- [ ] Singleton pattern properly documented
- [ ] Device selection logic consolidated in one place
- [ ] No bare except clauses
- [ ] GPU memory checking code not duplicated
- [ ] Code passes `mypy` type checking
- [ ] `get_embeddings()` function under 50 lines

---

## 3. Tools Folder - Retrieval Utilities

**Priority: MEDIUM-HIGH** (Critical for RAG pipeline)

### Current State Analysis
The tools folder contains 4 modules implementing the RAG retrieval pipeline:
- `guideline_tools.py` (Guideline search with section organization)
- `patient_tools.py` (Patient record retrieval)
- `document_tools.py` (Document manipulation utilities)
- `hybrid_search.py` (RRFHybridSearch - 709 lines, complex two-stage retrieval)

The hybrid_search.py is particularly complex with RRF fusion, cross-encoder reranking, recency boosting, and Danish medical text tokenization.

### Refactoring Tasks

#### 3.1 Code Structure & Organization
- [ ] **Break down hybrid_search.py**: Currently 709 lines, too complex
  - Extract `RRFAlgorithm` class (RRF fusion logic only)
  - Extract `MedicalTextTokenizer` class (Danish medical tokenization)
  - Extract `RecencyCalculator` class (recency boost logic)
  - Extract `ScoreNormalizer` class (score normalization utilities)
  - Keep `RRFHybridSearch` as orchestrator only

- [ ] **Create unified retrieval interface**: Standardize across tools
  ```python
  # tools/base_retriever.py
  from abc import ABC, abstractmethod
  from dataclasses import dataclass

  @dataclass
  class RetrievalResult:
      """Standard retrieval result format"""
      content: str
      metadata: Dict[str, Any]
      score: float
      source_type: str  # "guideline", "patient", "document"

  class BaseRetriever(ABC):
      @abstractmethod
      def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
          """Retrieve relevant documents"""
          pass
  ```

- [ ] **Separate search strategies**: Make swappable
  - `SemanticSearchStrategy` (embedding-based)
  - `KeywordSearchStrategy` (BM25-based)
  - `HybridSearchStrategy` (RRF combination)
  - Use strategy pattern for easy switching

- [ ] **Extract configuration**: Move magic numbers to constants
  - RRF_K_VALUE = 60
  - RERANK_WINDOW_DEFAULT = 100
  - CROSS_ENCODER_WEIGHT = 90.0
  - RECENCY_WEIGHT = 10.0
  - DECAY_CONSTANT = 3

#### 3.2 Naming Conventions
- [ ] **Improve variable names in hybrid_search.py**:
  - `k` → `rrf_smoothing_constant` (more descriptive)
  - `query_doc_pairs` → `query_document_pairs`
  - `doc_idx` → `document_index`
  - `rrf_scores` → `reciprocal_rank_fusion_scores`

- [ ] **Standardize result field names**: Across all tools
  - Decide on `timestamp` vs `date`
  - Standardize `entry_type` vs `note_type` vs `document_type`
  - Use consistent score naming (total_score, relevance_score, etc.)

- [ ] **Function naming consistency**:
  - `retrieve_guidelines_by_section()` vs `retrieve_guideline_knowledge()`
  - Consider: `get_guideline_sections()` and `search_guideline_content()`
  - Use retrieve_ for database ops, search_ for queries

#### 3.3 Documentation
- [ ] **Add comprehensive module docstrings**: Especially hybrid_search.py
  ```python
  """
  Two-stage hybrid search using RRF fusion and cross-encoder reranking.

  This module implements a sophisticated retrieval pipeline optimized for Danish
  medical text. The search process has two distinct stages:

  Stage 1 - Filtering (RRF):
      Combines semantic (embedding-based) and keyword (BM25-based) search using
      Reciprocal Rank Fusion to narrow the document corpus.

  Stage 2 - Reranking:
      Applies cross-encoder neural reranking and recency boosting to provide
      fresh, context-aware relevance scores.

  Key Classes:
      RRFHybridSearch: Main search orchestrator
      RRFHybridRetriever: ChromaDB adapter for hybrid search

  Algorithm Details:
      - RRF Formula: score = Σ 1/(k + rank) for each ranking system
      - Cross-encoder: 0-90 point scale (neural relevance)
      - Recency: 0-10 point scale (temporal relevance)
      - Total: 0-100 maximum score

  Danish Medical Text Handling:
      - Custom tokenizer preserving dosages (mg, ml, etc.)
      - Medical abbreviation preservation
      - Danish stopword filtering

  Example:
      >>> search = RRFHybridSearch(embedding_model, k=60)
      >>> search.index_documents(documents)
      >>> results = search.rrf_search("diabetes behandling", top_k=5)
  """
  ```

- [ ] **Document the two-stage scoring approach**: Add diagrams/flowcharts
  - Explain why RRF scores are discarded after Stage 1
  - Document score ranges and their meanings
  - Provide examples of score interpretation

- [ ] **Add docstrings for complex algorithms**:
  - `_apply_reciprocal_rank_fusion()`: Explain RRF formula
  - `_tokenize_danish_medical()`: Document preserved patterns
  - `_calculate_recency_boost()`: Explain exponential decay

#### 3.4 Type Hints
- [ ] **Add complete type annotations**: Especially for complex structures
  ```python
  from typing import List, Dict, Tuple, Optional, Literal
  from dataclasses import dataclass

  RankingList = List[Tuple[int, float]]  # [(doc_idx, score), ...]
  DocumentDict = Dict[str, Any]
  ScoringWeights = Dict[str, float]

  def rrf_search(
      self,
      query: str,
      top_k: int = 15,
      rank_window: int = 100,
      enable_boosting: bool = True,
      note_types: Optional[List[str]] = None
  ) -> List[Dict[str, Any]]:
      ...
  ```

- [ ] **Create result data classes**: Replace dict returns
  ```python
  @dataclass
  class HybridSearchResult:
      content: str
      entry_type: str
      date: str
      document_id: str
      chunk_index: int
      score: float
      rrf_score: float
      cross_encoder_score: float
      recency_boost: float
      semantic_rank: Optional[int]
      keyword_rank: Optional[int]
  ```

- [ ] **Type hint guideline tools**: Standardize return types
  - `retrieve_guidelines_by_section()` → `Dict[str, str]` or custom SectionMap type
  - `retrieve_guideline_knowledge()` → `GuidelineSearchResult` dataclass

#### 3.5 Error Handling
- [ ] **Add validation**: Check inputs before processing
  - Validate `top_k > 0`
  - Validate `rank_window > 0`
  - Check documents are indexed before search
  - Validate date formats in recency calculation

- [ ] **Handle edge cases in hybrid_search.py**:
  - Empty document corpus
  - Query with no matches
  - All documents have same score
  - Division by zero in normalization

- [ ] **Add specific error handling for guideline tools**:
  - Handle missing sections gracefully
  - Provide fallback when no guidelines found
  - Handle metadata inconsistencies

- [ ] **Use custom exceptions**:
  ```python
  # tools/exceptions.py
  class RetrievalError(Exception):
      """Base exception for retrieval errors"""

  class EmptyCorpusError(RetrievalError):
      """No documents available for search"""

  class InvalidSearchParametersError(RetrievalError):
      """Search parameters are invalid"""

  class DateParsingError(RetrievalError):
      """Failed to parse document date"""
  ```

#### 3.6 Logging
- [ ] **Replace print statements**: Convert to proper logging
  ```python
  import logging
  logger = logging.getLogger(__name__)

  # Before:
  print(f"[INFO] Performing RRF search (k={self.k})")

  # After:
  logger.info(f"Performing RRF search with k={self.k}")
  ```

- [ ] **Add structured logging for search operations**:
  ```python
  logger.debug(
      "Hybrid search executed",
      extra={
          'query_length': len(query),
          'top_k': top_k,
          'results_found': len(results),
          'semantic_results': len(semantic_ranking),
          'keyword_results': len(keyword_ranking),
          'execution_time_ms': elapsed_time
      }
  )
  ```

- [ ] **Log performance metrics**: Track search performance
  - RRF fusion time
  - Cross-encoder reranking time
  - Total search time
  - Number of documents processed

#### 3.7 Code Duplication
- [ ] **Extract score normalization**: Used in multiple places
  ```python
  # tools/scoring.py
  def normalize_scores(
      scores: np.ndarray,
      target_range: Tuple[float, float] = (0.0, 100.0)
  ) -> np.ndarray:
      """Normalize scores to target range"""
      ...
  ```

- [ ] **Consolidate result formatting**: Similar code in guideline/patient tools
  - Create `ResultFormatter` class
  - Shared markdown formatting
  - Consistent section headers
  - Unified metadata display

- [ ] **Extract ranking logic**: Shared between semantic and keyword
  - Generic `_rank_by_score()` function
  - Shared filtering by note_types
  - Common window size application

#### 3.8 Function Complexity
- [ ] **Break down `rrf_search()`**: Currently 100+ lines
  - Extract Stage 1 (filtering) to `_stage1_filter()`
  - Extract Stage 2 (reranking) to `_stage2_rerank()`
  - Extract result assembly to `_assemble_results()`

- [ ] **Simplify `retrieve_guideline_knowledge()`**: 234 lines is too long
  - Extract filtering logic to `_filter_by_query_terms()`
  - Extract grouping logic to `_group_by_document_section()`
  - Extract output formatting to `_format_guideline_output()`

- [ ] **Reduce nesting in `_load_cross_encoder()`**: Deep try-catch nesting
  - Use early returns
  - Extract each attempt strategy to separate function
  - Simplify fallback chain

#### 3.9 Testing Considerations
- [ ] **Make search testable**: Allow dependency injection
  - Accept embedding model as parameter (don't create internally)
  - Accept cross-encoder as parameter
  - Allow mock databases for testing

- [ ] **Create test fixtures**: Sample data for testing
  - Sample Danish medical documents
  - Sample queries with expected rankings
  - Edge case scenarios (empty results, tie scores, etc.)

- [ ] **Add ranking validation tests**:
  - Verify RRF formula correctness
  - Test score normalization bounds
  - Validate recency calculation

#### 3.10 Performance
- [ ] **Profile RRF search**: Identify bottlenecks
  - Measure embedding generation time
  - Measure BM25 scoring time
  - Measure cross-encoder time
  - Optimize slowest component

- [ ] **Add caching**: For repeated queries
  - Cache embedding for common queries
  - Cache BM25 index (already done)
  - Cache cross-encoder results

- [ ] **Batch operations**: Where possible
  - Batch cross-encoder scoring (already using batch_size)
  - Consider batch embedding generation
  - Optimize document indexing

- [ ] **Optimize Danish tokenization**: Profile and improve
  - Pre-compile regex patterns
  - Consider caching tokenized queries
  - Optimize stopword filtering

### Success Criteria for Tools Folder
- [ ] All modules have comprehensive docstrings
- [ ] All functions have complete type hints
- [ ] Unified retrieval interface implemented
- [ ] No functions exceed 50 lines
- [ ] Print statements replaced with logging
- [ ] Custom exceptions defined and used
- [ ] Result formats standardized across tools
- [ ] RRF algorithm extracted to separate class
- [ ] Danish tokenization extracted to separate module
- [ ] Code passes `pylint` with score > 8.0
- [ ] `hybrid_search.py` split into logical modules

---

## 4. Config Folder - Configuration Management

**Priority: MEDIUM** (Impacts all modules)

### Current State Analysis
The config folder contains 3 modules managing system configuration:
- `settings.py` (256 lines - paths, model settings, device configuration)
- `llm_config.py` (vLLM setup and configuration)
- `reference_settings.py` (Reference presets for different modes)

The settings.py is comprehensive but uses module-level constants which makes testing difficult and lacks validation.

### Refactoring Tasks

#### 4.1 Code Structure & Organization
- [ ] **Migrate to Pydantic**: Use BaseSettings for validation
  ```python
  # config/settings.py
  from pydantic import BaseSettings, Field, validator
  from pathlib import Path
  from typing import Literal

  class MINDSettings(BaseSettings):
      """Main configuration for MIND system"""

      # Paths
      base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
      data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

      # Models
      embedding_model_name: str = Field(
          default="Qwen/Qwen3-Embedding-0.6B",
          description="HuggingFace embedding model identifier"
      )

      # Device configuration
      embedding_device_mode: Literal["auto", "cpu", "cuda"] = Field(
          default="cpu",
          description="Device selection mode for embedding model"
      )

      @validator('embedding_device_mode')
      def validate_device_mode(cls, v):
          if v not in ["auto", "cpu", "cuda"]:
              raise ValueError(f"Invalid device mode: {v}")
          return v

      class Config:
          env_file = ".env"
          env_file_encoding = "utf-8"
          env_prefix = "MIND_"
  ```

- [ ] **Separate configuration concerns**: Split settings.py
  - `config/paths.py` - All path-related settings
  - `config/models.py` - Model names and device settings
  - `config/retrieval.py` - RAG and search parameters
  - `config/generation.py` - Document generation settings
  - Keep `settings.py` as aggregator

- [ ] **Create configuration factory**: Centralize config creation
  ```python
  # config/__init__.py
  def get_settings() -> MINDSettings:
      """Get validated configuration singleton"""
      ...
  ```

- [ ] **Extract device logic**: Move device selection from settings
  - Device determination is business logic, not configuration
  - Move to `core/device_manager.py`
  - Settings should only store device mode preference

#### 4.2 Naming Conventions
- [ ] **Consistent naming for environment variables**:
  - Use `MIND_` prefix for all env vars
  - Convert to uppercase: `MIND_EMBEDDING_MODEL_NAME`
  - Document all supported env vars

- [ ] **Standardize config variable naming**:
  - All caps for constants: `EMBEDDING_MODEL_NAME`
  - lowercase for pydantic fields: `embedding_model_name`
  - Choose one style and apply consistently

- [ ] **Clear names for boolean flags**:
  - `USE_HYBRID_MULTI_FACT_APPROACH` → `enable_multi_fact_retrieval` (verb-based)
  - Make intent clear from name

#### 4.3 Documentation
- [ ] **Add comprehensive module docstrings**:
  ```python
  """
  Configuration management for the MIND medical documentation system.

  This module provides centralized configuration using Pydantic BaseSettings
  for validation and environment variable support. Configuration can be loaded
  from:
      - Environment variables (MIND_* prefix)
      - .env file
      - Direct instantiation with overrides

  Configuration Categories:
      - Paths: Data directories and file locations
      - Models: ML model identifiers and device settings
      - Retrieval: RAG search parameters
      - Generation: Document generation settings
      - vLLM: Language model server configuration

  Environment Variables:
      MIND_EMBEDDING_MODEL_NAME: HuggingFace model for embeddings
      MIND_EMBEDDING_DEVICE_MODE: Device mode (auto/cpu/cuda)
      MIND_VLLM_MODE: vLLM mode (server/local)
      ... (document all supported vars)

  Example:
      >>> from config import get_settings
      >>> settings = get_settings()
      >>> print(settings.embedding_model_name)
  """
  ```

- [ ] **Document each configuration parameter**: Add descriptions
  - Purpose of each setting
  - Valid values and ranges
  - Default values
  - Impact on system behavior

- [ ] **Create configuration guide**: Documentation file
  - How to configure for different deployment scenarios
  - Development vs production settings
  - GPU vs CPU configuration
  - Performance tuning guide

#### 4.4 Type Hints
- [ ] **Add complete type annotations**: For all config functions
  ```python
  from pathlib import Path
  from typing import Tuple, Dict, Optional

  def ensure_directories() -> None:
      """Create all necessary directories"""
      ...

  def get_patient_file_path() -> Optional[Path]:
      """Get path to patient file if it exists"""
      ...

  def get_device_config() -> Tuple[str, str, str]:
      """Get device configuration with fallback"""
      ...
  ```

- [ ] **Use Literal types**: For enumerated values
  ```python
  from typing import Literal

  DeviceMode = Literal["auto", "cpu", "cuda"]
  VLLMMode = Literal["server", "local"]
  ```

- [ ] **Type hint LLM config**: Ensure type safety
  - Type hint all LLM initialization parameters
  - Use TypedDict for configuration dictionaries

#### 4.5 Error Handling
- [ ] **Add configuration validation**: Catch errors early
  - Validate paths exist (or can be created)
  - Validate model names are valid
  - Validate numeric ranges (temperature, top_p, etc.)
  - Validate mutually exclusive options

- [ ] **Use Pydantic validators**: Built-in validation
  ```python
  @validator('temperature')
  def validate_temperature(cls, v):
      if not 0.0 <= v <= 2.0:
          raise ValueError(f"Temperature must be between 0.0 and 2.0, got {v}")
      return v
  ```

- [ ] **Custom exceptions for config errors**:
  ```python
  # config/exceptions.py
  class ConfigurationError(Exception):
      """Base exception for configuration errors"""

  class InvalidConfigValueError(ConfigurationError):
      """Configuration value is invalid"""

  class MissingRequiredConfigError(ConfigurationError):
      """Required configuration is missing"""

  class InvalidPathError(ConfigurationError):
      """Path configuration is invalid"""
  ```

- [ ] **Fail fast on invalid configuration**: Don't start with bad config
  - Validate all settings at startup
  - Provide clear error messages
  - Suggest corrections

#### 4.6 Logging
- [ ] **Log configuration loading**: Track what config is active
  ```python
  import logging
  logger = logging.getLogger(__name__)

  def get_settings() -> MINDSettings:
      logger.info("Loading MIND configuration")
      settings = MINDSettings()
      logger.info(f"Embedding model: {settings.embedding_model_name}")
      logger.info(f"Device mode: {settings.embedding_device_mode}")
      return settings
  ```

- [ ] **Log environment variable usage**: Debug configuration issues
  - Log which env vars were found
  - Log which values came from .env vs environment
  - Log final computed values

- [ ] **Avoid logging secrets**: Don't log sensitive values
  - API keys, passwords should not be logged
  - Mask sensitive values in logs

#### 4.7 Code Duplication
- [ ] **Extract path creation logic**: Used in multiple places
  ```python
  def ensure_directory(path: Path, description: str = "") -> Path:
      """Ensure a directory exists, create if needed"""
      path.mkdir(parents=True, exist_ok=True)
      logger.debug(f"Ensured directory exists: {path} ({description})")
      return path
  ```

- [ ] **Consolidate device config logic**: Duplicated in settings.py
  - Single function for device determination
  - Remove duplicate GPU memory checks
  - Share with core/device_manager.py

#### 4.8 Function Complexity
- [ ] **Simplify `get_device_config()`**: Currently 100+ lines
  - Extract `_check_cuda_availability()`
  - Extract `_check_gpu_memory()`
  - Extract `_determine_auto_device()`
  - Main function should orchestrate only

- [ ] **Break down settings initialization**: Separate concerns
  - Path setup separate from model setup
  - Device configuration separate from model names

#### 4.9 Testing Considerations
- [ ] **Make configuration testable**: Easy to override
  - Don't rely on global state
  - Allow dependency injection of settings
  - Support test configuration fixtures

- [ ] **Create test configuration**: Separate test settings
  ```python
  # config/test_settings.py
  def get_test_settings() -> MINDSettings:
      """Get settings optimized for testing"""
      return MINDSettings(
          embedding_device_mode="cpu",  # Always CPU for tests
          vllm_mode="local",
          # ... test-optimized values
      )
  ```

- [ ] **Environment variable mocking**: For testing
  - Use pytest-env or similar
  - Test different configuration scenarios

#### 4.10 Performance
- [ ] **Cache configuration**: Don't recompute
  - Singleton pattern for settings
  - Lazy loading where appropriate
  - Avoid repeated file I/O

- [ ] **Optimize imports**: Reduce startup time
  - Don't execute heavy operations at import time
  - Move `get_device_config()` execution to runtime
  - Lazy load torch only when needed

### Success Criteria for Config Folder
- [ ] Pydantic BaseSettings implemented
- [ ] All configuration parameters documented
- [ ] Environment variable support (MIND_* prefix)
- [ ] .env file support
- [ ] All validators in place
- [ ] Configuration validation at startup
- [ ] Type hints for all config functions
- [ ] Custom exceptions defined
- [ ] No heavy operations at import time
- [ ] Test configuration available
- [ ] Configuration guide document created

---

## 5. Utils Folder - Supporting Utilities

**Priority: MEDIUM** (Supporting infrastructure)

### Current State Analysis
The utils folder contains 6 modules providing support functionality:
- `pdf_utils.py` - PDF generation with ReportLab
- `text_processing.py` - Text parsing (section splitting, date parsing, document assembly)
- `token_tracker.py` - Token usage tracking
- `error_handling.py` - Retry decorators and safe wrappers (already has some good patterns)
- `validation_report_logger.py` - Validation report generation
- `profiling.py` - Performance profiling utilities

### Refactoring Tasks

#### 5.1 Code Structure & Organization
- [ ] **Organize text_processing.py**: Group related functions
  - Section manipulation functions together
  - Date/time functions together
  - Document assembly functions together
  - Consider splitting into submodules if it grows

- [ ] **Separate PDF concerns**: Split rendering from styling
  - `pdf_renderer.py` - Core PDF generation logic
  - `pdf_styles.py` - Styling constants and configuration
  - `pdf_utils.py` - High-level PDF utilities

- [ ] **Create report generation framework**: Generalize validation reporting
  ```python
  # utils/reporting.py
  class ReportGenerator(ABC):
      """Base class for report generation"""

      @abstractmethod
      def generate(self, data: Any) -> str:
          """Generate report from data"""
          pass

  class MarkdownReportGenerator(ReportGenerator):
      """Generate markdown reports"""
      ...

  class ValidationReportGenerator(ReportGenerator):
      """Generate validation reports"""
      ...
  ```

- [ ] **Extract profiling decorators**: Make reusable
  - Create decorator library
  - Timing decorators
  - Memory profiling decorators
  - Token usage decorators

#### 5.2 Naming Conventions
- [ ] **Improve function names**:
  - `split_section_into_subsections()` → `parse_section_subsections()` (clearer intent)
  - `parse_date_safe()` → `parse_medical_record_date()` (more specific)
  - `assemble_final_document()` → `assemble_document_from_sections()`

- [ ] **Consistent prefix usage**:
  - `parse_*` for text parsing operations
  - `format_*` for formatting operations
  - `generate_*` for generation operations
  - `extract_*` for extraction operations

- [ ] **Clear variable names in text_processing.py**:
  - `parts` → `section_parts` or `split_sections`
  - `i` → `section_index` in loops
  - Make loop variables descriptive

#### 5.3 Documentation
- [ ] **Add comprehensive module docstrings**:
  ```python
  """
  Text processing utilities for medical document generation.

  This module provides functions for parsing and manipulating medical text,
  including section extraction, date parsing, and document assembly.

  Key Functions:
      parse_section_subsections: Split section into subsections by markers
      parse_medical_record_date: Parse dates in medical record formats
      assemble_document_from_sections: Assemble final document from parts

  Section Format:
      Documents use these markers:
      - Overskrift: Section title marker
      - Sub_section: Subsection title marker
      - SUBSECTION_TITLE: Alternative subsection marker

  Date Formats:
      Supports multiple Danish medical date formats:
      - "yy.mm.dd HH:MM"
      - "yy.mm.dd"

  Example:
      >>> sections = parse_section_subsections(section_text)
      >>> for subsection in sections:
      ...     print(subsection['title'], subsection['content'])
  """
  ```

- [ ] **Document PDF styling options**: Explain customization
  - Available fonts
  - Color schemes
  - Layout options
  - How to modify styles

- [ ] **Document token tracker usage**: Integration guide
  - How to enable/disable
  - How to read statistics
  - Performance implications

#### 5.4 Type Hints
- [ ] **Add complete type annotations**: All utility functions
  ```python
  from typing import List, Dict, Optional, Any, Tuple
  from datetime import datetime
  from pathlib import Path

  def parse_section_subsections(section_text: str) -> List[Dict[str, str]]:
      """Split section into subsections"""
      ...

  def parse_medical_record_date(date_str: str) -> datetime:
      """Parse medical record date"""
      ...

  def generate_pdf(
      content: str,
      output_path: Path,
      title: str = "Medical Document"
  ) -> Path:
      """Generate PDF from markdown content"""
      ...
  ```

- [ ] **Create data classes for structured returns**:
  ```python
  @dataclass
  class Subsection:
      """Represents a document subsection"""
      title: str
      content: str
      intro: Optional[str] = None

  @dataclass
  class TokenUsageStats:
      """Token usage statistics"""
      total_prompt_tokens: int
      total_completion_tokens: int
      total_tokens: int
      operations: Dict[str, Dict[str, int]]
  ```

- [ ] **Type hint report structures**: Consistent report types
  - ValidationReport dataclass
  - SectionReport dataclass
  - UsageReport dataclass

#### 5.5 Error Handling
- [ ] **Improve error handling in text_processing.py**:
  - Handle non-string inputs gracefully
  - Add validation for section markers
  - Provide fallbacks for malformed input

- [ ] **Add specific exceptions**:
  ```python
  # utils/exceptions.py
  class UtilityError(Exception):
      """Base exception for utility errors"""

  class TextParsingError(UtilityError):
      """Failed to parse text"""

  class DateParsingError(UtilityError):
      """Failed to parse date"""

  class PDFGenerationError(UtilityError):
      """Failed to generate PDF"""

  class InvalidSectionFormatError(TextParsingError):
      """Section format is invalid"""
  ```

- [ ] **Enhance retry decorator**: More control
  - Allow custom exception types to retry
  - Allow max time limit (not just max retries)
  - Add jitter to backoff
  - Log retry attempts properly

- [ ] **Handle edge cases in date parsing**:
  - Invalid date formats
  - Future dates (might be typos)
  - Missing dates
  - Document fallback behavior

#### 5.6 Logging
- [ ] **Add logging to text processing**: Track operations
  ```python
  import logging
  logger = logging.getLogger(__name__)

  def parse_section_subsections(section_text: str) -> List[Dict[str, str]]:
      logger.debug(f"Parsing section into subsections (length: {len(section_text)})")
      subsections = ...
      logger.debug(f"Found {len(subsections)} subsections")
      return subsections
  ```

- [ ] **Log PDF generation**: Track PDF operations
  - Log PDF file creation
  - Log styling applied
  - Log any warnings or issues
  - Track PDF generation time

- [ ] **Structured logging for profiling**: Performance tracking
  ```python
  logger.info(
      "Operation completed",
      extra={
          'operation': operation_name,
          'duration_ms': duration,
          'tokens_used': token_count
      }
  )
  ```

#### 5.7 Code Duplication
- [ ] **Extract regex patterns**: Don't repeat compiled patterns
  ```python
  # utils/text_patterns.py
  import re

  # Compile patterns once
  SUBSECTION_PATTERN = re.compile(r'(Sub_section:\s*[^\n]+)')
  DATE_PATTERN_YMD_HMS = re.compile(r'\d{2}\.\d{2}\.\d{2}\s+\d{2}:\d{2}')
  DATE_PATTERN_YMD = re.compile(r'\d{2}\.\d{2}\.\d{2}')
  ```

- [ ] **Consolidate date parsing**: Single authoritative function
  - Avoid duplicate date parsing logic
  - Share DATE_FORMATS configuration
  - Single source of truth for date handling

- [ ] **Extract common report formatting**: Shared utilities
  - Common markdown formatting
  - Table generation
  - Section headers

#### 5.8 Function Complexity
- [ ] **Simplify `split_section_into_subsections()`**: Reduce complexity
  - Extract subsection header parsing
  - Extract intro text handling
  - Separate concerns more clearly

- [ ] **Break down PDF generation**: If complex
  - Separate content parsing from rendering
  - Extract style application
  - Create page layout helpers

- [ ] **Simplify retry decorator**: Better structure
  - Extract backoff calculation
  - Extract logging logic
  - Make more readable

#### 5.9 Testing Considerations
- [ ] **Create test fixtures**: Sample data for utilities
  - Sample section text with various formats
  - Sample dates in different formats
  - Edge cases (empty, malformed, etc.)

- [ ] **Make functions pure**: Where possible
  - Avoid side effects
  - Make testable in isolation
  - Use dependency injection

- [ ] **Add property-based tests**: For text processing
  - Test with random valid/invalid inputs
  - Verify invariants
  - Use hypothesis library

#### 5.10 Performance
- [ ] **Optimize regex usage**: Compile once, reuse
  - Pre-compile all patterns
  - Cache compiled regexes
  - Profile regex performance

- [ ] **Profile PDF generation**: Optimize if needed
  - Measure generation time
  - Identify bottlenecks
  - Optimize slow operations

- [ ] **Optimize token tracking**: Minimal overhead
  - Make tracking optional/toggleable
  - Use efficient data structures
  - Batch statistics updates

### Success Criteria for Utils Folder
- [ ] All modules have comprehensive docstrings
- [ ] All functions have complete type hints
- [ ] Custom exceptions defined and used
- [ ] Regex patterns pre-compiled and reused
- [ ] No functions exceed 50 lines
- [ ] Logging added to key operations
- [ ] Data classes used for structured returns
- [ ] PDF styling separated from logic
- [ ] Report generation framework created
- [ ] Code passes `pylint` with score > 8.5

---

## 6. Agents Folder - Agent Framework

**Priority: LOW-MEDIUM** (Single file, limited scope)

### Current State Analysis
The agents folder contains 1 module:
- `retrieval_agent.py` - Retrieval agent (if it exists, or this may be placeholder)

This is a minimal folder that may expand in the future.

### Refactoring Tasks

#### 6.1 Code Structure & Organization
- [ ] **Define agent interface**: Create base agent class
  ```python
  # agents/base_agent.py
  from abc import ABC, abstractmethod
  from typing import Any, Dict, List

  class BaseAgent(ABC):
      """Base class for all agents"""

      def __init__(self, name: str, config: Dict[str, Any]):
          self.name = name
          self.config = config

      @abstractmethod
      def run(self, input_data: Any) -> Any:
          """Execute agent logic"""
          pass

      @abstractmethod
      def reset(self) -> None:
          """Reset agent state"""
          pass
  ```

- [ ] **Separate agent concerns**: If agent grows complex
  - Agent logic separate from tool definitions
  - Agent configuration separate from implementation
  - Agent state management separate from execution

#### 6.2 Documentation
- [ ] **Document agent architecture**: Explain design
  - What agents are responsible for
  - How agents interact with tools
  - Agent lifecycle and state management
  - How to create new agents

- [ ] **Add comprehensive docstrings**: For agent classes
  - Purpose and responsibilities
  - Input/output specifications
  - Configuration options
  - Usage examples

#### 6.3 Type Hints & Error Handling
- [ ] **Add complete type annotations**: All agent code
- [ ] **Define agent exceptions**: Specific error types
  ```python
  # agents/exceptions.py
  class AgentError(Exception):
      """Base exception for agent errors"""

  class AgentExecutionError(AgentError):
      """Agent execution failed"""

  class AgentConfigurationError(AgentError):
      """Agent configuration invalid"""
  ```

#### 6.4 Testing & Configuration
- [ ] **Make agents testable**: Dependency injection
- [ ] **Add agent configuration**: Externalize settings
- [ ] **Improve error handling**: Graceful failures

### Success Criteria for Agents Folder
- [ ] Base agent interface defined
- [ ] All agents have comprehensive docstrings
- [ ] Type hints for all agent code
- [ ] Custom exceptions defined
- [ ] Agent configuration externalized
- [ ] Agents are testable and mockable

---

## 7. Scripts Folder - Utility Scripts

**Priority: LOW** (Support scripts)

### Current State Analysis
The scripts folder contains utility scripts:
- `launch_with_gpu.sh` - GPU launch script
- `verify_gpu_setup.py` - GPU verification utility

These are operational/deployment scripts.

### Refactoring Tasks

#### 7.1 Shell Script Improvements
- [ ] **Add error checking**: Set proper bash options
  ```bash
  #!/bin/bash
  set -euo pipefail  # Exit on error, undefined var, pipe failure
  IFS=$'\n\t'        # Better word splitting

  # Add error handler
  trap 'echo "Error on line $LINENO" >&2' ERR
  ```

- [ ] **Add input validation**: Check arguments
  ```bash
  if [ "$#" -ne 2 ]; then
      echo "Usage: $0 <arg1> <arg2>"
      exit 1
  fi
  ```

- [ ] **Add logging**: Better output
  ```bash
  # Use consistent logging functions
  log_info() {
      echo "[INFO] $1" >&2
  }

  log_error() {
      echo "[ERROR] $1" >&2
  }
  ```

- [ ] **Add help text**: Document usage
  ```bash
  show_help() {
      cat << EOF
  Usage: ${0##*/} [OPTIONS]

  Launch the MIND system with GPU support.

  OPTIONS:
      -h, --help              Display this help and exit
      -g, --gpu-id ID         GPU ID to use (default: 0)
      -m, --model MODEL       Model to load
  EOF
  }
  ```

#### 7.2 Python Script Improvements
- [ ] **Improve verify_gpu_setup.py**: Better verification
  ```python
  """
  GPU setup verification utility.

  Verifies that the system is properly configured for GPU acceleration,
  including CUDA availability, memory, and model compatibility.
  """

  def verify_cuda_available() -> bool:
      """Verify CUDA is available"""
      ...

  def verify_gpu_memory() -> Dict[str, float]:
      """Check GPU memory availability"""
      ...

  def verify_models_loadable() -> bool:
      """Verify models can be loaded"""
      ...

  def main():
      """Run all verification checks"""
      print("MIND GPU Setup Verification")
      print("=" * 50)

      # Run checks with clear output
      checks = [
          ("CUDA availability", verify_cuda_available),
          ("GPU memory", verify_gpu_memory),
          ("Model loading", verify_models_loadable),
      ]

      for name, check_fn in checks:
          print(f"\nChecking {name}...")
          try:
              result = check_fn()
              print(f"✓ {name}: OK")
          except Exception as e:
              print(f"✗ {name}: FAILED - {e}")
  ```

#### 7.3 Documentation
- [ ] **Add script docstrings**: Explain purpose
  - What the script does
  - When to use it
  - Required dependencies
  - Example usage

- [ ] **Create scripts README**: Usage guide
  - List all available scripts
  - Purpose of each script
  - Usage examples
  - Troubleshooting

#### 7.4 Error Handling
- [ ] **Add comprehensive error checking**: In all scripts
- [ ] **Provide helpful error messages**: Guide users
- [ ] **Add exit codes**: Proper error signaling
  ```python
  import sys

  EXIT_SUCCESS = 0
  EXIT_CONFIG_ERROR = 1
  EXIT_RUNTIME_ERROR = 2

  try:
      main()
      sys.exit(EXIT_SUCCESS)
  except ConfigurationError as e:
      print(f"Configuration error: {e}", file=sys.stderr)
      sys.exit(EXIT_CONFIG_ERROR)
  ```

### Success Criteria for Scripts Folder
- [ ] All bash scripts have error handling (set -euo pipefail)
- [ ] All scripts have help text
- [ ] All scripts have proper logging
- [ ] Python scripts have comprehensive docstrings
- [ ] verify_gpu_setup.py provides clear output
- [ ] Scripts README created
- [ ] All scripts have proper exit codes

---

## 8. Root Level - Entry Points

**Priority: MEDIUM** (User-facing)

### Current State Analysis
Root level files are the main entry points:
- `generate_document_direct.py` - Main document generation entry point
- Other potential scripts and entry points

These are the user's primary interface to the system.

### Refactoring Tasks

#### 8.1 Code Structure & Organization
- [ ] **Implement proper CLI**: Use argparse or click
  ```python
  # generate_document_direct.py
  import argparse
  from pathlib import Path
  from typing import Optional

  def parse_arguments() -> argparse.Namespace:
      """Parse command-line arguments"""
      parser = argparse.ArgumentParser(
          description="Generate medical documentation using MIND system",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter
      )

      parser.add_argument(
          "--patient-id",
          type=str,
          required=True,
          help="Patient identifier"
      )

      parser.add_argument(
          "--guideline",
          type=str,
          required=True,
          help="Guideline document to use"
      )

      parser.add_argument(
          "--output",
          type=Path,
          default=Path("output.pdf"),
          help="Output file path"
      )

      parser.add_argument(
          "--config",
          type=Path,
          help="Configuration file path"
      )

      parser.add_argument(
          "-v", "--verbose",
          action="store_true",
          help="Enable verbose output"
      )

      return parser.parse_args()
  ```

- [ ] **Separate CLI from logic**: Keep entry point clean
  ```python
  # Main entry point
  def main():
      """Main entry point"""
      args = parse_arguments()
      setup_logging(args.verbose)

      try:
          result = generate_document(
              patient_id=args.patient_id,
              guideline=args.guideline,
              output_path=args.output
          )
          print(f"✓ Document generated: {result}")
          return 0
      except Exception as e:
          logger.error(f"Generation failed: {e}", exc_info=True)
          print(f"✗ Error: {e}", file=sys.stderr)
          return 1

  if __name__ == "__main__":
      sys.exit(main())
  ```

- [ ] **Add progress indicators**: User feedback
  ```python
  from tqdm import tqdm

  with tqdm(total=100, desc="Generating document") as pbar:
      pbar.set_description("Loading models")
      pbar.update(10)

      pbar.set_description("Retrieving patient data")
      pbar.update(20)

      # ... continue with progress updates
  ```

#### 8.2 Configuration
- [ ] **Support configuration files**: YAML/JSON config
  ```python
  import yaml

  def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
      """Load configuration from file"""
      if config_path is None:
          return {}

      with open(config_path) as f:
          return yaml.safe_load(f)
  ```

- [ ] **Environment variable support**: Override defaults
  ```python
  import os

  patient_id = args.patient_id or os.getenv("MIND_PATIENT_ID")
  ```

- [ ] **Configuration precedence**: Command line > env var > config file > defaults

#### 8.3 Error Handling
- [ ] **User-friendly error messages**: Non-technical users
  ```python
  except FileNotFoundError as e:
      print(f"✗ Error: Could not find file '{e.filename}'")
      print(f"  Please check that the file exists and try again.")
      return 1

  except DatabaseConnectionError as e:
      print(f"✗ Error: Could not connect to database")
      print(f"  Make sure the vector database is initialized.")
      print(f"  Run 'python scripts/initialize_db.py' to set up the database.")
      return 1
  ```

- [ ] **Add helpful suggestions**: Guide users to solutions
- [ ] **Proper exit codes**: Signal different error types

#### 8.4 Logging
- [ ] **Setup logging properly**: Based on verbosity
  ```python
  import logging

  def setup_logging(verbose: bool = False):
      """Configure logging"""
      level = logging.DEBUG if verbose else logging.INFO

      logging.basicConfig(
          level=level,
          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
          handlers=[
              logging.StreamHandler(),
              logging.FileHandler('mind.log')
          ]
      )
  ```

- [ ] **Separate user output from logs**: Clear distinction
  - Use print() for user messages
  - Use logging for diagnostic information
  - Log file for full details

#### 8.5 Documentation
- [ ] **Add comprehensive --help**: Clear usage info
- [ ] **Create examples**: Sample invocations
  ```bash
  # Generate document for patient 12345
  python generate_document_direct.py \
      --patient-id 12345 \
      --guideline "diabetes_guideline.pdf" \
      --output "patient_12345_diabetes.pdf"

  # With custom configuration
  python generate_document_direct.py \
      --patient-id 12345 \
      --guideline "diabetes_guideline.pdf" \
      --config my_config.yaml \
      --verbose
  ```

- [ ] **Add docstring to main()**: Document entry point
  ```python
  def main():
      """
      Main entry point for MIND document generation.

      This script generates medical documentation by:
      1. Loading patient records from vector database
      2. Retrieving relevant clinical guidelines
      3. Generating document sections using LLM
      4. Validating and refining content
      5. Producing final PDF output

      Exit codes:
          0: Success
          1: Configuration or input error
          2: Generation error
          3: Database error
      """
  ```

#### 8.6 User Experience
- [ ] **Add version information**: --version flag
  ```python
  parser.add_argument(
      "--version",
      action="version",
      version=f"MIND v{__version__}"
  )
  ```

- [ ] **Validate inputs early**: Fail fast with clear messages
  ```python
  def validate_inputs(args):
      """Validate all inputs before starting generation"""
      if not patient_file_exists(args.patient_id):
          raise ValueError(f"No patient record found for ID: {args.patient_id}")

      if not guideline_exists(args.guideline):
          raise ValueError(f"Guideline not found: {args.guideline}")

      if args.output.exists() and not args.force:
          raise ValueError(f"Output file exists: {args.output}. Use --force to overwrite.")
  ```

- [ ] **Add dry-run mode**: Preview without executing
  ```python
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Show what would be done without executing"
  )
  ```

### Success Criteria for Root Level
- [ ] Comprehensive argparse CLI implemented
- [ ] Configuration file support (YAML)
- [ ] Environment variable support
- [ ] User-friendly error messages
- [ ] Progress indicators for long operations
- [ ] Proper exit codes
- [ ] --help text is clear and comprehensive
- [ ] Logging configured based on verbosity
- [ ] Input validation with helpful messages
- [ ] Clean separation of CLI and business logic
- [ ] Examples documented

---

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)
1. Core folder refactoring
2. Config folder refactoring
3. Establish coding standards document

### Phase 2: Business Logic (Weeks 3-5)
4. Generation folder refactoring (most complex)
5. Tools folder refactoring

### Phase 3: Support & Polish (Week 6)
6. Utils folder refactoring
7. Agents folder refactoring
8. Scripts and root level cleanup

### Phase 4: Integration (Week 7)
9. End-to-end testing
10. Documentation updates
11. Performance optimization

---

## Tools & Standards

### Code Quality Tools
- **Linting**: `pylint`, `flake8`
- **Type Checking**: `mypy`
- **Formatting**: `black` (auto-formatting)
- **Import Sorting**: `isort`
- **Documentation**: `pydocstyle`

### Coding Standards
- **Style Guide**: PEP 8
- **Docstring Format**: Google Style
- **Type Hints**: PEP 484, PEP 526
- **Line Length**: 88 characters (black default) or 100
- **Complexity**: Max cyclomatic complexity 10

### Documentation Standards
- Module-level docstrings for all files
- Class docstrings with attributes section
- Function docstrings with Args, Returns, Raises
- Inline comments for complex logic
- README updates for architectural changes

---

## Progress Tracking

- [ ] **Section 1**: Generation Folder ← START HERE
- [ ] **Section 2**: Core Folder
- [ ] **Section 3**: Tools Folder
- [ ] **Section 4**: Config Folder
- [ ] **Section 5**: Utils Folder
- [ ] **Section 6**: Agents Folder
- [ ] **Section 7**: Scripts Folder
- [ ] **Section 8**: Root Level

---

## Notes
- Each section should be completed and tested before moving to the next
- Create feature branch for each major refactoring section
- Run full test suite after each section
- Update documentation as you refactor
- Keep a changelog of major changes
