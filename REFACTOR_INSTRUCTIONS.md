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

### Modules to Refactor
- `database.py` - Vector database management
- `embeddings.py` - Embedding model handling
- `reranker.py` - Cross-encoder reranking
- `memory.py` - Session memory

### Key Focus Areas
- [ ] Database connection management and error handling
- [ ] Resource cleanup (context managers)
- [ ] Type hints for embedding vectors
- [ ] Consistent error handling across all core modules
- [ ] Performance optimization for vector operations

---

## 3. Tools Folder - Retrieval Utilities

**Priority: MEDIUM-HIGH** (Critical for RAG pipeline)

### Modules to Refactor
- `guideline_tools.py` - Guideline retrieval
- `patient_tools.py` - Patient record retrieval
- `document_tools.py` - Document manipulation
- `hybrid_search.py` - RRF hybrid search

### Key Focus Areas
- [ ] Unify retrieval interfaces
- [ ] Standardize search result format
- [ ] Extract RRF algorithm to separate utility
- [ ] Add caching for repeated searches
- [ ] Improve error handling for missing data

---

## 4. Config Folder - Configuration Management

**Priority: MEDIUM** (Impacts all modules)

### Modules to Refactor
- `settings.py` - Main configuration
- `llm_config.py` - vLLM configuration
- `reference_settings.py` - Reference presets

### Key Focus Areas
- [ ] Create configuration schema/validation (pydantic)
- [ ] Environment variable support
- [ ] Configuration file support (YAML/JSON)
- [ ] Validation for all config values
- [ ] Documentation for each config parameter

---

## 5. Utils Folder - Supporting Utilities

**Priority: MEDIUM** (Supporting infrastructure)

### Modules to Refactor
- `pdf_utils.py` - PDF generation
- `text_processing.py` - Text parsing and manipulation
- `token_tracker.py` - Token usage tracking
- `error_handling.py` - Retry logic and error utilities
- `validation_report_logger.py` - Validation reporting

### Key Focus Areas
- [ ] Consolidate text processing functions
- [ ] Extract PDF styling configuration
- [ ] Improve token tracking accuracy
- [ ] Standardize retry mechanisms
- [ ] Create reusable report generation framework

---

## 6. Agents Folder - Agent Framework

**Priority: LOW-MEDIUM** (Single file, limited scope)

### Modules to Refactor
- `retrieval_agent.py` - Retrieval agent

### Key Focus Areas
- [ ] Document agent architecture
- [ ] Standardize agent interface
- [ ] Improve error handling
- [ ] Add agent configuration options

---

## 7. Scripts Folder - Utility Scripts

**Priority: LOW** (Support scripts)

### Files to Refactor
- `launch_with_gpu.sh` - Launch script
- `verify_gpu_setup.py` - GPU verification

### Key Focus Areas
- [ ] Add error checking to bash scripts
- [ ] Improve GPU verification output
- [ ] Add script documentation

---

## 8. Root Level - Entry Points

**Priority: MEDIUM** (User-facing)

### Files to Refactor
- `generate_document_direct.py` - Main entry point
- Other root-level scripts

### Key Focus Areas
- [ ] Command-line argument parsing (argparse)
- [ ] Configuration file loading
- [ ] User-friendly error messages
- [ ] Progress indicators
- [ ] Clean exit handling

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
