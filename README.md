# MIND - Medical Intelligence for Nurse Documentation

An intelligent system that leverages multiple LLMs to automatically generate draft medical documents for nurses. The system combines hospital guidelines with patient Electronic Health Records (EHR) to create accurate, well-referenced medical documents through a sophisticated RAG (Retrieval-Augmented Generation) pipeline.

## Overview

MIND automates the creation of various medical documents (plejeforløbsplan, udskrivningsrapport, behandlingsplan, sygeplejerapport) by:
1. Retrieving relevant hospital guidelines from a vector database
2. Analyzing patient EHR data through intelligent fact-based retrieval
3. Generating document content section by section with source citations
4. Validating accuracy through two-stage quality assurance
5. Producing professional PDF documents with proper formatting

## System Architecture & Document Generation Flow

### Entry Point: `generate_document_direct.py`

The main entry point for document generation. This script bypasses the chat agent and directly generates medical documents.

**Command-line usage:**
```bash
python generate_document_direct.py --type plejeforløbsplan --patient patient.pdf
python generate_document_direct.py --type udskrivningsrapport --patient patient.pdf --refs detailed --validate --cycles 3
```

**Key parameters:**
- `--type`: Document type (plejeforløbsplan, udskrivningsrapport, behandlingsplan, sygeplejerapport)
- `--patient`: Path to patient PDF file (required)
- `--refs`: Reference detail level (none, minimal, balanced, detailed)
- `--validate`: Enable two-stage subsection validation
- `--cycles`: Maximum validation cycles per subsection (default: 2)

### Phase 1: Guideline Retrieval

**Location:** `tools/guideline_tools.py` → `retrieve_guidelines_by_section()`

**Process:**
1. Performs semantic search in the guideline vector database using the document type query
2. Identifies the most relevant guideline document (e.g., "Plejeforløbsplan_retningslinje.pdf")
3. Retrieves ALL sections from that guideline document
4. Filters out "Unknown Section" entries (sections without proper headers)
5. Returns a dictionary: `{section_title: section_content}`

**Example output:**
```python
{
    "Stamoplysninger": "...",
    "Funktionsniveau": "...",
    "Medicinsk behandling": "..."
}
```

### Phase 2: Document Generation Initialization

**Location:** `generation/document_generator.py` → `EnhancedDocumentGenerator`

**Process:**
1. Stores retrieved guidelines in `memory_manager.retrieved_guidelines`
2. Initializes DocumentGenerator with configuration:
   - Reference settings (include_references, max_references_per_section)
   - Validation settings (enable_subsection_validation, max_revision_cycles)
3. Prepares tracking structures for sources and validation reports

### Phase 3: Section-by-Section Generation

**Location:** `generation/section_generator.py` → `generate_section_with_hybrid_approach()`

For each section from the guidelines:

#### 3.1: Section Splitting
**Function:** `split_section_into_subsections()`
- Parses the section content to identify subsections
- Extracts section intro and individual subsection guidelines
- Creates structured subsection data

#### 3.2: Subsection Generation (Fact-by-Fact Approach)
**Function:** `generate_subsection_with_hybrid_approach()`

This is the core generation pipeline with three phases:

##### **PHASE 1: Fact Parsing**
**Location:** `generation/fact_parser.py` → `GuidelineFactParser`

1. Extracts NOTE_TYPES from guidelines:
   - Format: `[NOTE_TYPES: type1, type2]` (specific types) or `[NOTE_TYPES: ALL]` (no filtering)
   - Examples: `[NOTE_TYPES: Hjemmesygepleje, Observationer]`
   - Used to filter patient records during RAG retrieval
2. Analyzes guideline text using LLM to identify required facts
3. For each guideline requirement, creates a `RequiredFact` with:
   - `description`: What information to find
   - `search_query`: Optimized RAG query
   - `note_types`: Allowed note types for filtering (extracted from step 1)
4. Extracts format instructions (e.g., "Svar kort", "Pas på ikke at konkludere", "Max 60 words")

**Example:**
```
Guideline: "Får patienten behov for medicindosering? Hvis ja: tabletter, øjendråber?"

→ Fact 1: "Behov for hjælp til medicindosering (ja/nej)"
   Query: "medicindosering hjælp behov"

→ Fact 2: "Type medicindosering: tabletter, øjendråber, salve"
   Query: "medicin tabletter øjendråber administration"
```

##### **PHASE 2: Fact-by-Fact Answering**
**Location:** `generation/fact_based_generator.py` → `FactBasedGenerator`

For each required fact:

1. **RAG Retrieval** (`tools/patient_tools.py` → `get_patient_info_with_sources`)
   - Searches patient EHR vector database using two-stage RRF hybrid search
   - Filters by note types if specified in guidelines (`[NOTE_TYPES: type1, type2]` or `[NOTE_TYPES: ALL]`)
   - Retrieves top-k most relevant sources (max 5 per fact, configurable)
   - Returns rich metadata including:
     - **Relevance**: Percentage score (0-100%)
     - **RRF scores**: Normalized (0-100), raw value, semantic/keyword ranks
     - **Cross-encoder score**: 0-90 scale
     - **Recency boost**: 0-10 scale (exponential decay from most recent DB entry)
     - **Timestamp** and **entry_type** (note type)
     - Full content and snippet
   - Fallback strategy: RRF → Lightweight hybrid → Pure Chroma semantic search

2. **Answer Generation**
   - Facts are processed using **batched LLM calls** for efficiency
   - LLM answers each fact using retrieved sources
   - Determines if fact is answerable from available data
   - Generates answer with source citations: `[Kilde: NoteType - DD.MM.YYYY]`
   - Returns: `(answer_text, is_answerable)`

3. **Validation** (Optional - if `enable_validation=True`)
   - **Stage 1: Fact-checking** - Validates answer against patient sources
     - Checks factual correctness
     - Verifies source citations are accurate
     - Ensures newest information is used
   - **Stage 2: Guideline adherence** - Ensures format compliance
     - Checks if answer follows format instructions
     - Validates structure and style requirements
   - Corrects errors and tracks revisions (fact-check vs guideline revisions)
   - Max validation cycles: configurable (default: 2, min: 1, max: 3)
   - Tracks validation history for quality metrics

4. **Source Tracking**
   - Records all sources used for this fact
   - Includes comprehensive metadata:
     - `entry_type` (note type), `timestamp`, `relevance` percentage
     - RRF scores (normalized, raw, semantic/keyword ranks)
     - Cross-encoder score and recency boost
     - Full content and snippet

##### **PHASE 3: Subsection Assembly**
**Location:** `generation/fact_based_generator.py` → `assemble_subsection_from_facts()`

1. Combines all fact-answers into coherent subsection text
2. Applies format instructions from guidelines
3. Handles unanswerable facts (creates "Kunne ikke besvares" list)
4. Inserts source citations: `[Kilde: NoteType - DD.MM.YYYY]`
5. Returns formatted subsection with `SUBSECTION_TITLE:` marker

### Phase 4: Document Assembly

**Location:** `generation/document_generator.py` → `generate_complete_document()`

**Process:**
1. Combines all section outputs with proper markers:
   - `Overskrift:` for section headings
   - `SUBSECTION_TITLE:` for subsection headings
2. Extracts final section text using `extract_final_section()`
3. Assembles final document structure

**Optional appendices (saved separately to `/reports/{base_name}_appendices.txt`):**

#### Reference Appendix
If `include_references=True`:
- Groups sources by entry_type (note type)
- Sorts by timestamp (newest first)
- Shows comprehensive RRF metadata:
  - Relevance percentage
  - RRF scores (normalized, raw, semantic/keyword ranks)
  - Cross-encoder score and recency boost
  - Content snippets
- Saved as separate text file for detailed analysis
- Format example:
  ```
  ═══════════════════════════════════════
  KILDEHENVISNINGER OG DOKUMENTATION
  ═══════════════════════════════════════

  ### HJEMMESYGEPLEJE (15 kilder)
  ## [1] Hjemmesygepleje (15.03.2024)
  *Total: 45.3/40 | CE: 30.5/30 | Recency: 14.8/10 | [RRF Filter: 75.2/100]*
  *Sem: #3 | Key: #5 | Relevans: 92%*
  Uddrag: "Patient modtager hjælp til medicindosering..."
  ```

#### Validation Report
If `enable_validation=True`:
- Overall statistics (total subsections, validation cycles, revisions)
- Two-stage breakdown:
  - **Fact-check revisions**: Count of fact-checking corrections
  - **Guideline revisions**: Count of format/adherence corrections
- Section-by-section quality metrics
- Documents which subsections required corrections and why
- Saved to separate appendices file for quality assurance

### Phase 5: PDF Generation

**Location:** `utils/pdf_utils.py` → `save_to_pdf()`

**Process:**

1. **Document Parsing** (`parse_enhanced_document()`)
   - Parses text markers (`Overskrift:`, `SUBSECTION_TITLE:`)
   - Creates structured section/subsection hierarchy

2. **PDF Building** (`EnhancedPDFBuilder`)
   - Uses ReportLab for professional formatting
   - Applies custom styles:
     - **Document header**: Colored gradient box with title
     - **Section headings**: Large, bold, with background color
     - **Subsection headings**: Medium, indented, highlighted
     - **Body text**: Clean, readable font with proper spacing
     - **Warning boxes**: Yellow highlighted for "Kunne ikke besvares"
     - **Citations**: Superscript numbers `[1]` in text

3. **Citation Extraction** (`CitationExtractor`)
   - Finds all `[Kilde: ...]` patterns
   - Assigns sequential numbers to unique sources
   - Replaces inline citations with superscript references
   - Collects sources for footnotes

4. **Footnote Generation**
   - Adds "Kilder:" section after each subsection
   - Lists numbered sources with note type and timestamp
   - Deduplicates sources while preserving order

5. **PDF Rendering**
   - Renders sections sequentially
   - Adds page breaks between major sections
   - Saves to `generated_documents/{output_name}`

**Example PDF structure:**
```
┌─────────────────────────────────────┐
│   Plejeforløbsplan                  │  ← Header (colored)
├─────────────────────────────────────┤
│ Stamoplysninger                     │  ← Section heading
│                                     │
│   Navn og alder                     │  ← Subsection heading
│   Patienten er 78 år gammel[1]      │  ← Body with citation
│                                     │
│   Kilder:                           │  ← Footnotes
│   [1] Stamkort - 15.03.2024        │
│                                     │
│   Diagnostiske oplysninger          │  ← Next subsection
│   ...                               │
└─────────────────────────────────────┘
```

### Phase 6: Document Indexing

**Location:** `generation/document_generator.py` → `index_final_document()`

**Process:**
1. Creates/updates vector database for generated documents
2. Enables future retrieval and analysis of generated documents
3. Updates global database manager with new generated docs DB
4. Stored in: `databases/generated_documents_db/`

### Phase 7: Statistics & Reporting

**Final output includes:**

1. **Source statistics** (if references enabled):
   - Total source references
   - Unique sources
   - Average relevance score
   - High relevance sources (≥80%)
   - Type distribution

2. **Quality metrics** (if validation enabled):
   - Total validation cycles
   - Fact-check revisions
   - Guideline revisions
   - Sections with revisions
   - Average cycles per section

3. **Validation report PDF**:
   - Saved to `reports/validation_report.pdf`
   - Detailed quality assurance documentation

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ENTRY POINT                                              │
│    generate_document_direct.py                              │
│    • Parse command-line arguments                           │
│    • Configure reference & validation settings              │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GUIDELINE RETRIEVAL                                      │
│    tools/guideline_tools.py                                 │
│    • Semantic search in guideline vector DB                 │
│    • Retrieve all sections from best matching guideline    │
│    • Return: {section_title: content, ...}                 │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. INITIALIZE GENERATOR                                     │
│    generation/document_generator.py                         │
│    • Store guidelines in memory                             │
│    • Create DocumentGenerator with config                   │
│    • Initialize tracking structures                         │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │ FOR EACH SECTION       │
        └────────┬───────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. SPLIT SECTION INTO SUBSECTIONS                           │
│    utils/text_processing.py                                 │
│    • Parse section content                                  │
│    • Identify subsection boundaries                         │
│    • Extract intro and subsection guidelines                │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │ FOR EACH SUBSECTION    │
        └────────┬───────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. FACT PARSING (Phase 1)                                   │
│    generation/fact_parser.py                                │
│    • Extract NOTE_TYPES from guidelines                     │
│    • LLM analyzes guideline → identify required facts       │
│    • Create RequiredFact objects with search queries        │
│    • Extract format instructions                            │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │ FOR EACH FACT          │
        └────────┬───────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. FACT ANSWERING (Phase 2)                                 │
│    generation/fact_based_generator.py                       │
│                                                              │
│    6a. RAG RETRIEVAL                                        │
│        tools/patient_tools.py (RRFPatientRetriever)         │
│        • Two-stage RRF hybrid search:                       │
│          - Stage 1: RRF filtering (semantic + BM25)         │
│          - Stage 2: Cross-encoder + recency ranking         │
│        • Filter by NOTE_TYPES if specified                  │
│        • Return max 5 sources with rich metadata            │
│                                                              │
│    6b. LLM ANSWER GENERATION                                │
│        • Batched LLM calls for efficiency                   │
│        • LLM answers fact using sources                     │
│        • Determine answerability                            │
│        • Generate citations: [Kilde: Type - Date]          │
│                                                              │
│    6c. VALIDATION (if enabled)                              │
│        generation/fact_validator.py                         │
│        • Stage 1: Fact-check against patient data          │
│          - Check factual correctness                        │
│          - Verify source citations                          │
│          - Ensure newest information used                   │
│        • Stage 2: Guideline adherence check                │
│          - Validate format compliance                       │
│        • Apply corrections, track revisions                 │
│        • Max cycles: 2 (default), range: 1-3               │
│                                                              │
│    6d. SOURCE TRACKING                                      │
│        • Record sources with comprehensive metadata:        │
│          - Relevance %, RRF scores, CE score, recency      │
│          - Entry type, timestamp, content                   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. SUBSECTION ASSEMBLY (Phase 3)                            │
│    generation/fact_based_generator.py                       │
│    • Combine fact-answers into coherent text                │
│    • Apply format instructions                              │
│    • Insert source citations [Kilde: ...]                  │
│    • Handle unanswerable facts                              │
│    • Return formatted subsection                            │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │ END SUBSECTION LOOP    │
        └────────┬───────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. COMBINE SUBSECTIONS → SECTION                            │
│    • Join all subsection outputs                            │
│    • Deduplicate sources                                    │
│    • Aggregate validation details                           │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │ END SECTION LOOP       │
        └────────┬───────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. DOCUMENT ASSEMBLY                                        │
│    generation/document_generator.py                         │
│    • Combine all sections                                   │
│    • Generate final document text                           │
│    • Save appendices separately (if enabled):               │
│      - Reference appendix with RRF metadata                 │
│      - Validation report with quality metrics               │
│      → reports/{base_name}_appendices.txt                  │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. PDF GENERATION                                          │
│     utils/pdf_utils.py                                      │
│     • Parse document structure                              │
│     • Extract citations → superscript numbers               │
│     • Create PDF with ReportLab                             │
│       - Styled headers, sections, subsections               │
│       - Body text with superscript citations                │
│       - Footnotes with source references                    │
│     • Save to generated_documents/                          │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 11. DOCUMENT INDEXING                                       │
│     • Create vector embeddings of generated document        │
│     • Store in generated_documents_db                       │
│     • Enable future retrieval                               │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 12. STATISTICS & REPORTING                                  │
│     • Print source statistics:                              │
│       - Total/unique sources, avg relevance                 │
│       - High relevance sources (≥80%)                       │
│       - Type distribution                                   │
│     • Print quality metrics (if validation enabled):        │
│       - Validation cycles, revisions (fact-check vs guide)  │
│       - Sections with revisions                             │
│     • Output files:                                         │
│       - generated_documents/{name}.pdf (main document)      │
│       - reports/{name}_appendices.txt (if enabled)         │
└─────────────────────────────────────────────────────────────┘
```

## Key Technologies

### RAG (Retrieval-Augmented Generation)
- **Guideline DB**: ChromaDB vector database with hospital guidelines
- **Patient EHR DB**: ChromaDB vector database with patient records
- **Embeddings**: Qwen3-Embedding-0.6B for semantic search (configurable CPU/GPU)
- **Reranker**: Qwen3-Reranker-0.6B (cross-encoder) for relevance scoring (configurable CPU/GPU)
- **Search**: Two-stage hybrid retrieval system
  - **Stage 1 (Filtering)**: RRF (Reciprocal Rank Fusion) combines semantic + BM25 keyword search
    - Semantic search via embedding model
    - BM25 keyword matching with Danish stopwords
    - RRF fusion formula: `Σ (1 / (k + rank))` where k=60
    - Normalized to 0-100 scale for filtering
  - **Stage 2 (Ranking)**: Fresh scoring with cross-encoder + recency
    - Cross-encoder neural relevance: 0-90 scale
    - Recency boost (exponential decay): 0-10 scale
    - Final score = Cross-encoder + Recency (max 100)
  - **Fallback chain**: RRF → Lightweight hybrid → Pure Chroma semantic search

### LLM Configuration
- **Generation**: Configurable LLM (vLLM with Qwen3-30B, GPT-4, Claude, etc.)
- **Validation**: Two-stage validation with fact-checking and guideline adherence
- **Fact Parsing**: JSON-based structured extraction

### Device Configuration
The system supports flexible device selection for embedding and reranker models via the `core/device_manager.py` module:
- **Auto mode** (default): Automatically selects GPU or CPU based on available memory
- **CPU mode**: Forces CPU usage (useful when GPU is fully utilized by vLLM)
- **GPU mode**: Forces GPU usage (requires sufficient free GPU memory)

Configure via environment variables:
```bash
export MIND_EMBEDDING_DEVICE_MODE=auto   # or "cpu", "cuda"
export MIND_RERANKER_DEVICE_MODE=auto    # or "cpu", "cuda"
```

Or via configuration:
```python
from config import get_settings
settings = get_settings()
settings.models.embedding_device_mode = "auto"
settings.models.reranker_device_mode = "cpu"
```

For detailed GPU configuration and vLLM setup, see [GPU Setup Guide](docs/GPU_SETUP.md).

### Key Configuration Parameters

The system uses **Pydantic BaseSettings** for validated configuration management. Configuration can be set via:
- Environment variables (MIND_* prefix)
- `.env` file
- Direct instantiation

**Quick Start**:
```python
from config import get_settings

settings = get_settings()
settings.ensure_initialized()  # Create required directories
```

**Model Configuration** (`config/models.py`):
```python
# Via Python
settings.models.embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
settings.models.reranker_model_name = "Qwen/Qwen3-Reranker-0.6B"
settings.models.embedding_device_mode = "cpu"  # or 'auto', 'cuda'
settings.models.reranker_device_mode = "cpu"   # or 'auto', 'cuda'

# Via environment variables
MIND_EMBEDDING_MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
MIND_EMBEDDING_DEVICE_MODE="auto"
```

**Retrieval Configuration** (`config/retrieval.py`):
```python
settings.retrieval.initial_retrieval_k = 20           # Initial RRF candidates
settings.retrieval.final_retrieval_k = 10             # After reranking
settings.retrieval.similarity_score_threshold = 0.5   # Minimum relevance
settings.retrieval.batch_size_rerank = 1              # Cross-encoder batch
settings.retrieval.max_sources_per_fact = 5           # Max patient sources
```

**Generation Configuration** (`config/generation.py`):
```python
settings.generation.enable_multi_fact_retrieval = True  # Fact-based generation
settings.generation.default_validation_cycles = 2       # Default per subsection
settings.generation.max_validation_cycles = 3           # Hard limit
settings.generation.enable_subsection_validation = True # Toggle validation
settings.generation.temperature = 0.2                   # LLM temperature
settings.generation.top_p = 0.95                        # Nucleus sampling
```

**vLLM Configuration** (`config/vllm.py`):
```python
settings.vllm.mode = "server"                # or 'local'
settings.vllm.server_url = "http://localhost:8000"
settings.vllm.model_name = "Qwen/Qwen3-30B"
```

**Reference Settings** (`config/reference_settings.py`):
```python
# Use presets
from config import apply_preset
apply_preset("balanced")  # none, minimal, balanced, detailed

# Or configure manually
settings.reference.include_references = True
settings.reference.max_references_per_section = 3
```

**Path Configuration** (`config/paths.py`):
```python
settings.paths.guideline_dir = Path("data/Retningslinjer")
settings.paths.patient_record_dir = Path("data/Journalnotater")
settings.paths.generated_docs_dir = Path("generated_documents")
```

### Quality Assurance
- **Two-stage validation**:
  1. Fact-checking: Validates against patient sources
  2. Guideline adherence: Ensures format compliance
- **Source tracking**: Every claim linked to source with timestamp
- **Revision tracking**: Detailed logs of all corrections

### PDF Generation
- **ReportLab**: Professional PDF creation with custom styling
- **Citation management**: Automatic footnote generation
- **Visual hierarchy**: Clear section/subsection structure
- **Warning boxes**: Highlighted unanswerable items

## Directory Structure

```
MIND/
├── generate_document_direct.py      # Entry point
├── core/                            # System fundamentals
│   ├── database.py                  # Vector DB management
│   ├── memory.py                    # Session memory
│   ├── embeddings.py                # Embedding functions
│   ├── reranker.py                  # Cross-encoder reranker
│   ├── device_manager.py            # GPU/CPU device selection
│   ├── types.py                     # Type definitions
│   └── exceptions.py                # Core exceptions
├── generation/                      # Document generation pipeline
│   ├── document_generator.py        # Main document orchestration
│   ├── section_generator.py         # Section/subsection generation
│   ├── fact_parser.py               # Guideline fact extraction
│   ├── fact_based_generator.py      # Fact answering & assembly
│   ├── fact_validator.py            # Two-stage validation
│   ├── models.py                    # Data classes for generation
│   ├── constants.py                 # Generation constants
│   └── exceptions.py                # Generation exceptions
├── tools/                           # Retrieval utilities
│   ├── guideline_tools.py           # Guideline retrieval
│   ├── patient_tools.py             # Patient EHR retrieval (RRF)
│   ├── hybrid_search.py             # Two-stage RRF hybrid search
│   ├── base_retriever.py            # Base retriever interface
│   ├── rrf_algorithm.py             # RRF fusion algorithm
│   ├── tokenizer.py                 # Danish medical tokenization
│   ├── scoring.py                   # Score normalization utilities
│   ├── constants.py                 # Retrieval constants
│   └── exceptions.py                # Retrieval exceptions
├── utils/                           # Supporting utilities
│   ├── pdf_utils.py                 # PDF generation & formatting
│   ├── pdf_styles.py                # PDF styling configuration
│   ├── text_processing.py           # Text parsing & assembly
│   ├── text_patterns.py             # Regex patterns for parsing
│   ├── validation_report_logger.py  # Validation reporting
│   ├── error_handling.py            # Retry & error handling
│   ├── profiling.py                 # Performance profiling
│   ├── token_tracker.py             # Token usage tracking
│   └── exceptions.py                # Utility exceptions
├── config/                          # Configuration management (Pydantic)
│   ├── settings.py                  # Main settings aggregator
│   ├── paths.py                     # Path configuration
│   ├── models.py                    # Model configuration
│   ├── retrieval.py                 # Retrieval configuration
│   ├── generation.py                # Generation configuration
│   ├── vllm.py                      # vLLM configuration
│   ├── llm_config.py                # LLM client management
│   ├── reference_settings.py        # Reference presets
│   └── exceptions.py                # Configuration exceptions
├── agents/                          # Agent framework
│   ├── base_agent.py                # Base agent interface
│   ├── retrieval_agent.py           # Retrieval agent
│   └── exceptions.py                # Agent exceptions
├── databases/
│   ├── guidelines_db/               # Guideline vector DB
│   ├── patient_db/                  # Patient EHR vector DB
│   └── generated_documents_db/      # Generated docs vector DB
├── generated_documents/             # Output PDFs
└── reports/                         # Appendices and validation reports
```

## Example Usage

### Basic generation:
```bash
python generate_document_direct.py --type plejeforløbsplan --patient patient.pdf
```

### With detailed references and validation:
```bash
python generate_document_direct.py \
  --type udskrivningsrapport \
  --patient patient.pdf \
  --refs detailed \
  --validate \
  --cycles 3 \
  --verbose
```

### Custom query:
```bash
python generate_document_direct.py \
  --query "Generér en sygeplejerapport" \
  --patient patient.pdf \
  --output rapport.pdf \
  --validate
```

## Output

Each generation produces:
1. **Main document PDF** (`generated_documents/{document_name}.pdf`):
   - Professional medical document with citations
   - Structured sections and subsections
   - Inline superscript citations `[1]`
   - Footnotes with source references (NoteType - Date)
   - Highlighted warning boxes for unanswerable items

2. **Appendices file** (`reports/{document_name}_appendices.txt`) - if references or validation enabled:
   - **Reference appendix**: Detailed source information with RRF metadata
     - Relevance percentages, RRF scores, cross-encoder scores, recency boosts
     - Grouped by entry type (note type)
     - Sorted by timestamp (newest first)
   - **Validation appendix**: Quality assurance documentation
     - Overall statistics (validation cycles, revisions)
     - Two-stage breakdown (fact-check vs guideline revisions)
     - Section-by-section quality metrics

3. **Console statistics**:
   - Source counts and types
   - Average relevance scores
   - High relevance sources (≥80%)
   - Validation metrics (if enabled)
   - Quality scores and revision counts
