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

1. Extracts NOTE_TYPES from guidelines (e.g., `[NOTE_TYPES: Hjemmesygepleje, Observationer]`)
2. Analyzes guideline text using LLM to identify required facts
3. For each guideline requirement, creates a `RequiredFact` with:
   - `description`: What information to find
   - `search_query`: Optimized RAG query
   - `note_types`: Allowed note types for filtering
4. Extracts format instructions (e.g., "Svar kort", "Pas på ikke at konkludere")

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
   - Searches patient EHR vector database
   - Filters by note types if specified
   - Retrieves top-k most relevant sources with metadata
   - Returns: `{patient_info: str, sources: [dict]}`

2. **Answer Generation**
   - LLM answers the specific fact using retrieved sources
   - Determines if fact is answerable from available data
   - Returns: `(answer_text, is_answerable)`

3. **Validation** (Optional - if `enable_validation=True`)
   - **Stage 1: Fact-checking** - Validates answer against patient sources
   - **Stage 2: Guideline adherence** - Ensures format compliance
   - Corrects errors and tracks revisions
   - Max validation cycles: configurable (default: 2)

4. **Source Tracking**
   - Records all sources used for this fact
   - Includes metadata: entry_type, timestamp, relevance score

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

**Optional appendices (if enabled):**

#### Reference Appendix
If `include_references=True`:
- Groups sources by entry_type (note type)
- Sorts by timestamp (newest first)
- Shows relevance scores and snippets
- Format:
  ```
  ═══════════════════════════════════════
  KILDEHENVISNINGER OG DOKUMENTATION
  ═══════════════════════════════════════

  ### HJEMMESYGEPLEJE (15 kilder)
  • 15.03.2024 (Relevans: 95%)
    Uddrag: "Patient modtager hjælp til medicindosering..."
  ```

#### Validation Report
If `enable_validation=True`:
- Overall statistics (total subsections, validation cycles, revisions)
- Two-stage breakdown (fact-check vs guideline revisions)
- Section-by-section quality metrics
- Documents which subsections required corrections

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
│        tools/patient_tools.py                               │
│        • Search patient EHR vector DB                       │
│        • Filter by note types                               │
│        • Return sources with metadata                       │
│                                                              │
│    6b. LLM ANSWER GENERATION                                │
│        • LLM answers fact using sources                     │
│        • Determine answerability                            │
│                                                              │
│    6c. VALIDATION (if enabled)                              │
│        generation/fact_validator.py                         │
│        • Stage 1: Fact-check against patient data          │
│        • Stage 2: Guideline adherence check                │
│        • Apply corrections, track revisions                 │
│                                                              │
│    6d. SOURCE TRACKING                                      │
│        • Record sources with metadata                       │
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
│    • Add reference appendix (if enabled)                    │
│    • Add validation report (if enabled)                     │
│    • Generate final document text                           │
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
│     • Print source statistics                               │
│     • Print quality metrics                                 │
│     • Generate validation report PDF                        │
│     • Output: {document_name}.pdf + validation_report.pdf  │
└─────────────────────────────────────────────────────────────┘
```

## Key Technologies

### RAG (Retrieval-Augmented Generation)
- **Guideline DB**: ChromaDB vector database with hospital guidelines
- **Patient EHR DB**: ChromaDB vector database with patient records
- **Embeddings**: Qwen3-Embedding-0.6B for semantic search (configurable CPU/GPU)
- **Reranker**: Qwen3-Reranker-0.6B for relevance scoring (configurable CPU/GPU)
- **Search**: RRF (Reciprocal Rank Fusion) for hybrid retrieval

### LLM Configuration
- **Generation**: Configurable LLM (vLLM with Qwen3-30B, GPT-4, Claude, etc.)
- **Validation**: Two-stage validation with fact-checking and guideline adherence
- **Fact Parsing**: JSON-based structured extraction

### Device Configuration
The system supports flexible device selection for embedding and reranker models:
- **Auto mode** (default): Automatically selects GPU or CPU based on available memory
- **CPU mode**: Forces CPU usage (useful when GPU is fully utilized by vLLM)
- **GPU mode**: Forces GPU usage (requires sufficient free GPU memory)

Configure via environment variables:
```bash
export EMBEDDING_DEVICE_MODE=auto   # or "cpu", "cuda"
export RERANKER_DEVICE_MODE=auto    # or "cpu", "cuda"
```

For detailed GPU configuration and vLLM setup, see [GPU Setup Guide](docs/GPU_SETUP.md).

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
├── core/
│   ├── database.py                  # Vector DB management
│   ├── memory.py                    # Session memory
│   └── embeddings.py                # Embedding functions
├── generation/
│   ├── document_generator.py        # Main document orchestration
│   ├── section_generator.py         # Section/subsection generation
│   ├── fact_parser.py               # Guideline fact extraction
│   ├── fact_based_generator.py      # Fact answering & assembly
│   └── fact_validator.py            # Two-stage validation
├── tools/
│   ├── guideline_tools.py           # Guideline retrieval
│   └── patient_tools.py             # Patient EHR retrieval
├── utils/
│   ├── pdf_utils.py                 # PDF generation & formatting
│   ├── text_processing.py           # Text parsing & assembly
│   ├── validation_report_logger.py  # Validation reporting
│   └── error_handling.py            # Retry & error handling
├── config/
│   ├── settings.py                  # Configuration
│   ├── llm_config.py                # LLM settings
│   └── reference_settings.py        # Reference presets
├── databases/
│   ├── guidelines_db/               # Guideline vector DB
│   ├── patient_db/                  # Patient EHR vector DB
│   └── generated_documents_db/      # Generated docs vector DB
└── generated_documents/             # Output PDFs
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
1. **Main document PDF**: Professional medical document with citations
2. **Validation report PDF**: Quality assurance documentation (if validation enabled)
3. **Console statistics**: Source counts, validation metrics, quality scores

The generated PDFs include:
- Structured sections and subsections
- Inline superscript citations `[1]`
- Footnotes with source references
- Highlighted warning boxes for unanswerable items
- Optional reference appendix with detailed source information
- Optional validation appendix with quality metrics
