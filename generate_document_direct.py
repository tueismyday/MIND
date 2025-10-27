#!/usr/bin/env python3
"""
Direct document generation script - bypasses the chat agent for immediate document creation.
This script generates medical documents directly from guidelines and patient data without agent interaction.
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.database import db_manager
from core.memory import memory_manager
from tools.guideline_tools import retrieve_guidelines_by_section
from generation.document_generator import DocumentGenerator
from config.settings import get_patient_file_path, DEFAULT_OUTPUT_NAME, ensure_directories
from config.reference_settings import REFERENCE_PRESETS, apply_preset

def main():
    """Main function with command-line interface for direct document generation."""
    parser = argparse.ArgumentParser(
        description="Direct Medical Document Generator with Two-Stage Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_document_direct.py --type plejeforløbsplan --patient patient.pdf
  python generate_document_direct.py --type udskrivningsrapport --patient patient.pdf --refs detailed --validate
  python generate_document_direct.py --query "Generér behandlingsplan" --patient patient.pdf --output behandling.pdf --validate --cycles 3
        """
    )
    
    # Document specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--type", choices=["plejeforløbsplan", "udskrivningsrapport", "behandlingsplan", "sygeplejerapport"],
                      help="Type of document to generate")
    group.add_argument("--query", type=str,
                      help="Custom query for guideline retrieval (e.g., 'Generér en plejeforløbsplan')")
    
    # Files
    parser.add_argument("--patient", type=str, required=True,
                       help="Path to patient PDF file (required)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_NAME,
                       help="Output PDF filename (default: generated_medical_document.pdf)")
    
    # Reference options
    parser.add_argument("--refs", choices=["none", "minimal", "balanced", "detailed"], default="balanced",
                       help="Reference detail level (default: balanced)")
    parser.add_argument("--max-refs", type=int,
                       help="Custom maximum references per section (overrides --refs preset)")
    
    # Two-stage validation options
    parser.add_argument("--validate", action="store_true",
                       help="Enable two-stage subsection validation (fact-checking then guideline adherence)")
    parser.add_argument("--cycles", type=int, default=2,
                       help="Maximum validation cycles per subsection (default: 2)")
    
    # Advanced options
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed progress information")
    parser.add_argument("--stats", action="store_true", default=True,
                       help="Show generation statistics (default: True)")
    parser.add_argument("--no-stats", action="store_true",
                       help="Disable generation statistics")
    
    args = parser.parse_args()
    
    # Setup
    ensure_directories()
    
    if args.verbose:
        print("=" * 60)
        print("DIRECT MEDICAL DOCUMENT GENERATOR")
        print("=" * 60)
        db_manager.print_database_info()
    
    # Validate patient file
    if not os.path.exists(args.patient):
        print(f"ERROR: Patient file not found: {args.patient}")
        sys.exit(1)
    
    # Determine document type query
    if args.type:
        type_queries = {
            "plejeforløbsplan": "Generér en plejeforløbsplan",
            "udskrivningsrapport": "Generér en udskrivningsrapport", 
            "behandlingsplan": "Generér en behandlingsplan",
            "sygeplejerapport": "Generér en sygeplejerapport"
        }
        query = type_queries[args.type]
        print(f"[INFO] Generating {args.type}")
    else:
        query = args.query
        print(f"[INFO] Using custom query: {query}")
    
    # Configure references
    if args.max_refs:
        include_refs = True
        max_refs = args.max_refs
        print(f"[CONFIG] Custom references: {max_refs} per section")
    else:
        preset = REFERENCE_PRESETS[args.refs]
        include_refs = preset["include_references"]
        max_refs = preset["max_references_per_section"]
        print(f"[CONFIG] Reference preset: {args.refs} - {preset['description']}")
    
    # Configure validation
    if args.validate:
        print(f"[CONFIG] Two-stage validation: ENABLED")
        print(f"[CONFIG] Max validation cycles: {args.cycles}")
        print(f"[CONFIG] Stage 1: Fact-checking against patient records")
        print(f"[CONFIG] Stage 2: Guideline adherence checking")
    else:
        print(f"[CONFIG] Two-stage validation: DISABLED")
    
    # Configure statistics
    show_stats = args.stats and not args.no_stats
    
    print(f"[CONFIG] Patient file: {args.patient}")
    print(f"[CONFIG] Output file: {args.output}")
    
    try:
        # Generate document directly
        generate_document_direct(
            query=query,
            patient_pdf_path=args.patient,
            output_name=args.output,
            include_references=include_refs,
            max_references=max_refs,
            enable_validation=args.validate,
            max_validation_cycles=args.cycles,
            show_statistics=show_stats,
            verbose=args.verbose
        )
        
        print(f"\n[SUCCESS] Document generated: {args.output}")
        
    except Exception as e:
        print(f"\n[ERROR] Document generation failed: {str(e)}")
        # ALWAYS print full traceback to identify the exact error location
        import traceback
        print("\n[FULL TRACEBACK]")
        traceback.print_exc()
        sys.exit(1)

def generate_document_direct(query: str, patient_pdf_path: str, output_name: str,
                           include_references: bool = True, max_references: int = 3,
                           enable_validation: bool = False, max_validation_cycles: int = 2,
                           show_statistics: bool = True, verbose: bool = False):
    """
    Generate a medical document directly without using the chat agent.
    
    Args:
        query: Query to retrieve guidelines (e.g., "Generér en plejeforløbsplan")
        patient_pdf_path: Path to patient PDF file
        output_name: Output filename for generated document
        include_references: Whether to include source references
        max_references: Maximum references per section
        enable_validation: Whether to enable two-stage subsection validation
        max_validation_cycles: Maximum validation cycles per subsection
        show_statistics: Whether to show generation statistics
        verbose: Show detailed progress information
    """
    
    if verbose:
        print(f"\n[STEP 1] Retrieving guidelines for: {query}")
    
    # Step 1: Retrieve guidelines directly
    guidelines = retrieve_guidelines_by_section(query)
    
    if isinstance(guidelines, str) and "No relevant guidelines found" in guidelines:
        raise Exception(f"No guidelines found for query: {query}")
    
    if verbose:
        sections = list(guidelines.keys())
        print(f"[SUCCESS] Retrieved {len(sections)} guideline sections:")
        for section in sections:
            print(f"  • {section}")
    
    # Step 2: Store guidelines in memory for generation process
    memory_manager.retrieved_guidelines = guidelines
    
    if verbose:
        print(f"\n[STEP 2] Initializing document generator")
        print(f"  • References: {'Enabled' if include_references else 'Disabled'}")
        if include_references:
            print(f"  • Max references per section: {max_references}")
        print(f"  • Two-stage validation: {'Enabled' if enable_validation else 'Disabled'}")
        if enable_validation:
            print(f"  • Max validation cycles: {max_validation_cycles}")
    
    # Step 3: Create document generator with specified settings
    doc_generator = DocumentGenerator(
        include_references=include_references,
        max_references_per_section=max_references,
        enable_subsection_validation=enable_validation,
        max_revision_cycles=max_validation_cycles
    )
    
    if verbose:
        print(f"\n[STEP 3] Generating document sections...")
    
    # Step 4: Generate the complete document
    # Settings are already configured in the DocumentGenerator instance
    doc_generator.generate_complete_document(
        guidelines=guidelines,
        patient_pdf_path=patient_pdf_path,
        output_name=output_name
    )
    
    # Step 5: Show statistics if requested
    if show_statistics:
        print(f"\n[STATISTICS] Document generation completed:")
        
        if include_references:
            stats = doc_generator.get_source_statistics()
            print(f"  • Total source references: {stats.get('total_sources', 0)}")
            print(f"  • Unique sources: {stats.get('unique_sources', 0)}")
            print(f"  • Average relevance: {stats.get('average_relevance', 0)}%")
            print(f"  • High relevance sources (≥80%): {stats.get('high_relevance_count', 0)}")
            
            type_dist = stats.get('type_distribution', {})
            if type_dist:
                print(f"  • Source types: {', '.join(f'{k}({v})' for k, v in type_dist.items())}")
        
        if enable_validation:
            quality_metrics = doc_generator.get_quality_metrics()
            if quality_metrics.get('validation_enabled'):
                overall = quality_metrics.get('overall_quality', {})
                print(f"  • Quality assurance:")
                print(f"    - Total validation cycles: {overall.get('total_cycles', 0)}")
                print(f"    - Fact-check revisions: {overall.get('fact_check_revisions', 0)}")
                print(f"    - Guideline revisions: {overall.get('guideline_revisions', 0)}")
                print(f"    - Sections with revisions: {overall.get('total_revisions', 0)}")
                print(f"    - Average cycles per section: {overall.get('avg_cycles_per_section', 0):.1f}")


if __name__ == "__main__":
    main()
