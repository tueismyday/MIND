"""
Document generation workflow with two-stage subsection-level validation and source reference tracking.
Orchestrates document creation with detailed citations, traceability, and granular quality control.
"""

import Chroma_db_generated_document_final
from langchain_chroma import Chroma

from core.database import db_manager
from core.embeddings import get_embeddings
from generation.section_generator import generate_section_with_hybrid_approach
from utils.pdf_utils import save_to_pdf
from utils.text_processing import extract_final_section, assemble_final_document
from config.settings import GENERATED_DOCS_DB_DIR, DEFAULT_VALIDATION_CYCLES, MAX_SOURCES_PER_FACT

class DocumentGenerator:
    """ document generator with multi-fact retrieval and smart validation."""
    
    def __init__(self, include_references: bool = True, max_references_per_section: int = 3,
                 enable_subsection_validation: bool = True, max_revision_cycles: int = None,
                 use_hybrid_approach: bool = True): 
        """
        Initialize the  document generator.
        
        Args:
            include_references: Whether to include source references
            max_references_per_section: Maximum number of source references per section
            enable_subsection_validation: Whether to enable validation
            max_revision_cycles: Maximum number of revision cycles (deprecated with hybrid approach)
            use_hybrid_approach: Whether to use new hybrid multi-fact approach (default: True)
        """
        self.embeddings = get_embeddings()
        self.include_references = include_references
        self.max_references_per_section = max_references_per_section
        self.enable_subsection_validation = enable_subsection_validation
        self.max_revision_cycles = max_revision_cycles if max_revision_cycles is not None else DEFAULT_VALIDATION_CYCLES
        self.use_hybrid_approach = use_hybrid_approach 
        self.all_sources_used = []
        self.validation_report = {}
    
    def generate_complete_document(self, guidelines: dict, patient_pdf_path: str, 
                                 output_name: str, include_references: bool = None,
                                 max_references_per_section: int = None,
                                 enable_validation: bool = None,
                                 max_revision_cycles: int = None) -> None:
        """
        Generate a complete medical document with optional two-stage subsection-level validation and source references.
        
        Args:
            guidelines (dict): Dictionary of guideline sections
            patient_pdf_path (str): Path to the patient PDF file  
            output_name (str): Name for the output PDF file
            include_references (bool): Whether to include source references (overrides default)
            max_references_per_section (int): Max references per section (overrides default)
            enable_validation (bool): Whether to enable two-stage subsection validation (overrides default)
            max_revision_cycles (int): Max revision cycles per subsection (overrides default)
        """
        # Use provided parameters or fall back to instance defaults
        if include_references is None:
            include_references = self.include_references
        if max_references_per_section is None:
            max_references_per_section = self.max_references_per_section
        if enable_validation is None:
            enable_validation = self.enable_subsection_validation
        if max_revision_cycles is None:
            max_revision_cycles = self.max_revision_cycles
            
        print(f"[INFO] Starting document generation ...")
        print(f"  • References: {include_references}")
        print(f"  • Validation: {enable_validation}")
        
        # LEGACY, FIX, DATA NOT NEEDED ANYMORE
        patient_data = None
        
        # Generate each section
        generated_sections = {}
        self.all_sources_used = []
        self.validation_report = {
            'total_sections': len(guidelines),
            'sections': {},
            'summary': {},
            'validation_mode': 'two_stage' if enable_validation else 'disabled'
        }
        
        for i, (title, guideline) in enumerate(guidelines.items(), 1):
            print(f"[INFO] Creating section {i}/{len(guidelines)}: '{title}'")

            # Generate section using hybrid approach
            section_output, section_sources, validation_details = generate_section_with_hybrid_approach(
                section_title=title,
                section_guidelines=guideline,
                patient_data=patient_data,
                max_sources_per_fact=MAX_SOURCES_PER_FACT,
                enable_validation=enable_validation
            )
            
            # Store sources and validation details
            self.all_sources_used.extend(section_sources)
            self.validation_report['sections'][title] = validation_details
            
            print(f"[INFO] Section generated with {len(section_sources)} references")
            print("[RESULT] Section is completed!")
            
            # Extract the final section text
            final_section = extract_final_section(section_output)
            generated_sections[title] = final_section
            
        print("[RESULT] All sections have been completed!")
        
        # Generate validation summary if validation was enabled
        if enable_validation and self.validation_report['sections']:
            self._generate_validation_summary()
        
        # Create document with optional reference appendix and validation report
        if include_references and self.all_sources_used:
            final_document = self._assemble_document(generated_sections)
        else:
            final_document = assemble_final_document(generated_sections, self.validation_report.get('sections') if enable_validation else None)
        
        # Save the document
        save_to_pdf(final_document, output_name)

        # Save appendices to separate text file in /reports
        if include_references or enable_validation:
            self._save_appendices_to_file(output_name)
        
        
        # Index for future retrieval
        ###################### self.index_final_document()
        
        # Print comprehensive summary
        self._print_generation_summary(output_name, generated_sections, enable_validation, include_references)
    
    def _generate_validation_summary(self):
        """Generate summary statistics from two-stage validation details."""
        summary = {
            'total_subsections': 0,
            'total_validation_cycles': 0,
            'sections_with_revisions': 0,
            'average_cycles_per_section': 0,
            'fact_checking_revisions': 0,
            'guideline_revisions': 0,
            'quality_metrics': {}
        }
        
        for section_name, section_details in self.validation_report['sections'].items():
            if '_section_summary' in section_details:
                section_summary = section_details['_section_summary']
                
                # Safe gets with defaults
                summary['total_subsections'] += section_summary.get('total_subsections', 0)
                summary['total_validation_cycles'] += section_summary.get('total_validation_cycles', 0)
                
                # Check if any subsection in this section had revisions
                had_revisions = False
                fact_check_revisions = 0
                guideline_revisions = 0
                
                for subsection_name, subsection_details in section_details.items():
                    if not subsection_name.startswith('_'):
                        # Safe check for validation_history
                        validation_history = subsection_details.get('validation_history', [])
                        
                        for cycle in validation_history:
                            if cycle.get('fact_check_revision_applied', False):
                                fact_check_revisions += 1
                                had_revisions = True
                            if cycle.get('guideline_revision_applied', False):
                                guideline_revisions += 1
                                had_revisions = True
                
                if had_revisions:
                    summary['sections_with_revisions'] += 1
                
                summary['fact_checking_revisions'] += fact_check_revisions
                summary['guideline_revisions'] += guideline_revisions
        
        # Calculate averages (safe division)
        if self.validation_report['total_sections'] > 0:
            summary['average_cycles_per_section'] = summary['total_validation_cycles'] / self.validation_report['total_sections']
        
        # Store summary
        self.validation_report['summary'] = summary


    def _assemble_document(self, section_outputs: dict) -> str:
        """Assemble main document WITHOUT appendices (saved separately to /reports)."""
        main_document = assemble_final_document(section_outputs, self.validation_report.get('sections'))
        return main_document
    
    def _create_reference_appendix(self) -> list:
        """Create a comprehensive reference appendix."""
        
        appendix = []
        appendix.append("\n" + "="*50)
        appendix.append("KILDEHENVISNINGER OG DOKUMENTATION")
        appendix.append("="*50)
        
        # Group sources by type and date
        source_groups = {}
        for source in self.all_sources_used:
            entry_type = source.get('entry_type', 'Ukendt notetype')
            if entry_type not in source_groups:
                source_groups[entry_type] = []
            source_groups[entry_type].append(source)
        
        # Sort each group by date (newest first)
        for entry_type in source_groups:
            source_groups[entry_type].sort(
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )
        
        appendix.append(f"\nDette dokument er baseret på {len(self.all_sources_used)} kildehenvisninger fra patientjournalen.")
        appendix.append(f"Kilder fordelt på {len(source_groups)} forskellige notetyper:\n")
        
        # List sources by type
        for entry_type, sources in source_groups.items():
            appendix.append(f"### {entry_type.upper()} ({len(sources)} kilder)")
            
            # Deduplicate sources with same timestamp
            seen_timestamps = set()
            unique_sources = []
            for source in sources:
                timestamp = source.get('timestamp', '')
                if timestamp and timestamp not in seen_timestamps:
                    unique_sources.append(source)
                    seen_timestamps.add(timestamp)
            
            for source in unique_sources[:10]:  # Limit to top 10 per type
                timestamp = source.get('timestamp', 'Ukendt tidspunkt')
                relevance = source.get('relevance', 0)
                snippet = source.get('snippet', '')[:80]
                
                appendix.append(f"• **{timestamp}** (Relevans: {relevance}%)")
                if snippet:
                    appendix.append(f"  Uddrag: \"{snippet}...\"")
            
            if len(sources) > 10:
                appendix.append(f"  ... og {len(sources) - 10} flere kilder af denne type")
            appendix.append("")
        
        return appendix
    
    def _create_validation_appendix(self) -> list:
        """Create a two-stage validation quality report appendix."""
        
        appendix = []
        appendix.append("\n" + "="*50)
        appendix.append("KVALITETSSIKRINGSRAPPORT - TO-TRINS VALIDERING")
        appendix.append("="*50)
        
        summary = self.validation_report.get('summary', {})
        
        appendix.append(f"\nDette dokument er blevet kvalitetssikret gennem to-trins subsektions-niveau validering.")
        appendix.append(f"Valideringsstatistik:\n")
        
        # Overall statistics
        appendix.append(f"### OVERORDNET STATISTIK")
        appendix.append(f"• Totale sektioner: {self.validation_report['total_sections']}")
        appendix.append(f"• Totale undersektioner: {summary.get('total_subsections', 0)}")
        appendix.append(f"• Totale valideringscyklusser: {summary.get('total_validation_cycles', 0)}")
        appendix.append(f"• Sektioner med revisioner: {summary.get('sections_with_revisions', 0)}")
        appendix.append(f"• Gennemsnitlige cyklusser pr. sektion: {summary.get('average_cycles_per_section', 0):.1f}")
        appendix.append("")
        
        # Two-stage validation breakdown
        appendix.append("### TO-TRINS VALIDERINGSSTATISTIK")
        appendix.append(f"• Trin 1 - Faktatjek revisioner: {summary.get('fact_checking_revisions', 0)}")
        appendix.append(f"• Trin 2 - Retningslinjer revisioner: {summary.get('guideline_revisions', 0)}")
        appendix.append("")
        
        # Section-by-section breakdown
        appendix.append("### SEKTIONSVIS KVALITETSSIKRING")
        
        for section_name, section_details in self.validation_report['sections'].items():
            if '_section_summary' in section_details:
                section_summary = section_details['_section_summary']
                
                appendix.append(f"#### {section_name}")
                appendix.append(f"• Undersektioner: {section_summary['total_subsections']}")
                appendix.append(f"• Valideringscyklusser: {section_summary['total_validation_cycles']}")
                appendix.append(f"• Kilder: {section_summary['total_sources']}")
                
                # Count revisions by stage
                stage1_revisions = 0
                stage2_revisions = 0
                
                for subsection_name, subsection_details in section_details.items():
                    if not subsection_name.startswith('_'):
                        for cycle in subsection_details['validation_history']:
                            if cycle.get('fact_check_revision_applied', False):
                                stage1_revisions += 1
                            if cycle.get('guideline_revision_applied', False):
                                stage2_revisions += 1
                
                if stage1_revisions > 0 or stage2_revisions > 0:
                    appendix.append(f"• Faktatjek revisioner: {stage1_revisions}")
                    appendix.append(f"• Retningslinjer revisioner: {stage2_revisions}")
                
                appendix.append("")
        
        return appendix

    def _save_appendices_to_file(self, output_name: str):
        """Save references and validation appendices to separate text file in /reports."""
        import os
        from pathlib import Path
        
        # Create reports directory if it doesn't exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Create filename based on main document name
        base_name = Path(output_name).stem
        appendix_filename = reports_dir / f"{base_name}_appendices.txt"
        
        # Build appendix content
        appendix_parts = []
        
        if self.all_sources_used:
            appendix_parts.extend(self._create_reference_appendix())
        
        if isinstance(self.validation_report, dict) and self.validation_report.get('sections'):
            appendix_parts.extend(self._create_validation_appendix())

        print(f"[INFO] Appendices/validation report saved to {appendix_filename}")
        
    def _print_generation_summary(self, output_name: str, generated_sections: dict, 
                                enable_validation: bool, include_references: bool):
        """Print comprehensive generation summary."""
        
        print(f"\n[RESULT] Document generated successfully:")
        print(f"  • File: {output_name}")
        print(f"  • Sections: {len(generated_sections)}")
        
        if include_references:
            total_refs = len(self.all_sources_used)
            unique_refs = len(set(f"{s.get('entry_type', '')}-{s.get('timestamp', '')}" for s in self.all_sources_used))
            print(f"  • Total references: {total_refs}")
            print(f"  • Unique sources: {unique_refs}")
        
        if enable_validation and self.validation_report.get('summary'):
            summary = self.validation_report['summary']
            print(f"  • Two-stage quality assurance:")
            print(f"    - Subsections validated: {summary.get('total_subsections', 0)}")
            print(f"    - Validation cycles: {summary.get('total_validation_cycles', 0)}")
            print(f"    - Sections revised: {summary.get('sections_with_revisions', 0)}")
            print(f"    - Fact-check revisions: {summary.get('fact_checking_revisions', 0)}")
            print(f"    - Guideline revisions: {summary.get('guideline_revisions', 0)}")
            print(f"    - Avg cycles/section: {summary.get('average_cycles_per_section', 0):.1f}")
    
    def get_validation_report(self) -> dict:
        """Get detailed two-stage validation report."""
        return self.validation_report
    
    def get_source_statistics(self) -> dict:
        """Get statistics about sources used in document generation."""
        if not self.all_sources_used:
            return {"total_sources": 0}
        
        # Basic counts
        total_sources = len(self.all_sources_used)
        unique_sources = len(set(f"{s.get('entry_type', '')}-{s.get('timestamp', '')}" 
                               for s in self.all_sources_used))
        
        # Type distribution
        type_counts = {}
        for source in self.all_sources_used:
            entry_type = source.get('entry_type', 'Unknown')
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
        
        # Relevance distribution
        relevance_scores = [s.get('relevance', 0) for s in self.all_sources_used]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        return {
            "total_sources": total_sources,
            "unique_sources": unique_sources,
            "type_distribution": type_counts,
            "average_relevance": round(avg_relevance, 1),
            "high_relevance_count": sum(1 for r in relevance_scores if r >= 80)
        }
    
    def get_quality_metrics(self) -> dict:
        """Get quality metrics from two-stage validation process."""
        if not self.validation_report.get('sections'):
            return {"validation_enabled": False}
        
        metrics = {
            "validation_enabled": True,
            "validation_mode": "two_stage",
            "sections_analyzed": len(self.validation_report['sections']),
            "subsection_metrics": {},
            "overall_quality": {}
        }
        
        total_revisions = 0
        total_cycles = 0
        fact_check_revisions = 0
        guideline_revisions = 0
        
        for section_name, section_details in self.validation_report['sections'].items():
            section_metrics = {
                "subsections": 0,
                "revisions_made": 0,
                "fact_check_revisions": 0,
                "guideline_revisions": 0,
                "avg_cycles": 0
            }
            
            subsection_cycles = []
            for subsection_name, subsection_details in section_details.items():
                if not subsection_name.startswith('_'):
                    section_metrics["subsections"] += 1
                    cycles = len(subsection_details['validation_history'])
                    subsection_cycles.append(cycles)
                    total_cycles += cycles
                    
                    # Check for revisions by stage
                    had_revisions = False
                    for cycle in subsection_details['validation_history']:
                        if cycle.get('fact_check_revision_applied', False):
                            section_metrics["fact_check_revisions"] += 1
                            fact_check_revisions += 1
                            had_revisions = True
                        if cycle.get('guideline_revision_applied', False):
                            section_metrics["guideline_revisions"] += 1
                            guideline_revisions += 1
                            had_revisions = True
                    
                    if had_revisions:
                        section_metrics["revisions_made"] += 1
                        total_revisions += 1
            
            if subsection_cycles:
                section_metrics["avg_cycles"] = sum(subsection_cycles) / len(subsection_cycles)
            
            metrics["subsection_metrics"][section_name] = section_metrics
        
        # Overall quality metrics
        metrics["overall_quality"] = {
            "total_revisions": total_revisions,
            "total_cycles": total_cycles,
            "fact_check_revisions": fact_check_revisions,
            "guideline_revisions": guideline_revisions,
            "revision_rate": total_revisions / metrics["sections_analyzed"] if metrics["sections_analyzed"] > 0 else 0,
            "avg_cycles_per_section": total_cycles / metrics["sections_analyzed"] if metrics["sections_analyzed"] > 0 else 0,
            "fact_check_revision_rate": fact_check_revisions / total_cycles if total_cycles > 0 else 0,
            "guideline_revision_rate": guideline_revisions / total_cycles if total_cycles > 0 else 0
        }
        
        return metrics
    
    
    def index_final_document(self):
        """Index the generated document for future retrieval."""
        print("[INFO] Indexing generated document for future retrieval...")
        
        # Creating vectordatabase for the generated documents, enabling retrieval
        Chroma_db_generated_document_final.main()
        
        # Update the global database manager with the new generated docs DB
        db_manager._generated_docs_db = Chroma(
            persist_directory=str(GENERATED_DOCS_DB_DIR), 
            embedding_function=self.embeddings
        )
        
        print("[RESULT] Document has been indexed and is available for retrieval.")

            
