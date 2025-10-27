"""
Validation Report Logger
Tracks validation metrics and generates comprehensive PDF reports for each subsection.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


@dataclass
class ValidationMetrics:
    """Metrics for a single validation cycle"""
    cycle_number: int
    has_issues: bool
    issues_count: int
    temporal_claims: int
    static_claims: int
    context_hit_rate: float
    fallback_searches: int
    correction_applied: bool
    duration_seconds: float
    issues_by_status: Dict[str, int] = field(default_factory=dict)
    issues_by_confidence: Dict[str, int] = field(default_factory=dict)


@dataclass
class SubsectionReport:
    """Complete report for a subsection"""
    subsection_title: str
    section_title: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Generation metrics
    generation_duration: float = 0.0
    rag_calls_generation: int = 0
    
    # Validation cycles
    validation_cycles: List[ValidationMetrics] = field(default_factory=list)
    total_validation_duration: float = 0.0
    
    # Overall results
    final_issues_count: int = 0
    corrections_made: int = 0
    total_rag_calls: int = 0
    
    # Custom metrics (extensible)
    custom_metrics: Dict[str, any] = field(default_factory=dict)


class ValidationReportLogger:
    """
    Logger for tracking validation metrics and generating PDF reports.
    
    Designed to be extensible - you can add custom metrics to track anything.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.current_report: Optional[SubsectionReport] = None
        self.all_reports: List[SubsectionReport] = []
        
        # Timing helpers
        self._phase_start_time: Optional[float] = None
    
    def start_subsection(self, subsection_title: str, section_title: str) -> None:
        """
        Start tracking a new subsection.
        
        Args:
            subsection_title: Title of the subsection being generated
            section_title: Parent section title
        """
        self.current_report = SubsectionReport(
            subsection_title=subsection_title,
            section_title=section_title,
            start_time=datetime.now()
        )
        print(f"[REPORT] Started tracking: {subsection_title}")
    
    def end_subsection(self) -> None:
        """End tracking for current subsection."""
        if self.current_report:
            self.current_report.end_time = datetime.now()
            self.all_reports.append(self.current_report)
            print(f"[REPORT] Completed tracking: {self.current_report.subsection_title}")
            self.current_report = None
    
    def start_phase(self, phase_name: str) -> None:
        """
        Start timing a phase (generation, validation, etc).
        
        Args:
            phase_name: Name of the phase (for reference)
        """
        self._phase_start_time = time.time()
    
    def end_phase_generation(self, rag_calls: int = 0) -> None:
        """
        End generation phase and record metrics.
        
        Args:
            rag_calls: Number of RAG calls during generation
        """
        if self._phase_start_time and self.current_report:
            duration = time.time() - self._phase_start_time
            self.current_report.generation_duration = duration
            self.current_report.rag_calls_generation = rag_calls
            self._phase_start_time = None
    
    def log_validation_cycle(self, validation_result, cycle_number: int, 
                            correction_applied: bool, duration: float) -> None:
        """
        Log a complete validation cycle.
        
        Args:
            validation_result: ValidationResult from smart_validator
            cycle_number: Cycle number (1-indexed)
            correction_applied: Whether correction was applied
            duration: Duration of this cycle in seconds
        """
        if not self.current_report:
            return
        
        # Count issues by status and confidence
        issues_by_status = {}
        issues_by_confidence = {}
        
        for issue in validation_result.issues:
            status = issue.status
            confidence = issue.confidence
            issues_by_status[status] = issues_by_status.get(status, 0) + 1
            issues_by_confidence[confidence] = issues_by_confidence.get(confidence, 0) + 1
        
        # Extract temporal/static split (not directly available, estimate from context hits)
        temporal_claims = validation_result.fallback_searches - validation_result.context_hits
        static_claims = validation_result.context_hits + (len(validation_result.issues) - temporal_claims)
        
        metrics = ValidationMetrics(
            cycle_number=cycle_number,
            has_issues=validation_result.has_issues,
            issues_count=len(validation_result.issues),
            temporal_claims=max(0, temporal_claims),
            static_claims=max(0, static_claims),
            context_hit_rate=validation_result.context_hits / (validation_result.context_hits + validation_result.fallback_searches) if (validation_result.context_hits + validation_result.fallback_searches) > 0 else 0,
            fallback_searches=validation_result.fallback_searches,
            correction_applied=correction_applied,
            duration_seconds=duration,
            issues_by_status=issues_by_status,
            issues_by_confidence=issues_by_confidence
        )
        
        self.current_report.validation_cycles.append(metrics)
        self.current_report.total_validation_duration += duration
        
        if correction_applied:
            self.current_report.corrections_made += 1
    
    def log_custom_metric(self, key: str, value: any) -> None:
        """
        Log a custom metric (extensible).
        
        Args:
            key: Metric name
            value: Metric value (can be any type)
        """
        if self.current_report:
            self.current_report.custom_metrics[key] = value
    
    def finalize_report(self) -> None:
        """Calculate final metrics for current report."""
        if not self.current_report:
            return
        
        # Calculate totals
        self.current_report.final_issues_count = (
            self.current_report.validation_cycles[-1].issues_count 
            if self.current_report.validation_cycles 
            else 0
        )
        
        self.current_report.total_rag_calls = (
            self.current_report.rag_calls_generation +
            sum(cycle.fallback_searches for cycle in self.current_report.validation_cycles)
        )
    
    def generate_pdf_report(self, filename: str = "validation_report.pdf") -> None:
        """
        Generate comprehensive PDF report of all tracked subsections.
        
        Args:
            filename: Output PDF filename (default: validation_report.pdf)
        """
        if not self.all_reports:
            print("[REPORT] No reports to generate")
            return
        
        output_path = self.output_dir / filename
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=20
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=HexColor('#7f8c8d'),
            spaceAfter=8,
            spaceBefore=12
        )
        
        # Report header
        story.append(Paragraph("Validation Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            styles['Normal']
        ))
        story.append(Spacer(1, 20))
        
        # Executive summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        total_subsections = len(self.all_reports)
        total_cycles = sum(len(r.validation_cycles) for r in self.all_reports)
        total_corrections = sum(r.corrections_made for r in self.all_reports)
        total_rag_calls = sum(r.total_rag_calls for r in self.all_reports)
        total_time = sum(
            (r.generation_duration + r.total_validation_duration) 
            for r in self.all_reports
        )
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Subsections', str(total_subsections)],
            ['Total Validation Cycles', str(total_cycles)],
            ['Total Corrections Applied', str(total_corrections)],
            ['Total RAG Calls', str(total_rag_calls)],
            ['Total Processing Time', f"{total_time:.1f}s"],
            ['Average Time per Subsection', f"{total_time/total_subsections:.1f}s" if total_subsections > 0 else "N/A"]
        ]
        
        summary_table = Table(summary_data, colWidths=[8*cm, 8*cm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7'))
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 30))
        
        # Detailed subsection reports
        story.append(Paragraph("Subsection Details", heading_style))
        story.append(Spacer(1, 20))
        
        for report in self.all_reports:
            self._add_subsection_report(story, report, styles, subheading_style)
            story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        print(f"[REPORT] PDF generated: {output_path}")
    
    def _add_subsection_report(self, story, report: SubsectionReport, 
                               styles, subheading_style) -> None:
        """Add a single subsection report to the PDF story."""
        
        # Subsection header
        story.append(Paragraph(
            f"<b>{report.section_title}</b> → {report.subsection_title}", 
            subheading_style
        ))
        story.append(Spacer(1, 12))
        
        # Overview metrics
        overview_data = [
            ['Metric', 'Value'],
            ['Generation Time', f"{report.generation_duration:.2f}s"],
            ['RAG Calls (Generation)', str(report.rag_calls_generation)],
            ['Validation Time', f"{report.total_validation_duration:.2f}s"],
            ['Total RAG Calls', str(report.total_rag_calls)],
            ['Validation Cycles', str(len(report.validation_cycles))],
            ['Corrections Applied', str(report.corrections_made)],
            ['Final Issues', str(report.final_issues_count)]
        ]
        
        overview_table = Table(overview_data, colWidths=[8*cm, 8*cm])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#95a5a6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7'))
        ]))
        
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # Validation cycles
        if report.validation_cycles:
            story.append(Paragraph("Validation Cycles", styles['Heading4']))
            story.append(Spacer(1, 8))
            
            cycle_data = [['Cycle', 'Issues', 'Temporal', 'Static', 'Context Hit %', 
                          'RAG Calls', 'Duration', 'Corrected']]
            
            for cycle in report.validation_cycles:
                cycle_data.append([
                    str(cycle.cycle_number),
                    str(cycle.issues_count),
                    str(cycle.temporal_claims),
                    str(cycle.static_claims),
                    f"{cycle.context_hit_rate*100:.0f}%",
                    str(cycle.fallback_searches),
                    f"{cycle.duration_seconds:.1f}s",
                    '✓' if cycle.correction_applied else '✗'
                ])
            
            cycle_table = Table(cycle_data, colWidths=[1.5*cm, 1.5*cm, 2*cm, 2*cm, 
                                                      2.5*cm, 2*cm, 2*cm, 2*cm])
            cycle_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
                ('FONTSIZE', (0, 1), (-1, -1), 7)
            ]))
            
            story.append(cycle_table)
            story.append(Spacer(1, 20))
            
            # Issues breakdown (last cycle)
            last_cycle = report.validation_cycles[-1]
            if last_cycle.issues_by_status:
                story.append(Paragraph("Final Issues Breakdown", styles['Heading4']))
                story.append(Spacer(1, 8))
                
                status_data = [['Status', 'Count']]
                for status, count in last_cycle.issues_by_status.items():
                    status_data.append([status, str(count)])
                
                status_table = Table(status_data, colWidths=[8*cm, 4*cm])
                status_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e74c3c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7'))
                ]))
                
                story.append(status_table)
                story.append(Spacer(1, 12))
        
        # Custom metrics
        if report.custom_metrics:
            story.append(Paragraph("Custom Metrics", styles['Heading4']))
            story.append(Spacer(1, 8))
            
            custom_data = [['Metric', 'Value']]
            for key, value in report.custom_metrics.items():
                custom_data.append([key, str(value)])
            
            custom_table = Table(custom_data, colWidths=[8*cm, 8*cm])
            custom_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7'))
            ]))
            
            story.append(custom_table)


# Global logger instance
_report_logger: Optional[ValidationReportLogger] = None

def get_report_logger() -> ValidationReportLogger:
    """Get or create the global report logger instance."""
    global _report_logger
    if _report_logger is None:
        _report_logger = ValidationReportLogger()
    return _report_logger
