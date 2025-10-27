"""
Token Usage Tracking Utility

This module provides functionality to track and report token usage across LLM invocations.
Enabled when --verbose flag is used.
"""

import threading
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    """Represents token usage for a single LLM invocation"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    operation: str = "unknown"  # e.g., "fact_answering", "validation", "parsing"
    model: str = "unknown"


class TokenTracker:
    """
    Global token usage tracker with thread-safe operations.

    Usage:
        tracker = TokenTracker.get_instance()
        tracker.enable()  # Enable tracking
        tracker.record(prompt_tokens=100, completion_tokens=50, operation="fact_answering")
        report = tracker.get_report()
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.enabled = False
        self.usage_records: List[TokenUsage] = []
        self._records_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'TokenTracker':
        """Get singleton instance of TokenTracker (thread-safe)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def enable(self):
        """Enable token tracking"""
        self.enabled = True

    def disable(self):
        """Disable token tracking"""
        self.enabled = False

    def is_enabled(self) -> bool:
        """Check if tracking is enabled"""
        return self.enabled

    def record(self,
               prompt_tokens: int,
               completion_tokens: int,
               total_tokens: int = None,
               operation: str = "unknown",
               model: str = "unknown"):
        """
        Record token usage for an LLM invocation.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total tokens (calculated if not provided)
            operation: Type of operation (e.g., "fact_answering", "validation")
            model: Model name/identifier
        """
        if not self.enabled:
            return

        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            operation=operation,
            model=model
        )

        with self._records_lock:
            self.usage_records.append(usage)

    def get_statistics(self) -> Dict:
        """
        Get aggregated statistics about token usage.

        Returns:
            Dictionary with statistics including totals, averages, and breakdowns
        """
        with self._records_lock:
            if not self.usage_records:
                return {
                    "total_invocations": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "by_operation": {},
                    "by_model": {}
                }

            total_prompt = sum(r.prompt_tokens for r in self.usage_records)
            total_completion = sum(r.completion_tokens for r in self.usage_records)
            total_tokens = sum(r.total_tokens for r in self.usage_records)

            # Breakdown by operation
            by_operation = {}
            for record in self.usage_records:
                if record.operation not in by_operation:
                    by_operation[record.operation] = {
                        "count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                by_operation[record.operation]["count"] += 1
                by_operation[record.operation]["prompt_tokens"] += record.prompt_tokens
                by_operation[record.operation]["completion_tokens"] += record.completion_tokens
                by_operation[record.operation]["total_tokens"] += record.total_tokens

            # Breakdown by model
            by_model = {}
            for record in self.usage_records:
                if record.model not in by_model:
                    by_model[record.model] = {
                        "count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                by_model[record.model]["count"] += 1
                by_model[record.model]["prompt_tokens"] += record.prompt_tokens
                by_model[record.model]["completion_tokens"] += record.completion_tokens
                by_model[record.model]["total_tokens"] += record.total_tokens

            return {
                "total_invocations": len(self.usage_records),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "average_prompt_tokens": total_prompt / len(self.usage_records),
                "average_completion_tokens": total_completion / len(self.usage_records),
                "average_total_tokens": total_tokens / len(self.usage_records),
                "by_operation": by_operation,
                "by_model": by_model
            }

    def get_report(self, detailed: bool = True) -> str:
        """
        Generate a human-readable report of token usage.

        Args:
            detailed: If True, include breakdown by operation and model

        Returns:
            Formatted string report
        """
        stats = self.get_statistics()

        if stats["total_invocations"] == 0:
            return "\n[TOKEN USAGE] No LLM invocations tracked."

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("TOKEN USAGE REPORT")
        lines.append("=" * 70)

        # Overall statistics
        lines.append(f"\nTotal LLM Invocations: {stats['total_invocations']}")
        lines.append(f"Total Prompt Tokens:   {stats['total_prompt_tokens']:,}")
        lines.append(f"Total Completion Tokens: {stats['total_completion_tokens']:,}")
        lines.append(f"Total Tokens:          {stats['total_tokens']:,}")
        lines.append(f"\nAverage per Invocation:")
        lines.append(f"  Prompt Tokens:       {stats['average_prompt_tokens']:.1f}")
        lines.append(f"  Completion Tokens:   {stats['average_completion_tokens']:.1f}")
        lines.append(f"  Total Tokens:        {stats['average_total_tokens']:.1f}")

        if detailed:
            # Breakdown by operation
            if stats["by_operation"]:
                lines.append(f"\n{'-' * 70}")
                lines.append("BREAKDOWN BY OPERATION")
                lines.append("-" * 70)

                for operation, data in sorted(stats["by_operation"].items()):
                    lines.append(f"\n{operation}:")
                    lines.append(f"  Invocations:         {data['count']}")
                    lines.append(f"  Total Tokens:        {data['total_tokens']:,}")
                    lines.append(f"  Prompt Tokens:       {data['prompt_tokens']:,}")
                    lines.append(f"  Completion Tokens:   {data['completion_tokens']:,}")
                    avg_total = data['total_tokens'] / data['count'] if data['count'] > 0 else 0
                    lines.append(f"  Avg Tokens/Call:     {avg_total:.1f}")

            # Breakdown by model
            if stats["by_model"]:
                lines.append(f"\n{'-' * 70}")
                lines.append("BREAKDOWN BY MODEL")
                lines.append("-" * 70)

                for model, data in sorted(stats["by_model"].items()):
                    lines.append(f"\n{model}:")
                    lines.append(f"  Invocations:         {data['count']}")
                    lines.append(f"  Total Tokens:        {data['total_tokens']:,}")
                    lines.append(f"  Prompt Tokens:       {data['prompt_tokens']:,}")
                    lines.append(f"  Completion Tokens:   {data['completion_tokens']:,}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def reset(self):
        """Clear all recorded usage data"""
        with self._records_lock:
            self.usage_records.clear()

    def print_report(self, detailed: bool = True):
        """Print the token usage report to stdout"""
        print(self.get_report(detailed=detailed))


# Convenience function for global access
def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance"""
    return TokenTracker.get_instance()
