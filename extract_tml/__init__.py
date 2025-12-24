"""Top-level package interface for extract_tml.

Expose the main API: extract_tags and a thin CLI entry.
"""
from .core import extract_tags, dedup_results  # re-export

__all__ = ["extract_tags", "dedup_results"]
