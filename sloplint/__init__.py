"""
AI Slop CLI - Detect low-quality AI-generated text with interpretable metrics.

This package provides tools to analyze text and detect "AI slop" - low-utility,
low-quality, or stylistically padded text common in unedited LLM outputs.
"""

__version__ = "0.1.0"
__author__ = "Rob Gilks"
__license__ = "Apache-2.0"

from . import cli, combine, io, nlp, spans

__all__ = ["cli", "combine", "io", "nlp", "spans"]
