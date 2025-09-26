"""
Feature extractors for AI slop analysis.

This package contains all the per-axis metric calculators for detecting AI slop.
"""

from .repetition import extract_features as extract_repetition_features
from .verbosity import extract_features as extract_verbosity_features
from .density import extract_features as extract_density_features
from .coherence import extract_features as extract_coherence_features
from .templated import extract_features as extract_templated_features
from .tone import extract_features as extract_tone_features

__all__ = [
    "extract_repetition_features",
    "extract_verbosity_features",
    "extract_density_features",
    "extract_coherence_features",
    "extract_templated_features",
    "extract_tone_features",
]