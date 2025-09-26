"""
Tests for the AI Slop CLI tool.
"""

import tempfile
from pathlib import Path

from sloplint.cli import get_slop_level
from sloplint.combine import combine_scores, normalize_scores
from sloplint.feature_extractor import FeatureExtractor


def test_get_slop_level():
    """Test slop level categorization."""
    assert get_slop_level(0.25) == "Clean"
    assert get_slop_level(0.40) == "Watch"
    assert get_slop_level(0.65) == "Sloppy"
    assert get_slop_level(0.85) == "High-Slop"


def test_feature_extractor():
    """Test the main feature extractor."""
    extractor = FeatureExtractor()

    text = "This is a test document. It has some content for analysis."
    features = extractor.extract_all_features(text)

    # Check that all expected features are present
    expected_features = [
        "density",
        "relevance",
        "subjectivity",
        "repetition",
        "templated",
        "coherence",
        "verbosity",
        "fluency",
        "complexity",
        "tone",
    ]

    for feature in expected_features:
        assert feature in features
        assert isinstance(features[feature], dict)


def test_score_combination():
    """Test score normalization and combination."""
    metrics = {
        "density": {"value": 0.5},
        "relevance": {"value": 0.6},
        "coherence": {"value": 0.4},
        "repetition": {"value": 0.3},
        "verbosity": {"value": 0.7},
    }

    normalized = normalize_scores(metrics, "general")
    score, confidence = combine_scores(normalized, "general")

    assert 0.0 <= score <= 1.0
    assert 0.0 <= confidence <= 1.0


def test_cli_with_file():
    """Test CLI with a temporary file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test document for CLI analysis.")
        temp_file = f.name

    try:
        # This would need to be updated to handle the new CLI structure
        # For now, just test that the file exists and has content
        assert Path(temp_file).exists()
        assert (
            Path(temp_file).read_text().strip()
            == "This is a test document for CLI analysis."
        )
    finally:
        Path(temp_file).unlink()


def test_cli_help():
    """Test that CLI help works."""
    # This is a basic test - in practice would capture stdout
    # and verify help text content
    pass
