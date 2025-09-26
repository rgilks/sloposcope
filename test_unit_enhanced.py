#!/usr/bin/env python3
"""
Unit tests for enhanced slop detection features without LocalStack dependencies.
"""

import pytest
from sloplint.feature_extractor import FeatureExtractor
from sloplint.combine import normalize_scores, combine_scores, get_slop_level
from sloplint.nlp.pipeline import NLPPipeline

def test_nlp_pipeline_basic():
    """Test basic NLP pipeline functionality."""
    pipeline = NLPPipeline()
    
    text = "This is a test sentence."
    result = pipeline.process(text)
    
    assert result is not None
    assert "sentences" in result
    assert "tokens" in result
    assert len(result["sentences"]) > 0
    assert len(result["tokens"]) > 0

def test_nlp_pipeline_semantic():
    """Test semantic capabilities."""
    pipeline = NLPPipeline()
    
    # Test semantic similarity
    text1 = "The cat sat on the mat."
    text2 = "A feline rested on the rug."
    similarity = pipeline.calculate_semantic_similarity(text1, text2)
    
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0
    
    # Test semantic drift detection
    sentences = [
        "The weather is nice today.",
        "I like to go for walks.",
        "Quantum computing uses qubits.",
        "Machine learning is fascinating."
    ]
    
    drift_points = pipeline.detect_semantic_drift(sentences)
    assert isinstance(drift_points, list)

def test_feature_extraction():
    """Test enhanced feature extraction."""
    extractor = FeatureExtractor()
    
    text = "This is a test of the enhanced slop detection system."
    features = extractor.extract_all_features(text)
    
    assert isinstance(features, dict)
    assert "density" in features
    assert "coherence" in features
    
    # Check if semantic features are present
    density_features = features["density"]
    assert "has_semantic_features" in density_features
    assert "semantic_density" in density_features
    assert "conceptual_density" in density_features

def test_score_combination():
    """Test score normalization and combination."""
    extractor = FeatureExtractor()
    
    text = "This is a test sentence for score combination."
    features = extractor.extract_all_features(text)
    
    normalized = normalize_scores(features, "general")
    score, confidence = combine_scores(normalized, "general")
    level = get_slop_level(score)
    
    assert isinstance(score, float)
    assert isinstance(confidence, float)
    assert isinstance(level, str)
    assert 0.0 <= score <= 1.0
    assert 0.0 <= confidence <= 1.0
    assert level in ["Clean", "Watch", "Sloppy", "High-Slop"]

def test_slop_levels():
    """Test slop level classification."""
    assert get_slop_level(0.2) == "Clean"
    assert get_slop_level(0.4) == "Watch"
    assert get_slop_level(0.6) == "Sloppy"
    assert get_slop_level(0.8) == "High-Slop"

def test_enhanced_density_features():
    """Test enhanced density features specifically."""
    extractor = FeatureExtractor()
    
    text = "This is a test with multiple sentences. Each sentence should be analyzed. The semantic density should be calculated."
    features = extractor.extract_all_features(text)
    
    density = features["density"]
    
    # Check semantic features
    assert "semantic_density" in density
    assert "conceptual_density" in density
    assert "has_semantic_features" in density
    
    # Verify semantic features are working
    if density["has_semantic_features"]:
        assert isinstance(density["semantic_density"], float)
        assert isinstance(density["conceptual_density"], float)
        assert 0.0 <= density["semantic_density"] <= 1.0
        assert 0.0 <= density["conceptual_density"] <= 1.0

def test_enhanced_coherence_features():
    """Test enhanced coherence features specifically."""
    extractor = FeatureExtractor()
    
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    features = extractor.extract_all_features(text)
    
    coherence = features["coherence"]
    
    # Check enhanced features
    assert "has_semantic_features" in coherence
    assert "entity_continuity" in coherence
    assert "embedding_drift" in coherence
    
    # Verify coherence features are working
    assert isinstance(coherence["entity_continuity"], float)
    assert isinstance(coherence["embedding_drift"], float)
    assert 0.0 <= coherence["entity_continuity"] <= 1.0
    assert 0.0 <= coherence["embedding_drift"] <= 1.0

if __name__ == "__main__":
    # Run tests directly
    test_functions = [
        test_nlp_pipeline_basic,
        test_nlp_pipeline_semantic,
        test_feature_extraction,
        test_score_combination,
        test_slop_levels,
        test_enhanced_density_features,
        test_enhanced_coherence_features,
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed.")
