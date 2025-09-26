#!/usr/bin/env python3
"""
Direct test of enhanced slop detection features without LocalStack dependencies.

This test verifies that our enhanced features work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from sloplint.feature_extractor import FeatureExtractor
from sloplint.combine import normalize_scores, combine_scores, get_slop_level
from sloplint.nlp.pipeline import NLPPipeline

def test_nlp_pipeline():
    """Test the enhanced NLP pipeline."""
    print("🧪 Testing Enhanced NLP Pipeline")
    print("=" * 50)
    
    pipeline = NLPPipeline()
    
    # Test basic functionality
    text = "This is a test sentence. It has multiple sentences for testing."
    result = pipeline.process(text)
    
    if result:
        print(f"✅ NLP Pipeline working")
        print(f"   Sentences: {len(result['sentences'])}")
        print(f"   Tokens: {len(result['tokens'])}")
        print(f"   Has embeddings: {result['sentence_embeddings'] is not None}")
        print(f"   Model: {result['model_name']}")
        print(f"   Transformer: {result['has_transformer']}")
    else:
        print("❌ NLP Pipeline failed")
        return False
    
    return True

def test_feature_extraction():
    """Test enhanced feature extraction."""
    print("\n🧪 Testing Enhanced Feature Extraction")
    print("=" * 50)
    
    extractor = FeatureExtractor()
    
    test_texts = [
        {
            "name": "High-quality technical text",
            "text": "The implementation uses a distributed architecture with microservices. Each service handles specific business logic and communicates via REST APIs.",
            "expected_level": "Clean"
        },
        {
            "name": "AI-generated marketing text", 
            "text": "Discover the amazing benefits of our revolutionary product that will transform your life completely. Our innovative solution provides incredible results.",
            "expected_level": "Sloppy"
        },
        {
            "name": "Repetitive content",
            "text": "The system is very important. The system provides many benefits. The system is reliable and efficient.",
            "expected_level": "Sloppy"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_texts, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Text: {test_case['text'][:60]}...")
        
        try:
            # Extract features
            features = extractor.extract_all_features(test_case['text'])
            
            # Check if semantic features are available
            has_semantic = False
            if "density" in features and features["density"].get("has_semantic_features"):
                has_semantic = True
            if "coherence" in features and features["coherence"].get("has_semantic_features"):
                has_semantic = True
            
            # Normalize scores
            normalized = normalize_scores(features, "general")
            
            # Combine scores
            slop_score, confidence = combine_scores(normalized, "general")
            
            # Get slop level
            slop_level = get_slop_level(slop_score)
            
            print(f"   📊 Score: {slop_score:.3f}, Level: {slop_level}")
            print(f"   📈 Confidence: {confidence:.3f}")
            print(f"   🧠 Semantic Features: {'✅' if has_semantic else '❌'}")
            
            # Show enhanced features if available
            if has_semantic:
                if "density" in features:
                    density = features["density"]
                    print(f"   📊 Density - Semantic: {density.get('semantic_density', 0):.3f}, Conceptual: {density.get('conceptual_density', 0):.3f}")
                
                if "coherence" in features:
                    coherence = features["coherence"]
                    print(f"   🔗 Coherence - Entity: {coherence.get('entity_continuity', 0):.3f}, Drift: {coherence.get('embedding_drift', 0):.3f}")
            
            # Check if result matches expectation
            expected = test_case['expected_level']
            if slop_level == expected:
                print(f"   ✅ Correct prediction!")
            else:
                print(f"   ⚠️  Expected {expected}, got {slop_level}")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False
    
    return all_passed

def test_semantic_features():
    """Test specific semantic features."""
    print("\n🧪 Testing Semantic Features")
    print("=" * 50)
    
    pipeline = NLPPipeline()
    
    if not pipeline.has_semantic_capabilities():
        print("❌ Semantic capabilities not available")
        return False
    
    # Test semantic similarity
    text1 = "The cat sat on the mat."
    text2 = "A feline rested on the rug."
    similarity = pipeline.calculate_semantic_similarity(text1, text2)
    
    print(f"✅ Semantic similarity working")
    print(f"   Similarity between '{text1}' and '{text2}': {similarity:.3f}")
    
    # Test semantic drift detection
    sentences = [
        "The weather is nice today.",
        "I like to go for walks.",
        "Quantum computing uses qubits.",
        "Machine learning is fascinating."
    ]
    
    drift_points = pipeline.detect_semantic_drift(sentences, threshold=0.5)
    print(f"   Semantic drift points: {drift_points}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Enhanced Slop Detection Test Suite")
    print("=" * 60)
    
    tests = [
        ("NLP Pipeline", test_nlp_pipeline),
        ("Feature Extraction", test_feature_extraction), 
        ("Semantic Features", test_semantic_features),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} test PASSED")
            else:
                print(f"\n❌ {test_name} test FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} test ERROR: {e}")
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced slop detection is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
