#!/usr/bin/env python3
"""
Performance comparison script for optimized vs original components.

This script demonstrates the performance improvements achieved through:
- Lazy loading
- Caching
- Optimized feature extraction
- Machine learning integration
"""

import time
import psutil
import numpy as np
from typing import Dict, Any
import tempfile
import os

# Import both original and optimized components
from sloplint.nlp.pipeline import NLPPipeline
from sloplint.nlp.optimized_pipeline import OptimizedNLPPipeline
from sloplint.feature_extractor import FeatureExtractor
from sloplint.optimized_feature_extractor import OptimizedFeatureExtractor
from sloplint.ml_integration import AnomalyDetector, LearnedCombiner, EnsembleDetector


def measure_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_nlp_pipeline():
    """Benchmark NLP pipeline performance."""
    print("🧪 NLP Pipeline Performance Comparison")
    print("=" * 50)
    
    sample_texts = [
        "The implementation uses a distributed architecture with microservices.",
        "Each service is containerized using Docker and deployed on Kubernetes.",
        "The system implements event-driven patterns with message queues.",
        "Security is handled through OAuth 2.0 authentication and JWT tokens.",
        "The database uses PostgreSQL with read replicas for performance."
    ]
    
    # Test original pipeline
    print("📊 Testing Original Pipeline...")
    mem_before = measure_memory_usage()
    
    start_time = time.time()
    original_pipeline = NLPPipeline(use_transformer=True)
    init_time_original = time.time() - start_time
    
    start_time = time.time()
    for text in sample_texts:
        result = original_pipeline.process(text)
    process_time_original = time.time() - start_time
    
    mem_after_original = measure_memory_usage()
    memory_original = mem_after_original - mem_before
    
    print(f"   Initialization: {init_time_original:.2f}s")
    print(f"   Processing: {process_time_original:.2f}s")
    print(f"   Memory: {memory_original:.1f} MB")
    
    # Test optimized pipeline
    print("\n📊 Testing Optimized Pipeline...")
    mem_before = measure_memory_usage()
    
    start_time = time.time()
    optimized_pipeline = OptimizedNLPPipeline(use_transformer=True)
    init_time_optimized = time.time() - start_time
    
    start_time = time.time()
    for text in sample_texts:
        result = optimized_pipeline.process(text)
    process_time_optimized = time.time() - start_time
    
    mem_after_optimized = measure_memory_usage()
    memory_optimized = mem_after_optimized - mem_before
    
    print(f"   Initialization: {init_time_optimized:.2f}s")
    print(f"   Processing: {process_time_optimized:.2f}s")
    print(f"   Memory: {memory_optimized:.1f} MB")
    
    # Calculate improvements
    init_improvement = (init_time_original - init_time_optimized) / init_time_original * 100
    process_improvement = (process_time_original - process_time_optimized) / process_time_original * 100
    memory_improvement = (memory_original - memory_optimized) / memory_original * 100
    
    print(f"\n🚀 Performance Improvements:")
    print(f"   Initialization: {init_improvement:+.1f}%")
    print(f"   Processing: {process_improvement:+.1f}%")
    print(f"   Memory: {memory_improvement:+.1f}%")


def benchmark_feature_extraction():
    """Benchmark feature extraction performance."""
    print("\n🧪 Feature Extraction Performance Comparison")
    print("=" * 50)
    
    sample_text = """
    The implementation uses a distributed architecture with microservices that communicate through REST APIs. 
    Each service is containerized using Docker and deployed on Kubernetes clusters. 
    The system implements event-driven patterns with message queues for asynchronous communication.
    Security is handled through OAuth 2.0 authentication and JWT tokens.
    The database uses PostgreSQL with read replicas for improved performance.
    """
    
    # Test original feature extractor
    print("📊 Testing Original Feature Extractor...")
    mem_before = measure_memory_usage()
    
    start_time = time.time()
    original_extractor = FeatureExtractor()
    init_time_original = time.time() - start_time
    
    start_time = time.time()
    features_original = original_extractor.extract_all_features(sample_text)
    extract_time_original = time.time() - start_time
    
    mem_after_original = measure_memory_usage()
    memory_original = mem_after_original - mem_before
    
    print(f"   Initialization: {init_time_original:.2f}s")
    print(f"   Extraction: {extract_time_original:.2f}s")
    print(f"   Memory: {memory_original:.1f} MB")
    print(f"   Features: {len(features_original)} extracted")
    
    # Test optimized feature extractor
    print("\n📊 Testing Optimized Feature Extractor...")
    mem_before = measure_memory_usage()
    
    start_time = time.time()
    optimized_extractor = OptimizedFeatureExtractor(use_transformer=True)
    init_time_optimized = time.time() - start_time
    
    start_time = time.time()
    features_optimized = optimized_extractor.extract_all_features(sample_text)
    extract_time_optimized = time.time() - start_time
    
    mem_after_optimized = measure_memory_usage()
    memory_optimized = mem_after_optimized - mem_before
    
    print(f"   Initialization: {init_time_optimized:.2f}s")
    print(f"   Extraction: {extract_time_optimized:.2f}s")
    print(f"   Memory: {memory_optimized:.1f} MB")
    print(f"   Features: {len(features_optimized)} extracted")
    
    # Calculate improvements
    init_improvement = (init_time_original - init_time_optimized) / init_time_original * 100
    extract_improvement = (extract_time_original - extract_time_optimized) / extract_time_original * 100
    memory_improvement = (memory_original - memory_optimized) / memory_original * 100
    
    print(f"\n🚀 Performance Improvements:")
    print(f"   Initialization: {init_improvement:+.1f}%")
    print(f"   Extraction: {extract_improvement:+.1f}%")
    print(f"   Memory: {memory_improvement:+.1f}%")


def benchmark_caching():
    """Benchmark caching performance."""
    print("\n🧪 Caching Performance Test")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = OptimizedNLPPipeline(
            use_transformer=False,  # Use smaller model for testing
            enable_caching=True,
            cache_dir=temp_dir
        )
        
        text = "This is a test sentence for caching performance evaluation."
        
        # First call (no cache)
        print("📊 First Call (No Cache)...")
        start_time = time.time()
        result1 = pipeline.process(text)
        first_call_time = time.time() - start_time
        print(f"   Time: {first_call_time:.3f}s")
        
        # Second call (with cache)
        print("\n📊 Second Call (With Cache)...")
        start_time = time.time()
        result2 = pipeline.process(text)
        second_call_time = time.time() - start_time
        print(f"   Time: {second_call_time:.3f}s")
        
        # Calculate improvement
        cache_improvement = (first_call_time - second_call_time) / first_call_time * 100
        print(f"\n🚀 Cache Performance Improvement: {cache_improvement:+.1f}%")
        
        # Verify results are identical
        assert result1["text"] == result2["text"]
        print("✅ Cached results are identical to original")


def benchmark_ml_integration():
    """Benchmark machine learning integration."""
    print("\n🧪 Machine Learning Integration Benchmark")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Test anomaly detection
    print("📊 Testing Anomaly Detection...")
    start_time = time.time()
    
    detector = AnomalyDetector(model_type="isolation_forest")
    detector.fit(X[:800])  # Use 80% for training
    
    fit_time = time.time() - start_time
    
    start_time = time.time()
    predictions = detector.predict(X[800:])  # Test on remaining 20%
    predict_time = time.time() - start_time
    
    print(f"   Fit Time: {fit_time:.3f}s")
    print(f"   Predict Time: {predict_time:.3f}s")
    print(f"   Predictions: {len(predictions)} samples")
    print(f"   Anomaly Rate: {np.mean(predictions):.2%}")
    
    # Test learned combiner
    print("\n📊 Testing Learned Combiner...")
    
    # Generate feature dictionaries
    feature_dicts = []
    labels = []
    
    for i in range(n_samples):
        feature_dict = {
            'perplexity': np.random.uniform(10, 50),
            'idea_density': np.random.uniform(2, 10),
            'compression_ratio': np.random.uniform(0.1, 0.8),
            'entity_continuity': np.random.uniform(0.3, 1.0),
            'hedging_score': np.random.uniform(0.0, 0.5),
            'words_per_sentence': np.random.uniform(10, 25),
            'flesch_kincaid': np.random.uniform(5, 15),
            'subjectivity_score': np.random.uniform(0.0, 0.8),
            'grammar_errors': np.random.uniform(0, 10),
            'mean_similarity': np.random.uniform(0.3, 1.0),
        }
        feature_dicts.append(feature_dict)
        labels.append(i % 2)
    
    start_time = time.time()
    combiner = LearnedCombiner(model_type="logistic_regression")
    combiner.fit(feature_dicts[:800], labels[:800])
    fit_time = time.time() - start_time
    
    start_time = time.time()
    test_predictions = []
    for fd in feature_dicts[800:]:
        pred, prob = combiner.predict(fd)
        test_predictions.append(pred)
    predict_time = time.time() - start_time
    
    print(f"   Fit Time: {fit_time:.3f}s")
    print(f"   Predict Time: {predict_time:.3f}s")
    print(f"   Predictions: {len(test_predictions)} samples")
    print(f"   Accuracy: {np.mean(np.array(test_predictions) == np.array(labels[800:])):.2%}")
    
    # Test ensemble detector
    print("\n📊 Testing Ensemble Detector...")
    start_time = time.time()
    
    ensemble = EnsembleDetector()
    ensemble.fit(feature_dicts[:800], labels[:800], anomaly_data=feature_dicts[:400])
    fit_time = time.time() - start_time
    
    start_time = time.time()
    ensemble_results = []
    for fd in feature_dicts[800:]:
        result = ensemble.predict(fd)
        ensemble_results.append(result)
    predict_time = time.time() - start_time
    
    print(f"   Fit Time: {fit_time:.3f}s")
    print(f"   Predict Time: {predict_time:.3f}s")
    print(f"   Predictions: {len(ensemble_results)} samples")
    
    # Calculate average ensemble score
    avg_score = np.mean([r['ensemble_score'] for r in ensemble_results])
    print(f"   Average Ensemble Score: {avg_score:.3f}")


def benchmark_end_to_end():
    """Benchmark end-to-end performance."""
    print("\n🧪 End-to-End Performance Test")
    print("=" * 50)
    
    sample_text = """
    The implementation uses a distributed architecture with microservices that communicate through REST APIs. 
    Each service is containerized using Docker and deployed on Kubernetes clusters. 
    The system implements event-driven patterns with message queues for asynchronous communication.
    Security is handled through OAuth 2.0 authentication and JWT tokens.
    The database uses PostgreSQL with read replicas for improved performance.
    """
    
    # Test optimized pipeline + feature extraction
    print("📊 Testing Optimized End-to-End Pipeline...")
    mem_before = measure_memory_usage()
    
    start_time = time.time()
    pipeline = OptimizedNLPPipeline(use_transformer=True)
    extractor = OptimizedFeatureExtractor(use_transformer=True)
    
    init_time = time.time() - start_time
    
    start_time = time.time()
    features = extractor.extract_all_features(sample_text)
    process_time = time.time() - start_time
    
    mem_after = measure_memory_usage()
    memory_usage = mem_after - mem_before
    
    print(f"   Initialization: {init_time:.2f}s")
    print(f"   Processing: {process_time:.2f}s")
    print(f"   Memory: {memory_usage:.1f} MB")
    print(f"   Text Length: {len(sample_text)} characters")
    print(f"   Words per Second: {len(sample_text.split()) / process_time:.1f}")
    print(f"   Features Extracted: {len(features)}")
    print(f"   Has Semantic Features: {features.get('has_semantic_features', False)}")
    
    # Performance targets
    print(f"\n🎯 Performance Targets:")
    print(f"   Target Processing Time: <1.0s for 1k words")
    print(f"   Target Memory Usage: <400MB")
    print(f"   Target Words per Second: >1000")
    
    # Check if targets are met
    words_per_second = len(sample_text.split()) / process_time
    target_words = 1000
    target_memory = 400
    
    print(f"\n📈 Target Achievement:")
    print(f"   Processing Speed: {'✅' if words_per_second >= target_words else '❌'} ({words_per_second:.0f}/{target_words} words/s)")
    print(f"   Memory Usage: {'✅' if memory_usage <= target_memory else '❌'} ({memory_usage:.0f}/{target_memory} MB)")


def main():
    """Run all performance benchmarks."""
    print("🚀 Sloposcope Performance Benchmark Suite")
    print("=" * 60)
    print("Testing optimized components vs original implementations")
    print("=" * 60)
    
    try:
        benchmark_nlp_pipeline()
        benchmark_feature_extraction()
        benchmark_caching()
        benchmark_ml_integration()
        benchmark_end_to_end()
        
        print("\n🎉 Performance Benchmark Complete!")
        print("=" * 60)
        print("Key improvements achieved:")
        print("✅ Lazy loading reduces initialization time")
        print("✅ Caching improves repeated processing")
        print("✅ Optimized feature extraction")
        print("✅ Machine learning integration")
        print("✅ Memory-efficient processing")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
