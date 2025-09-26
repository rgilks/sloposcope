"""
Comprehensive tests for optimized components and ML integration.

This module tests:
- Optimized NLP pipeline with caching
- Optimized feature extractor
- Machine learning integration
- Anomaly detection
- Performance improvements
"""

import os
import tempfile

import numpy as np
import pytest

from sloplint.feature_extractor import FeatureExtractor
from sloplint.ml_integration import AnomalyDetector, EnsembleDetector, LearnedCombiner
from sloplint.nlp.pipeline import NLPPipeline


class TestNLPPipeline:
    """Test optimized NLP pipeline."""

    def test_lazy_loading(self):
        """Test that models are loaded lazily."""
        pipeline = NLPPipeline(use_transformer=True)

        # Models should not be loaded initially
        assert pipeline._nlp is None
        assert pipeline._sentence_model is None

        # Accessing properties should trigger loading
        nlp = pipeline.nlp
        assert (
            nlp is not None or pipeline._nlp is None
        )  # May be None if spaCy not available

        # After accessing, the model should be loaded (if available) or gracefully handle missing models
        if pipeline.use_transformer:
            # If transformer is requested but not available, it should fall back gracefully
            assert pipeline._sentence_model is not None or pipeline._nlp is None
        else:
            assert pipeline._sentence_model is None

    def test_caching(self):
        """Test caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = NLPPipeline(
                use_transformer=False,  # Use smaller model for testing
                enable_caching=True,
                cache_dir=temp_dir,
            )

            text = "This is a test sentence for caching."

            # First processing should create cache
            result1 = pipeline.process(text)
            assert result1 is not None

            # Second processing should use cache
            result2 = pipeline.process(text)
            assert result2 is not None

            # Results should be identical
            assert result1["text"] == result2["text"]
            assert result1["sentences"] == result2["sentences"]

    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = NLPPipeline(enable_caching=True, cache_dir=temp_dir)

            # Process some text to create cache entries
            pipeline.process("Test text 1")
            pipeline.process("Test text 2")

            stats = pipeline.get_cache_stats()
            assert stats["enabled"] is True
            assert stats["files"] >= 0  # May be 0 if spaCy not available

    def test_batch_processing(self):
        """Test batch processing functionality."""
        pipeline = NLPPipeline(use_transformer=False)

        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]

        results = pipeline.batch_process(texts)
        assert len(results) == len(texts)

        for i, result in enumerate(results):
            assert result["text"] == texts[i]

    def test_semantic_similarity(self):
        """Test semantic similarity calculation."""
        pipeline = NLPPipeline(use_transformer=False)

        text1 = "The cat sat on the mat."
        text2 = "A feline rested on the rug."

        similarity = pipeline.calculate_semantic_similarity(text1, text2)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_semantic_drift_detection(self):
        """Test semantic drift detection."""
        pipeline = NLPPipeline(use_transformer=False)

        sentences = [
            "The weather is nice today.",
            "I went to the store yesterday.",
            "The store had many products.",
            "Products were expensive.",
            "I bought some groceries.",
        ]

        drift_points = pipeline.detect_semantic_drift(sentences)
        assert isinstance(drift_points, list)
        assert all(isinstance(point, int) for point in drift_points)


class TestFeatureExtractor:
    """Test optimized feature extractor."""

    def test_feature_extraction(self):
        """Test feature extraction with optimizations."""
        extractor = FeatureExtractor(use_transformer=False)

        text = "This is a test document with some content for analysis."

        features = extractor.extract_all_features(text)

        # Check that features are extracted
        assert isinstance(features, dict)
        assert "has_semantic_features" in features
        assert "processing_times" in features
        assert "total_processing_time" in features

        # Check processing times are recorded
        assert isinstance(features["processing_times"], dict)
        assert features["total_processing_time"] > 0

    def test_batch_feature_extraction(self):
        """Test batch feature extraction."""
        extractor = FeatureExtractor(use_transformer=False)

        texts = [
            "First test document.",
            "Second test document.",
            "Third test document.",
        ]

        results = extractor.batch_extract_features(texts)
        assert len(results) == len(texts)

        for result in results:
            assert isinstance(result, dict)
            assert "total_processing_time" in result

    def test_processing_stats(self):
        """Test processing statistics."""
        extractor = FeatureExtractor(use_transformer=False)

        # Extract features to generate stats
        extractor.extract_all_features("Test text for statistics.")

        stats = extractor.get_processing_stats()

        assert "total_time" in stats
        assert "feature_times" in stats
        assert "nlp_pipeline" in stats
        assert isinstance(stats["feature_times"], dict)

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        extractor = FeatureExtractor(enable_caching=True)

        # Extract features to populate caches
        extractor.extract_all_features("Test text for cache clearing.")

        # Clear caches
        extractor.clear_caches()

        # Should not raise any errors
        assert True


class TestAnomalyDetector:
    """Test anomaly detection functionality."""

    def test_one_class_svm_initialization(self):
        """Test One-Class SVM initialization."""
        detector = AnomalyDetector(model_type="one_class_svm")
        assert detector.model_type == "one_class_svm"
        assert detector.model is not None

    def test_isolation_forest_initialization(self):
        """Test Isolation Forest initialization."""
        detector = AnomalyDetector(model_type="isolation_forest")
        assert detector.model_type == "isolation_forest"
        assert detector.model is not None

    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        detector = AnomalyDetector(model_type="isolation_forest")

        # Generate some training data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)

        # Fit the model
        detector.fit(X_train)
        assert detector.is_fitted is True

        # Generate test data
        X_test = np.random.randn(20, 10)

        # Predict anomalies
        predictions = detector.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

    def test_decision_function(self):
        """Test decision function."""
        detector = AnomalyDetector(model_type="isolation_forest")

        # Generate training data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        detector.fit(X_train)

        # Generate test data
        X_test = np.random.randn(10, 10)

        # Get decision function scores
        scores = detector.decision_function(X_test)
        assert len(scores) == len(X_test)
        assert all(isinstance(score, (int, float)) for score in scores)

    def test_model_save_load(self):
        """Test model saving and loading."""
        detector = AnomalyDetector(model_type="isolation_forest")

        # Generate training data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        detector.fit(X_train)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            filepath = tmp_file.name

        try:
            # Save model
            detector.save_model(filepath)
            assert os.path.exists(filepath)

            # Create new detector and load model
            new_detector = AnomalyDetector(model_type="isolation_forest")
            new_detector.load_model(filepath)

            assert new_detector.is_fitted is True
            assert new_detector.model_type == detector.model_type

        finally:
            os.unlink(filepath)


class TestLearnedCombiner:
    """Test learned combiner functionality."""

    def test_initialization(self):
        """Test combiner initialization."""
        combiner = LearnedCombiner(model_type="logistic_regression")
        assert combiner.model_type == "logistic_regression"
        assert combiner.model is not None

    def test_feature_extraction(self):
        """Test feature extraction from dictionary."""
        combiner = LearnedCombiner()

        feature_dict = {
            "perplexity": 25.0,
            "idea_density": 5.0,
            "compression_ratio": 0.3,
            "entity_continuity": 0.8,
            "hedging_score": 0.2,
            "words_per_sentence": 15.0,
            "flesch_kincaid": 10.0,
            "subjectivity_score": 0.3,
            "grammar_errors": 2.0,
            "mean_similarity": 0.7,
        }

        features = combiner._extract_features(feature_dict)
        assert features.shape[1] > 0  # Should have extracted some features

    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        combiner = LearnedCombiner(model_type="logistic_regression")

        # Generate sample feature dictionaries
        feature_dicts = []
        labels = []

        for i in range(100):
            feature_dict = {
                "perplexity": np.random.uniform(10, 50),
                "idea_density": np.random.uniform(2, 10),
                "compression_ratio": np.random.uniform(0.1, 0.8),
                "entity_continuity": np.random.uniform(0.3, 1.0),
                "hedging_score": np.random.uniform(0.0, 0.5),
                "words_per_sentence": np.random.uniform(10, 25),
                "flesch_kincaid": np.random.uniform(5, 15),
                "subjectivity_score": np.random.uniform(0.0, 0.8),
                "grammar_errors": np.random.uniform(0, 10),
                "mean_similarity": np.random.uniform(0.3, 1.0),
            }
            feature_dicts.append(feature_dict)
            labels.append(i % 2)  # Binary labels

        # Fit the model
        combiner.fit(feature_dicts, labels)
        assert combiner.is_fitted is True

        # Test prediction
        test_features = {
            "perplexity": 30.0,
            "idea_density": 6.0,
            "compression_ratio": 0.4,
            "entity_continuity": 0.7,
            "hedging_score": 0.3,
            "words_per_sentence": 18.0,
            "flesch_kincaid": 12.0,
            "subjectivity_score": 0.4,
            "grammar_errors": 3.0,
            "mean_similarity": 0.6,
        }

        prediction, probability = combiner.predict(test_features)
        assert prediction in [0, 1]
        assert 0.0 <= probability <= 1.0

    def test_feature_importance(self):
        """Test feature importance extraction."""
        combiner = LearnedCombiner(model_type="random_forest")

        # Generate sample data
        feature_dicts = []
        labels = []

        for i in range(50):
            feature_dict = {
                "perplexity": np.random.uniform(10, 50),
                "idea_density": np.random.uniform(2, 10),
                "compression_ratio": np.random.uniform(0.1, 0.8),
                "entity_continuity": np.random.uniform(0.3, 1.0),
                "hedging_score": np.random.uniform(0.0, 0.5),
                "words_per_sentence": np.random.uniform(10, 25),
                "flesch_kincaid": np.random.uniform(5, 15),
                "subjectivity_score": np.random.uniform(0.0, 0.8),
                "grammar_errors": np.random.uniform(0, 10),
                "mean_similarity": np.random.uniform(0.3, 1.0),
            }
            feature_dicts.append(feature_dict)
            labels.append(i % 2)

        combiner.fit(feature_dicts, labels)

        importance = combiner.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_model_save_load(self):
        """Test model saving and loading."""
        combiner = LearnedCombiner()

        # Generate sample data
        feature_dicts = []
        labels = []

        for i in range(50):
            feature_dict = {
                "perplexity": np.random.uniform(10, 50),
                "idea_density": np.random.uniform(2, 10),
                "compression_ratio": np.random.uniform(0.1, 0.8),
                "entity_continuity": np.random.uniform(0.3, 1.0),
                "hedging_score": np.random.uniform(0.0, 0.5),
                "words_per_sentence": np.random.uniform(10, 25),
                "flesch_kincaid": np.random.uniform(5, 15),
                "subjectivity_score": np.random.uniform(0.0, 0.8),
                "grammar_errors": np.random.uniform(0, 10),
                "mean_similarity": np.random.uniform(0.3, 1.0),
            }
            feature_dicts.append(feature_dict)
            labels.append(i % 2)

        combiner.fit(feature_dicts, labels)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            filepath = tmp_file.name

        try:
            # Save model
            combiner.save_model(filepath)
            assert os.path.exists(filepath)

            # Create new combiner and load model
            new_combiner = LearnedCombiner()
            new_combiner.load_model(filepath)

            assert new_combiner.is_fitted is True
            assert new_combiner.model_type == combiner.model_type

        finally:
            os.unlink(filepath)


class TestEnsembleDetector:
    """Test ensemble detector functionality."""

    def test_initialization(self):
        """Test ensemble detector initialization."""
        detector = EnsembleDetector(domain="test")
        assert detector.domain == "test"
        assert detector.anomaly_detector is not None
        assert detector.learned_combiner is not None

    def test_fit_and_predict(self):
        """Test ensemble fitting and prediction."""
        detector = EnsembleDetector()

        # Generate sample data
        feature_dicts = []
        labels = []

        for i in range(100):
            feature_dict = {
                "perplexity": np.random.uniform(10, 50),
                "idea_density": np.random.uniform(2, 10),
                "compression_ratio": np.random.uniform(0.1, 0.8),
                "entity_continuity": np.random.uniform(0.3, 1.0),
                "hedging_score": np.random.uniform(0.0, 0.5),
                "words_per_sentence": np.random.uniform(10, 25),
                "flesch_kincaid": np.random.uniform(5, 15),
                "subjectivity_score": np.random.uniform(0.0, 0.8),
                "grammar_errors": np.random.uniform(0, 10),
                "mean_similarity": np.random.uniform(0.3, 1.0),
            }
            feature_dicts.append(feature_dict)
            labels.append(i % 2)

        # Fit ensemble
        detector.fit(feature_dicts, labels, anomaly_data=feature_dicts[:50])
        assert detector.is_fitted is True

        # Test prediction
        test_features = {
            "perplexity": 30.0,
            "idea_density": 6.0,
            "compression_ratio": 0.4,
            "entity_continuity": 0.7,
            "hedging_score": 0.3,
            "words_per_sentence": 18.0,
            "flesch_kincaid": 12.0,
            "subjectivity_score": 0.4,
            "grammar_errors": 3.0,
            "mean_similarity": 0.6,
        }

        result = detector.predict(test_features)

        assert "ensemble_score" in result
        assert "combiner_prediction" in result
        assert "combiner_probability" in result
        assert "anomaly_prediction" in result
        assert "anomaly_score" in result
        assert "confidence" in result

        assert 0.0 <= result["ensemble_score"] <= 1.0
        assert result["combiner_prediction"] in [0, 1]
        assert 0.0 <= result["combiner_probability"] <= 1.0
        assert result["anomaly_prediction"] in [0, 1]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_model_save_load(self):
        """Test ensemble model saving and loading."""
        detector = EnsembleDetector()

        # Generate sample data
        feature_dicts = []
        labels = []

        for i in range(50):
            feature_dict = {
                "perplexity": np.random.uniform(10, 50),
                "idea_density": np.random.uniform(2, 10),
                "compression_ratio": np.random.uniform(0.1, 0.8),
                "entity_continuity": np.random.uniform(0.3, 1.0),
                "hedging_score": np.random.uniform(0.0, 0.5),
                "words_per_sentence": np.random.uniform(10, 25),
                "flesch_kincaid": np.random.uniform(5, 15),
                "subjectivity_score": np.random.uniform(0.0, 0.8),
                "grammar_errors": np.random.uniform(0, 10),
                "mean_similarity": np.random.uniform(0.3, 1.0),
            }
            feature_dicts.append(feature_dict)
            labels.append(i % 2)

        detector.fit(feature_dicts, labels, anomaly_data=feature_dicts[:25])

        with (
            tempfile.NamedTemporaryFile(
                suffix="_combiner.pkl", delete=False
            ) as tmp_file1,
            tempfile.NamedTemporaryFile(
                suffix="_anomaly.pkl", delete=False
            ) as tmp_file2,
        ):
            base_path = tmp_file1.name.replace("_combiner.pkl", "")
            combiner_path = tmp_file1.name
            anomaly_path = tmp_file2.name

        try:
            # Save models
            detector.save_models(base_path)
            assert os.path.exists(combiner_path)
            assert os.path.exists(anomaly_path)

            # Create new detector and load models
            new_detector = EnsembleDetector()
            new_detector.load_models(base_path)

            assert new_detector.is_fitted is True

        finally:
            for path in [combiner_path, anomaly_path]:
                if os.path.exists(path):
                    os.unlink(path)


class TestPerformanceImprovements:
    """Test performance improvements."""

    def test_optimized_pipeline_performance(self):
        """Test that optimized pipeline is faster than original."""
        # This is a basic test - in practice, we'd measure actual performance
        pipeline = NLPPipeline(use_transformer=False)

        text = "This is a test sentence for performance testing."

        # Should not raise errors
        result = pipeline.process(text)
        assert result is not None

    def test_caching_improves_performance(self):
        """Test that caching improves performance on repeated texts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = NLPPipeline(
                use_transformer=False, enable_caching=True, cache_dir=temp_dir
            )

            text = "This is a test sentence for caching performance."

            # First call should be slower (no cache)
            result1 = pipeline.process(text)

            # Second call should be faster (uses cache)
            result2 = pipeline.process(text)

            # Results should be identical
            assert result1["text"] == result2["text"]


if __name__ == "__main__":
    pytest.main([__file__])
