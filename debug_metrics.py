#!/usr/bin/env python3
"""
Debug script to see what metrics are being extracted and how they're processed.
"""

import sys
from sloplint.feature_extractor import FeatureExtractor
from sloplint.combine import normalize_scores, combine_scores, get_domain_weights


def debug_metrics():
    # Test with repetitive content
    test_text = "The project was successful. The team worked hard and the project was successful. Success was achieved through hard work. The successful outcome demonstrates success."

    print("🧪 DEBUGGING METRIC EXTRACTION")
    print(f"Test text: {test_text}")
    print()

    # Extract raw features
    extractor = FeatureExtractor()
    raw_features = extractor.extract_all_features(test_text)

    print("📊 RAW FEATURES EXTRACTED:")
    for metric_name, metric_data in raw_features.items():
        if isinstance(metric_data, dict):
            print(f"  {metric_name}:")
            for key, value in metric_data.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {metric_name}: {metric_data}")
    print()

    # Filter to just the main metrics (exclude metadata)
    main_metrics = {}
    for metric_name, metric_data in raw_features.items():
        if metric_name not in [
            "has_semantic_features",
            "model_name",
            "has_transformer",
            "processing_times",
            "total_processing_time",
        ]:
            if isinstance(metric_data, (int, float)):
                main_metrics[metric_name] = {"value": float(metric_data)}
            elif isinstance(metric_data, dict):
                # Handle complex metrics that return dictionaries
                if "value" in metric_data:
                    main_metrics[metric_name] = metric_data
                elif metric_name == "ngram_repetition":
                    # Special handling for ngram_repetition which has multiple values
                    main_metrics[metric_name] = metric_data
                else:
                    # Try to extract a single value from the dictionary
                    values = [
                        v for v in metric_data.values() if isinstance(v, (int, float))
                    ]
                    if values:
                        main_metrics[metric_name] = {"value": sum(values) / len(values)}
                    else:
                        main_metrics[metric_name] = {"value": 0.5}  # Default neutral
            elif isinstance(metric_data, list):
                main_metrics[metric_name] = {
                    "value": len(metric_data)
                }  # Use length as score
            else:
                main_metrics[metric_name] = {"value": 0.5}  # Default neutral

    print("🔍 MAIN METRICS FOR SCORING:")
    for metric_name, metric_data in main_metrics.items():
        print(f"  {metric_name}: {metric_data.get('value', 'N/A')}")
    print()

    # Normalize scores
    normalized_metrics = normalize_scores(main_metrics, "general")

    print("📈 NORMALIZED SCORES:")
    for metric_name, metric_data in normalized_metrics.items():
        print(f"  {metric_name}: {metric_data.get('value', 'N/A')}")
    print()

    # Combine scores
    final_score, confidence = combine_scores(normalized_metrics, "general")

    print("🎯 FINAL COMBINATION:")
    print(f"  Composite Score: {final_score}")
    print(f"  Confidence: {confidence}")

    # Show which metrics contributed most
    print("\n📋 TOP CONTRIBUTING METRICS:")
    contributions = []
    for metric_name, metric_data in normalized_metrics.items():
        score = metric_data.get("value", 0)
        contributions.append((metric_name, score))

    contributions.sort(key=lambda x: x[1], reverse=True)
    for metric_name, score in contributions[:5]:
        print(f"  {metric_name}: {score:.3f}")

    # Show domain weights being applied
    weights = get_domain_weights("general")
    print("\n⚖️ DOMAIN WEIGHTS APPLIED:")
    for dimension, weight in weights.items():
        print(f"  {dimension}: {weight}")

    # Show dimension mapping
    dimension_mapping = {
        "combined_density": "density",
        "perplexity_score": "density",
        "idea_density_score": "density",
        "semantic_density_score": "density",
        "conceptual_density_score": "density",
        "overall_repetition": "repetition",
        "ngram_repetition": "repetition",
        "sentence_repetition": "repetition",
        "compression_ratio": "repetition",
        "pattern_repetition": "repetition",
        "templated_score": "templated",
        "boilerplate_hits": "templated",
        "pos_diversity": "templated",
        "tone_score": "tone",
        "hedging_ratio": "tone",
        "sycophancy_ratio": "tone",
        "formality_ratio": "tone",
        "passive_ratio": "tone",
        "overall_verbosity": "verbosity",
        "words_per_sentence": "verbosity",
        "filler_ratio": "verbosity",
        "listiness": "verbosity",
        "sentence_variance": "verbosity",
        "coherence_score": "coherence",
        "entity_continuity": "coherence",
        "embedding_drift": "coherence",
        "relevance_score": "relevance",
        "mean_similarity": "relevance",
        "min_similarity": "relevance",
        "low_relevance_ratio": "relevance",
        "relevance_variance": "relevance",
        "factuality_score": "factuality",
        "unsupported_ratio": "factuality",
        "contradictions_count": "factuality",
        "subjectivity_score": "subjectivity",
        "subjective_ratio": "subjectivity",
        "bias_ratio": "subjectivity",
        "neutral_ratio": "subjectivity",
        "fluency_score": "fluency",
        "grammar_error_ratio": "fluency",
        "unnatural_phrase_ratio": "fluency",
        "fragment_ratio": "fluency",
        "complexity_score": "complexity",
        "complex_word_ratio": "complexity",
        "complex_phrase_ratio": "complexity",
        "flesch_kincaid_grade": "complexity",
    }
    print("\n🔗 DIMENSION MAPPING:")
    for metric, dimension in list(dimension_mapping.items())[:10]:  # Show first 10
        if metric in normalized_metrics:
            score = normalized_metrics[metric].get("value", 0)
            print(f"  {metric} -> {dimension}: {score:.3f}")


if __name__ == "__main__":
    debug_metrics()
