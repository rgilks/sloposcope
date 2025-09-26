#!/usr/bin/env python3
"""
BERT-First Slop Detection CLI

This script provides a simple CLI for exceptional English slop detection
using BERT as the primary method, with feature-based fallback.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from sloplint.bert_slop_classifier import SlopDetectionPipeline

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("❌ BERT classifier not available. Please install transformers library.")
    sys.exit(1)


def analyze_text(text: str, use_bert: bool = True) -> Dict[str, Any]:
    """Analyze text for slop using BERT-first approach."""
    pipeline = SlopDetectionPipeline(use_feature_fallback=True)

    if use_bert:
        result = pipeline.analyze(text)
    else:
        # Fallback to feature-based analysis
        from sloplint.optimized_feature_extractor import OptimizedFeatureExtractor

        extractor = OptimizedFeatureExtractor(use_transformer=False)
        features = extractor.extract_all_features(text)

        # Simple slop score calculation
        slop_indicators = [
            features.get("repetition_score", 0),
            features.get("templated_score", 0),
            features.get("coherence_score", 0),
        ]
        slop_score = max(slop_indicators) if slop_indicators else 0

        result = {
            "text": text,
            "feature_analysis": {
                "slop_score": slop_score,
                "is_slop": slop_score > 0.6,
                "features": features,
            },
            "method": "feature_based",
        }

    return result


def format_result(result: Dict[str, Any], format_type: str = "human") -> str:
    """Format analysis result for output."""
    if format_type == "json":
        return json.dumps(result, indent=2)

    # Human-readable format
    output = []
    output.append("🔍 Slop Detection Analysis")
    output.append("=" * 50)

    if "bert_analysis" in result:
        bert = result["bert_analysis"]
        output.append(
            f"Method: BERT Classifier ({result.get('model_name', 'unknown')})"
        )
        output.append(f"Prediction: {'🚨 SLOP' if bert['is_slop'] else '✅ QUALITY'}")
        output.append(f"Confidence: {bert['confidence']:.2f}")
        output.append(f"Slop Score: {bert['slop_score']:.2f}")
        output.append(f"Quality Score: {bert['quality_score']:.2f}")
        output.append(f"Explanation: {bert['explanation']}")

        if "consensus" in result:
            consensus = result["consensus"]
            output.append(
                f"\nConsensus: {'✅ Agree' if consensus['agreement'] else '⚠️ Disagree'}"
            )
            output.append(
                f"BERT: {'SLOP' if consensus['bert_says_slop'] else 'QUALITY'}"
            )
            output.append(
                f"Features: {'SLOP' if consensus['features_say_slop'] else 'QUALITY'}"
            )

    elif "feature_analysis" in result:
        features = result["feature_analysis"]
        output.append(f"Method: Feature-Based Analysis")
        output.append(
            f"Prediction: {'🚨 SLOP' if features['is_slop'] else '✅ QUALITY'}"
        )
        output.append(f"Slop Score: {features['slop_score']:.2f}")

    return "\n".join(output)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="BERT-First Slop Detection for English Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bert_slop_cli.py "This is high-quality technical content."
  python bert_slop_cli.py --file document.txt
  python bert_slop_cli.py --json "AI-generated slop text here"
  python bert_slop_cli.py --features-only "Text to analyze"
        """,
    )

    parser.add_argument("text", nargs="?", help="Text to analyze for slop")

    parser.add_argument(
        "--file", "-f", type=str, help="File containing text to analyze"
    )

    parser.add_argument(
        "--json", "-j", action="store_true", help="Output results in JSON format"
    )

    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Use feature-based analysis instead of BERT",
    )

    parser.add_argument("--model-path", type=str, help="Path to fine-tuned BERT model")

    args = parser.parse_args()

    # Get input text
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        # Interactive mode
        print("Enter text to analyze (press Ctrl+D when done):")
        try:
            text = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

    if not text:
        print("❌ No text provided for analysis")
        sys.exit(1)

    # Analyze text
    try:
        print("🔍 Analyzing text...")
        result = analyze_text(text, use_bert=not args.features_only)

        # Format and output result
        output = format_result(result, format_type="json" if args.json else "human")
        print(output)

        # Exit with appropriate code
        if "bert_analysis" in result:
            is_slop = result["bert_analysis"]["is_slop"]
        elif "feature_analysis" in result:
            is_slop = result["feature_analysis"]["is_slop"]
        else:
            is_slop = False

        sys.exit(1 if is_slop else 0)  # Exit code 1 for slop, 0 for quality

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
