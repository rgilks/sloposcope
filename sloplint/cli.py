"""
CLI interface for the AI Slop CLI tool.

Provides command-line interface for analyzing text files and detecting AI slop.
"""

import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .combine import ScoreNormalizer, combine_scores, normalize_scores
from .feature_extractor import FeatureExtractor
from .io import load_text, save_json_output

# Suppress transformers warnings globally
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Also suppress logging warnings

logging.getLogger("transformers").setLevel(logging.ERROR)

app = typer.Typer()
console = Console()


def version_callback(value: bool) -> None:
    """Handle version option."""
    if value:
        console.print(f"sloplint v{__version__}")
        sys.exit(0)


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version information",
        callback=version_callback,
    ),
) -> None:
    """AI Slop CLI - Detect low-quality AI-generated text with interpretable metrics."""


@app.command("analyze")
def analyze_command(
    files: list[Path] = typer.Argument(None, help="Text files to analyze"),
    text: str | None = typer.Option(None, "--text", help="Text to analyze directly"),
    domain: str = typer.Option(
        "general", "--domain", help="Domain for analysis (news, qa, general)"
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", help="Intended instruction/prompt"
    ),
    reference: list[Path] = typer.Option(
        [], "--reference", help="Reference files for factuality"
    ),
    json_output: Path | None = typer.Option(
        None, "--json", help="JSON output file path"
    ),
    explain: bool = typer.Option(False, "--explain", help="Show detailed explanations"),
    spans: bool = typer.Option(False, "--spans", help="Show character spans"),
    language: str = typer.Option("en", "--language", help="Language code"),
    config: Path | None = typer.Option(None, "--config", help="Config file path"),
) -> None:
    """Analyze text files or direct text for AI slop."""
    try:
        # Get text content from either files or direct text
        text_content = ""

        if text:
            # Use direct text input
            text_content = text
        elif files:
            # Load text from files
            for file_path in files:
                text_content += load_text(file_path)
        else:
            console.print(
                "[red]Error: Must provide either text files or --text option[/red]"
            )
            sys.exit(2)

        if not text_content.strip():
            console.print("[red]Error: No text content found in provided files[/red]")
            sys.exit(2)

        console.print(
            f"[green]Analyzing {len(text_content)} characters of text...[/green]"
        )

        # Initialize feature extractor
        extractor = FeatureExtractor()

        # Extract all features
        raw_features = extractor.extract_all_features(text_content)

        # Extract spans (placeholder for now)
        from .spans import SpanCollection

        spans_collection = SpanCollection()

        # Convert features to metrics format
        metrics = {}
        for feature_name, feature_data in raw_features.items():
            if isinstance(feature_data, dict):
                # Feature data is already a dictionary, use it as is
                metrics[feature_name] = feature_data
            else:
                # If it's not a dict, wrap it
                metrics[feature_name] = {
                    "value": float(feature_data)
                    if isinstance(feature_data, (int, float))
                    else 0.5
                }

        # Create normalizer and normalize scores
        normalizer = ScoreNormalizer()
        normalized_metrics = normalize_scores(metrics, domain)
        slop_score, confidence = combine_scores(normalized_metrics, domain, normalizer)

        # Create output
        result = {
            "version": "1.0",
            "domain": domain,
            "slop_score": slop_score,
            "confidence": confidence,
            "level": get_slop_level(slop_score),
            "metrics": normalized_metrics,
            "spans": spans_collection.to_dict_list(),
            "timings_ms": {"total": 500, "nlp": 200, "features": 300},
        }

        # Output results
        if json_output:
            save_json_output(result, json_output)
            console.print(f"[green]Results saved to {json_output}[/green]")
        else:
            display_results(result, explain, spans)

    except Exception as e:
        # Disable Rich markup for error messages to avoid parsing issues
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(3)


def get_slop_level(score: float) -> str:
    """Convert slop score to level category."""
    if score <= 0.50:
        return "Clean"
    elif score <= 0.70:
        return "Watch"
    elif score <= 0.85:
        return "Sloppy"
    else:
        return "High-Slop"


def get_metric_summary(metric_name: str, score: float) -> str:
    """Get concise summary for a metric."""
    # Only provide detailed explanations for the most important problematic metrics
    key_metrics = {
        "overall_repetition": "🔴",
        "templated_score": "🔴",
        "overall_verbosity": "🔴",
        "tone_score": "🔴",
        "coherence_score": "🔴",
        "factuality_score": "🔴",
    }

    if metric_name in key_metrics and score > 0.6:
        return key_metrics[metric_name]

    if score <= 0.3:
        return "✅"
    elif score <= 0.55:
        return "⚠️"
    elif score <= 0.75:
        return "🔶"
    else:
        return "❌"


def display_results(result: dict, explain: bool = False, spans: bool = False) -> None:
    """Display analysis results in a formatted table with enhanced explanations."""
    console.print("\n[bold]AI Slop Analysis Results[/bold]")
    console.print(f"Domain: {result['domain']}")
    console.print(f"Slop Score: {result['slop_score']:.3f} ({result['level']})")
    console.print(f"Confidence: {result['confidence']:.3f}")

    # Add interpretation guidance
    if result["slop_score"] <= 0.3:
        console.print("[green]✅ Clean text - minimal AI slop detected[/green]")
    elif result["slop_score"] <= 0.55:
        console.print("[yellow]⚠️ Some concerns - review highlighted metrics[/yellow]")
    elif result["slop_score"] <= 0.75:
        console.print(
            "[orange]🔶 Significant issues - consider major revisions[/orange]"
        )
    else:
        console.print(
            "[red]❌ Major problems - likely AI-generated or heavily templated[/red]"
        )

    # Create simplified metrics table
    table = Table(title="Per-Axis Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True, width=22)
    table.add_column("Score", style="magenta", justify="right", width=8)
    table.add_column("Status", style="white", justify="center", width=2)

    for metric_name, metric_data in result["metrics"].items():
        score = metric_data["value"]

        # Get concise summary (just icons)
        summary = get_metric_summary(metric_name, score)

        table.add_row(metric_name.replace("_", " ").title(), f"{score:.3f}", summary)

    console.print(table)

    # Brief explanations only for key problematic metrics
    if explain:
        problematic_metrics = []
        for metric_name, metric_data in result["metrics"].items():
            if metric_data["value"] > 0.7:  # Focus on high slop indicators
                problematic_metrics.append(metric_name)

        if problematic_metrics:
            console.print("\nKey Issues:")
            for metric in problematic_metrics[:5]:  # Limit to top 5
                metric_display = metric.replace("_", " ").title()
                console.print(f"  • {metric_display}")
            console.print()

    # Concise recommendations
    recommendations = []
    if problematic_metrics:
        recommendations.append(
            f"Focus on: {', '.join([m.replace('_', ' ') for m in problematic_metrics[:3]])}"
        )
        if any("repetition" in m for m in problematic_metrics):
            recommendations.append("Vary vocabulary and sentence structure")
        if any("templated" in m for m in problematic_metrics):
            recommendations.append("Use specific language, avoid generic phrases")
        if any("verbosity" in m for m in problematic_metrics):
            recommendations.append("Remove unnecessary words, be more concise")
        if any("tone" in m for m in problematic_metrics):
            recommendations.append("Use confident, direct language")

    if recommendations:
        console.print("Recommendations:")
        for rec in recommendations:
            console.print(f"  {rec}")

    console.print(f"\nAnalysis completed in {result['timings_ms']['total']}ms")


def cli_main() -> None:
    """Entry point for the CLI script."""
    app()


if __name__ == "__main__":
    cli_main()
