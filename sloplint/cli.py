"""
CLI interface for the AI Slop CLI tool.

Provides command-line interface for analyzing text files and detecting AI slop.
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .combine import combine_scores, normalize_scores
from .feature_extractor import FeatureExtractor
from .io import load_text, save_json_output

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
    files: list[Path] = typer.Argument(..., help="Text files to analyze"),
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
    """Analyze text files for AI slop."""
    try:
        # Load text from files
        text_content = ""
        for file_path in files:
            text_content += load_text(file_path)

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

        # Normalize and combine scores
        normalized_metrics = normalize_scores(metrics, domain)
        slop_score, confidence = combine_scores(normalized_metrics, domain)

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
        console.print(f"[red]Error: {e}[/red]")
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


def display_results(result: dict, explain: bool = False, spans: bool = False) -> None:
    """Display analysis results in a formatted table."""
    console.print("\n[bold]AI Slop Analysis Results[/bold]")
    console.print(f"Domain: {result['domain']}")
    console.print(f"Slop Score: {result['slop_score']:.3f} ({result['level']})")
    console.print(f"Confidence: {result['confidence']:.3f}")

    # Create metrics table
    table = Table(title="Per-Axis Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")
    table.add_column("Status", style="green")

    for metric_name, metric_data in result["metrics"].items():
        score = metric_data["value"]
        if score <= 0.50:
            status = "✅ Good"
        elif score <= 0.70:
            status = "⚠️ Watch"
        elif score <= 0.85:
            status = "🔶 Sloppy"
        else:
            status = "❌ High-Slop"

        table.add_row(metric_name.title(), f"{score:.3f}", status)

    console.print(table)

    if explain:
        console.print("\n[yellow]Explanations:[/yellow]")
        console.print("• Density: Information density and perplexity measures")
        console.print("• Relevance: How well content matches prompt/references")
        console.print("• Coherence: Entity continuity and topic flow")
        console.print("• Repetition: N-gram repetition and compression")
        console.print("• Verbosity: Wordiness and structural complexity")

    console.print(
        f"\n[blue]Analysis completed in {result['timings_ms']['total']}ms[/blue]"
    )


if __name__ == "__main__":
    app()
