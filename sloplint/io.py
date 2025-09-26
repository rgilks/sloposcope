"""
Input/Output utilities for the AI Slop CLI.

Handles loading text from various sources and saving results in different formats.
"""

import json
import sys
from pathlib import Path
from typing import Any

import requests


def load_text(source: str | Path) -> str:
    """Load text from a file path, URL, or stdin."""
    if source == "-" or str(source).lower() == "stdin":
        return sys.stdin.read()

    path = Path(source)

    if path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            return normalize_text(content)
        except UnicodeDecodeError:
            # Try with different encodings
            try:
                with open(path, encoding="latin-1") as f:
                    content = f.read()
                return normalize_text(content)
            except Exception as e:
                raise ValueError(f"Could not read file {path}: {e}") from e
    else:
        # Try as URL
        return load_from_url(str(source))


def load_from_url(url: str) -> str:
    """Load text from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return normalize_text(response.text)
    except Exception as e:
        raise ValueError(f"Could not load from URL {url}: {e}") from e


def normalize_text(text: str) -> str:
    """Normalize text encoding and remove control characters."""
    import unicodedata

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters (keep newlines and tabs)
    normalized = ""
    for char in text:
        if unicodedata.category(char)[0] != "C" or char in "\n\r\t":
            normalized += char

    return normalized.strip()


def load_references(reference_paths: list[str | Path]) -> list[str]:
    """Load reference texts from multiple sources."""
    references = []
    for ref_path in reference_paths:
        ref_text = load_text(ref_path)
        if ref_text:
            references.append(ref_text)
    return references


def save_json_output(data: dict[str, Any], output_path: str | Path) -> None:
    """Save analysis results to JSON file."""
    path = Path(output_path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from TOML file."""
    import tomli

    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        with open(path, "rb") as f:
            return tomli.load(f)
    except Exception as e:
        raise ValueError(f"Could not load config file {path}: {e}") from e


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """Save configuration to TOML file."""
    import tomli_w

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        tomli_w.dump(config, f)
