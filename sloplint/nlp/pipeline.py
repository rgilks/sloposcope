"""
NLP processing pipeline for text analysis.

Provides unified interface for spaCy processing, sentence segmentation,
tokenization, and linguistic feature extraction.
"""

import logging
from typing import Any

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)


class NLPPipeline:
    """Unified NLP processing pipeline."""

    def __init__(self, language: str = "en", model_name: str | None = None):
        """Initialize NLP pipeline."""
        self.language = language
        self.model_name = model_name or "en_core_web_sm"

        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Install with: pip install spacy")
            self.nlp = None
            return

        try:
            # Try to load the specified model
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(
                f"Model {self.model_name} not found. Using basic English model."
            )
            try:
                # Fallback to basic English model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error(
                    "No spaCy models available. Some features will be limited."
                )
                self.nlp = None

        if self.nlp:
            # Add custom components if needed
            self._add_custom_components()

    def _add_custom_components(self) -> None:
        """Add custom spaCy components for slop analysis."""
        if not self.nlp:
            return

        # Add sentence segmenter if not present
        if not self.nlp.has_pipe("parser"):
            self.nlp.add_pipe("sentencizer")

    def process(self, text: str) -> dict[str, Any] | None:
        """Process text through the NLP pipeline."""
        if not self.nlp:
            return self._fallback_processing(text)

        try:
            doc = self.nlp(text)

            return {
                "text": text,
                "sentences": [sent.text.strip() for sent in doc.sents],
                "tokens": [token.text for token in doc],
                "lemmas": [token.lemma_ for token in doc],
                "pos_tags": [token.pos_ for token in doc],
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "doc": doc,  # Keep reference to spaCy doc for advanced features
            }

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return self._fallback_processing(text)

    def _fallback_processing(self, text: str) -> dict[str, Any]:
        """Fallback processing when spaCy is not available."""
        logger.warning("Using fallback text processing")

        # Simple sentence splitting
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        # Simple tokenization
        tokens = text.split()

        return {
            "text": text,
            "sentences": sentences,
            "tokens": tokens,
            "lemmas": tokens,  # No lemmatization
            "pos_tags": ["UNK"] * len(tokens),  # No POS tagging
            "entities": [],  # No NER
            "doc": None,
        }

    def get_sentences(self, text: str) -> list[str]:
        """Extract sentences from text."""
        result = self.process(text)
        return result["sentences"] if result else []

    def get_tokens(self, text: str) -> list[str]:
        """Extract tokens from text."""
        result = self.process(text)
        return result["tokens"] if result else []

    def get_pos_tags(self, text: str) -> list[str]:
        """Extract POS tags from text."""
        result = self.process(text)
        return result["pos_tags"] if result else []

    def get_entities(self, text: str) -> list[tuple]:
        """Extract named entities from text."""
        result = self.process(text)
        return result["entities"] if result else []

    def is_available(self) -> bool:
        """Check if NLP pipeline is properly initialized."""
        return self.nlp is not None
