"""
NLP processing pipeline for text analysis.

Provides unified interface for spaCy processing, sentence segmentation,
tokenization, and linguistic feature extraction with enhanced semantic analysis.
"""

import logging
from typing import Any

try:
    import spacy
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Language = None

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class NLPPipeline:
    """Unified NLP processing pipeline with enhanced semantic analysis."""

    def __init__(
        self,
        language: str = "en",
        model_name: str | None = None,
        use_transformer: bool = True,
    ):
        """Initialize NLP pipeline with optional transformer-based models."""
        self.language = language
        self.use_transformer = use_transformer

        # Try transformer model first, fallback to small model
        if use_transformer:
            self.model_name = model_name or "en_core_web_trf"
        else:
            self.model_name = model_name or "en_core_web_sm"

        self.nlp: Language | None = None
        self.sentence_model: SentenceTransformer | None = None

        self._initialize_spacy()
        self._initialize_sentence_transformer()

    def _initialize_spacy(self) -> None:
        """Initialize spaCy model."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Install with: pip install spacy")
            return

        try:
            # Try to load the specified model
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(
                f"Model {self.model_name} not found. Trying fallback models."
            )

            # Try transformer model if small model was requested
            if self.model_name == "en_core_web_sm" and self.use_transformer:
                try:
                    self.nlp = spacy.load("en_core_web_trf")
                    self.model_name = "en_core_web_trf"
                    logger.info("Fallback to transformer model: en_core_web_trf")
                except OSError:
                    pass

            # Try small model if transformer was requested
            if not self.nlp:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.model_name = "en_core_web_sm"
                    logger.info("Fallback to small model: en_core_web_sm")
                except OSError:
                    logger.error(
                        "No spaCy models available. Some features will be limited."
                    )
                    self.nlp = None

        if self.nlp:
            # Add custom components if needed
            self._add_custom_components()

    def _initialize_sentence_transformer(self) -> None:
        """Initialize sentence transformer for semantic embeddings."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
            return

        try:
            # Use a lightweight but effective model
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence transformer: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None

    def _add_custom_components(self) -> None:
        """Add custom spaCy components for slop analysis."""
        if not self.nlp:
            return

        # Add sentence segmenter if not present
        if not self.nlp.has_pipe("parser"):
            self.nlp.add_pipe("sentencizer")

    def process(self, text: str) -> dict[str, Any] | None:
        """Process text through the NLP pipeline with semantic analysis."""
        if not self.nlp:
            return self._fallback_processing(text)

        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]

            # Get semantic embeddings if available
            sentence_embeddings = None
            if self.sentence_model and sentences:
                try:
                    sentence_embeddings = self.sentence_model.encode(sentences)
                except Exception as e:
                    logger.warning(f"Failed to generate sentence embeddings: {e}")

            return {
                "text": text,
                "sentences": sentences,
                "tokens": [token.text for token in doc],
                "lemmas": [token.lemma_ for token in doc],
                "pos_tags": [token.pos_ for token in doc],
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "sentence_embeddings": sentence_embeddings,
                "doc": doc,  # Keep reference to spaCy doc for advanced features
                "model_name": self.model_name,
                "has_transformer": "trf" in self.model_name,
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
            "sentence_embeddings": None,  # No embeddings
            "doc": None,
            "model_name": "fallback",
            "has_transformer": False,
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

    def get_sentence_embeddings(self, sentences: list[str]) -> list[list[float]] | None:
        """Get semantic embeddings for a list of sentences."""
        if not self.sentence_model:
            return None

        try:
            embeddings = self.sentence_model.encode(sentences)
            return embeddings.tolist()
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")
            return None

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.sentence_model:
            return 0.0

        try:
            embeddings = self.sentence_model.encode([text1, text2])
            # Calculate cosine similarity
            import numpy as np

            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to calculate semantic similarity: {e}")
            return 0.0

    def detect_semantic_drift(
        self, sentences: list[str], threshold: float = 0.7
    ) -> list[int]:
        """Detect sentences with significant semantic drift from previous sentence."""
        if not self.sentence_model or len(sentences) < 2:
            return []

        try:
            embeddings = self.sentence_model.encode(sentences)
            drift_points = []

            for i in range(1, len(embeddings)):
                # Calculate cosine similarity between consecutive sentences
                import numpy as np

                similarity = np.dot(embeddings[i - 1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
                )

                if similarity < threshold:
                    drift_points.append(i)

            return drift_points
        except Exception as e:
            logger.warning(f"Failed to detect semantic drift: {e}")
            return []

    def is_available(self) -> bool:
        """Check if NLP pipeline is properly initialized."""
        return self.nlp is not None

    def has_semantic_capabilities(self) -> bool:
        """Check if semantic analysis capabilities are available."""
        return self.sentence_model is not None
