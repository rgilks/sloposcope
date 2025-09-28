"""
Optimized NLP processing pipeline with lazy loading and caching.

This module provides performance optimizations for the NLP pipeline including:
- Lazy model loading
- Caching of processed results
- Memory-efficient processing
- Batch processing capabilities
"""

import hashlib
import logging
import os
import pickle
import warnings
from typing import Any

try:
    import spacy
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    # Create a dummy Language type for type annotations
    from typing import Any

    Language = Any

try:
    # Set environment variables before importing to suppress warnings
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Create a dummy SentenceTransformer type for type annotations
    from typing import Any

    SentenceTransformer = Any

logger = logging.getLogger(__name__)

# Suppress sentence-transformers warnings at module level
warnings.filterwarnings("ignore", message=".*loss_type.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*You passed along.*", category=UserWarning)


class NLPPipeline:
    """Optimized NLP processing pipeline with lazy loading and caching."""

    def __init__(
        self,
        language: str = "en",
        model_name: str | None = None,
        use_transformer: bool = True,
        cache_dir: str | None = None,
        enable_caching: bool = True,
    ):
        """Initialize optimized NLP pipeline."""
        self.language = language
        self.use_transformer = use_transformer
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".nlp_cache")

        # Create cache directory if it doesn't exist
        if self.enable_caching and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Lazy loading - models will be loaded on first use
        self._nlp: Language | None = None
        self._sentence_model: SentenceTransformer | None = None
        self._model_name: str | None = None
        self._has_transformer: bool = False

        # Set model name - default to small model, transformer is optional
        if use_transformer and model_name:
            self._model_name = model_name
        else:
            self._model_name = model_name or "en_core_web_sm"

    @property
    def nlp(self) -> Language | None:
        """Lazy load spaCy model."""
        if self._nlp is None:
            self._nlp = self._load_spacy_model()
        return self._nlp

    @property
    def sentence_model(self) -> SentenceTransformer | None:
        """Lazy load sentence transformer model."""
        if self._sentence_model is None and self.use_transformer:
            self._sentence_model = self._load_sentence_transformer()
        return self._sentence_model

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name or "unknown"

    @property
    def has_transformer(self) -> bool:
        """Check if transformer model is available."""
        return self._has_transformer

    def _load_spacy_model(self) -> Language | None:
        """Load spaCy model with fallback strategy."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available")
            return None

        try:
            # Load the specified model
            nlp = spacy.load(self._model_name)
            self._has_transformer = "trf" in self._model_name
            logger.info(f"Loaded spaCy model: {self._model_name}")
            return nlp

        except OSError:
            logger.error(f"Model {self._model_name} not found")
            # Try fallback model if the current model name is not already the fallback
            if self._model_name != "en_core_web_sm":
                logger.info("Trying fallback model: en_core_web_sm")
                try:
                    nlp = spacy.load("en_core_web_sm")
                    self._has_transformer = False
                    self._model_name = "en_core_web_sm"
                    logger.info("Loaded fallback model: en_core_web_sm")
                    return nlp
                except OSError:
                    logger.error("Fallback model en_core_web_sm not available")
                    return None
            return None

        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            return None

    def _load_sentence_transformer(self) -> SentenceTransformer | None:
        """Load sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available")
            return None

        try:
            # Set logging level to suppress transformer warnings
            logging.getLogger("transformers").setLevel(logging.ERROR)

            # Capture any remaining stderr output
            from contextlib import redirect_stderr
            from io import StringIO

            stderr_capture = StringIO()
            with redirect_stderr(stderr_capture):
                model = SentenceTransformer("all-MiniLM-L6-v2")

            logger.info("Loaded sentence transformer model: all-MiniLM-L6-v2")
            return model
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            return None

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> dict[str, Any] | None:
        """Get cached result if available."""
        if not self.enable_caching:
            return None

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None

    def _cache_result(self, cache_key: str, result: dict[str, Any]) -> None:
        """Cache processing result."""
        if not self.enable_caching:
            return

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Error caching result: {e}")

    def process(self, text: str) -> dict[str, Any]:
        """Process text through the optimized NLP pipeline."""
        if not text.strip():
            return self._fallback_processing(text)

        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug("Using cached result")
            return cached_result

        # Process with spaCy
        nlp = self.nlp
        if not nlp:
            return self._fallback_processing(text)

        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]

            # Get semantic embeddings if available
            sentence_embeddings = None
            if self.sentence_model and sentences:
                try:
                    sentence_embeddings = self.sentence_model.encode(sentences)
                except Exception as e:
                    logger.warning(f"Failed to generate sentence embeddings: {e}")

            result = {
                "text": text,
                "sentences": sentences,
                "tokens": [token.text for token in doc],
                "lemmas": [token.lemma_ for token in doc],
                "pos_tags": [token.pos_ for token in doc],
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "sentence_embeddings": sentence_embeddings,
                "doc": doc,  # Keep reference to spaCy doc for advanced features
                "model_name": self.model_name,
                "has_transformer": self.has_transformer,
            }

            # Cache the result
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return self._fallback_processing(text)

    def _fallback_processing(self, text: str) -> dict[str, Any]:
        """Fallback processing when spaCy is not available."""
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "text": text,
            "sentences": sentences,
            "tokens": text.split(),
            "lemmas": text.split(),
            "pos_tags": ["UNK"] * len(text.split()),
            "entities": [],
            "sentence_embeddings": None,
            "doc": None,
            "model_name": "fallback",
            "has_transformer": False,
        }

    def batch_process(self, texts: list[str]) -> list[dict[str, Any]]:
        """Process multiple texts efficiently."""
        results = []
        for text in texts:
            results.append(self.process(text))
        return results

    def get_sentence_embeddings(self, sentences: list[str]) -> Any | None:
        """Get embeddings for a list of sentences."""
        if not self.sentence_model or not sentences:
            return None

        try:
            return self.sentence_model.encode(sentences)
        except Exception as e:
            logger.error(f"Error generating sentence embeddings: {e}")
            return None

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        embeddings1 = self.get_sentence_embeddings([text1])
        embeddings2 = self.get_sentence_embeddings([text2])

        if embeddings1 is None or embeddings2 is None:
            return 0.0

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            return float(similarity)
        except ImportError:
            logger.warning("scikit-learn not available for similarity calculation")
            return 0.0

    def detect_semantic_drift(
        self, sentences: list[str], threshold: float = 0.8
    ) -> list[int]:
        """Detect semantic drift points in a sequence of sentences."""
        if not self.sentence_model or len(sentences) < 2:
            return []

        try:
            embeddings = self.sentence_model.encode(sentences)
            drift_points = []

            for i in range(1, len(embeddings)):
                similarity = self.calculate_semantic_similarity(
                    sentences[i - 1], sentences[i]
                )
                if similarity < threshold:
                    drift_points.append(i)

            return drift_points
        except Exception as e:
            logger.error(f"Error detecting semantic drift: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear the processing cache."""
        if not self.enable_caching or not os.path.exists(self.cache_dir):
            return

        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self.enable_caching or not os.path.exists(self.cache_dir):
            return {"enabled": False, "files": 0, "size_mb": 0}

        try:
            files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) for f in files
            )
            return {
                "enabled": True,
                "files": len(files),
                "size_mb": total_size / (1024 * 1024),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": False, "files": 0, "size_mb": 0}
