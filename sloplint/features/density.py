"""
Density analysis for AI slop detection.

Calculates information density using perplexity from language models,
semantic density using embeddings, and idea density based on linguistic features.
"""

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    logger.warning(
        "Transformers not available. Perplexity calculation will be limited."
    )


class DensityCalculator:
    """Calculates information density metrics."""

    def __init__(self, device: str = "auto") -> None:
        """Initialize with pre-trained language model."""
        self.model_name = "distilgpt2"
        self.tokenizer = None
        self.model = None

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU acceleration")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple Metal GPU acceleration")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU (no GPU available)")
        else:
            self.device = torch.device(device)
            logger.info(f"Using specified device: {device}")

        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Loading DistilGPT-2 for perplexity calculation...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)

                # Set padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                logger.info(f"✅ DistilGPT-2 loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load DistilGPT-2: {e}")
                self.model = None
                self.tokenizer = None
        else:
            logger.error("Transformers not available for perplexity calculation")

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity using DistilGPT-2."""
        if not self.model or not self.tokenizer:
            # Fallback: estimate based on text length
            return self._fallback_perplexity(text)

        try:
            # Tokenize text
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            # Convert loss to perplexity
            perplexity = float(math.exp(loss.item()))

            return perplexity

        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return self._fallback_perplexity(text)

    def _fallback_perplexity(self, text: str) -> float:
        """Fallback perplexity calculation when model is not available."""
        # Simple approximation based on text complexity
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)

        # Estimate perplexity based on word length and diversity
        # This is a very rough approximation
        base_perplexity = 20.0  # Base perplexity for English text
        complexity_factor = avg_word_length / 5.0  # Normalize around 5-letter words

        return base_perplexity * complexity_factor

    def calculate_idea_density(
        self, sentences: list[str], pos_tags: list[str]
    ) -> float:
        """Calculate idea density (propositions per 100 words)."""
        if not sentences:
            return 0.0

        # Simple heuristics for idea density
        total_words = sum(len(sentence.split()) for sentence in sentences)

        if total_words == 0:
            return 0.0

        # Count predicates (verbs) and complex noun phrases as idea indicators
        idea_indicators: float = 0.0

        # This is a simplified version - in practice would use dependency parsing
        for sentence in sentences:
            words = sentence.split()
            # Look for verbs and content words
            for i, word in enumerate(words):
                if (
                    word.lower() in ["is", "are", "was", "were", "be", "been", "being"]
                    and i < len(words) - 1
                ):
                    # Potential predicate
                    idea_indicators += 0.5  # Partial credit for copula
                elif word.endswith("ing") or word.endswith("ed"):
                    # Potential verb form
                    idea_indicators += 0.8

        # Normalize to propositions per 100 words
        idea_density = (idea_indicators / total_words) * 100

        return min(idea_density, 20.0)  # Cap at reasonable maximum

    def calculate_semantic_density(
        self, sentence_embeddings: np.ndarray | None
    ) -> float:
        """Calculate semantic density using sentence embeddings."""
        if sentence_embeddings is None or len(sentence_embeddings) == 0:
            return 0.0

        try:
            # Calculate semantic diversity as the average pairwise distance between sentences
            if len(sentence_embeddings) == 1:
                return 0.5  # Single sentence, moderate density

            # Calculate pairwise cosine distances
            distances = []
            for i in range(len(sentence_embeddings)):
                for j in range(i + 1, len(sentence_embeddings)):
                    # Cosine similarity
                    similarity = np.dot(
                        sentence_embeddings[i], sentence_embeddings[j]
                    ) / (
                        np.linalg.norm(sentence_embeddings[i])
                        * np.linalg.norm(sentence_embeddings[j])
                    )
                    # Convert to distance (1 - similarity)
                    distance = 1 - similarity
                    distances.append(distance)

            if not distances:
                return 0.5

            # Higher average distance = higher semantic diversity = higher density
            avg_distance = np.mean(distances)

            # Normalize to [0, 1] range
            semantic_density = min(
                avg_distance * 1.5, 1.0
            )  # Reduced scale factor for better normalization

            return float(semantic_density)

        except Exception as e:
            logger.warning(f"Error calculating semantic density: {e}")
            return 0.5

    def calculate_conceptual_density(
        self, sentences: list[str], sentence_embeddings: np.ndarray | None
    ) -> float:
        """Calculate conceptual density based on semantic clustering."""
        if (
            not sentences
            or sentence_embeddings is None
            or len(sentence_embeddings) == 0
        ):
            return 0.0

        try:
            if len(sentence_embeddings) < 2:
                return 0.5  # Single sentence, moderate density

            # Simple clustering-based density calculation
            # Group sentences by semantic similarity
            clusters = []
            used_indices = set()

            for i, embedding in enumerate(sentence_embeddings):
                if i in used_indices:
                    continue

                cluster = [i]
                used_indices.add(i)

                # Find similar sentences
                for j, other_embedding in enumerate(sentence_embeddings):
                    if j in used_indices or i == j:
                        continue

                    similarity = np.dot(embedding, other_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
                    )

                    if similarity > 0.7:  # High similarity threshold
                        cluster.append(j)
                        used_indices.add(j)

                clusters.append(cluster)

            # Calculate density based on cluster distribution
            # More clusters with fewer sentences = higher conceptual density
            if not clusters:
                return 0.5

            avg_cluster_size = len(sentences) / len(clusters)
            # Normalize: smaller clusters = higher density
            # More conservative normalization
            conceptual_density = max(0.0, min(1.0, 1.0 - (avg_cluster_size - 1) / 3.0))

            return float(conceptual_density)

        except Exception as e:
            logger.warning(f"Error calculating conceptual density: {e}")
            return 0.5


def extract_features(
    text: str,
    sentences: list[str],
    tokens: list[str],
    pos_tags: list[str] | None = None,
    sentence_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract all density-related features including semantic analysis."""
    try:
        calculator = DensityCalculator()

        # Calculate perplexity
        perplexity = calculator.calculate_perplexity(text)

        # Calculate idea density
        idea_density = calculator.calculate_idea_density(sentences, pos_tags or [])

        # Calculate semantic density if embeddings are available
        semantic_density = 0.0
        conceptual_density = 0.0

        if sentence_embeddings is not None:
            semantic_density = calculator.calculate_semantic_density(
                sentence_embeddings
            )
            conceptual_density = calculator.calculate_conceptual_density(
                sentences, sentence_embeddings
            )

        # Calculate combined density score
        # Lower perplexity = higher information density
        # Higher idea density = higher information density
        if perplexity > 0:
            perplexity_score = 1.0 / math.log(perplexity + 1)  # Inverse perplexity
        else:
            perplexity_score = 1.0

        # Normalize idea density to [0,1]
        idea_density_score = min(
            idea_density / 15.0, 1.0
        )  # Normalize around 15 ideas/100 words

        # Enhanced combination with semantic features
        if sentence_embeddings is not None:
            # Weighted combination: perplexity (30%), idea density (30%), semantic (25%), conceptual (15%)
            combined_density = (
                perplexity_score * 0.3
                + idea_density_score * 0.3
                + semantic_density * 0.25
                + conceptual_density * 0.15
            )
        else:
            # Fallback to original combination
            combined_density = (perplexity_score + idea_density_score) / 2.0

        return {
            "perplexity": perplexity,
            "idea_density": idea_density,
            "semantic_density": semantic_density,
            "conceptual_density": conceptual_density,
            "perplexity_score": perplexity_score,
            "idea_density_score": idea_density_score,
            "semantic_density_score": semantic_density,
            "conceptual_density_score": conceptual_density,
            "combined_density": combined_density,
            "has_semantic_features": sentence_embeddings is not None,
        }

    except Exception as e:
        logger.error(f"Error in density feature extraction: {e}")
        return {
            "perplexity": 25.0,  # Default perplexity
            "idea_density": 5.0,  # Default idea density
            "semantic_density": 0.5,
            "conceptual_density": 0.5,
            "perplexity_score": 0.5,
            "idea_density_score": 0.5,
            "semantic_density_score": 0.5,
            "conceptual_density_score": 0.5,
            "combined_density": 0.5,
            "has_semantic_features": False,
        }
