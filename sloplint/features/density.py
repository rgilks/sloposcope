"""
Density analysis for AI slop detection.

Calculates information density using perplexity from language models
and idea density based on linguistic features.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
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
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
                    self.device
                )

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


def extract_features(
    text: str,
    sentences: list[str],
    tokens: list[str],
    pos_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Extract all density-related features."""
    try:
        calculator = DensityCalculator()

        # Calculate perplexity
        perplexity = calculator.calculate_perplexity(text)

        # Calculate idea density
        idea_density = calculator.calculate_idea_density(sentences, pos_tags or [])

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

        # Combine with equal weights (as per SPEC)
        combined_density = (perplexity_score + idea_density_score) / 2.0

        return {
            "perplexity": perplexity,
            "idea_density": idea_density,
            "perplexity_score": perplexity_score,
            "idea_density_score": idea_density_score,
            "combined_density": combined_density,
        }

    except Exception as e:
        logger.error(f"Error in density feature extraction: {e}")
        return {
            "perplexity": 25.0,  # Default perplexity
            "idea_density": 5.0,  # Default idea density
            "perplexity_score": 0.5,
            "idea_density_score": 0.5,
            "combined_density": 0.5,
        }
