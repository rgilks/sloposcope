"""
Coherence analysis for AI slop detection.

Calculates entity grid continuity and topic drift to measure text coherence.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Coherence analysis will be limited.")


class EntityGridCoherence:
    """Calculates coherence using entity grid analysis."""

    def __init__(self):
        """Initialize entity grid analyzer."""
        self.entity_roles = ['S', 'O', 'X', '-']  # Subject, Object, Other, None

    def extract_entities(self, sentences: List[str]) -> Dict[str, List[str]]:
        """Extract entities and their roles from sentences."""
        if not SPACY_AVAILABLE:
            return self._fallback_entity_extraction(sentences)

        try:
            nlp = spacy.load("en_core_web_sm")

            entities_by_sentence = {}
            for i, sentence in enumerate(sentences):
                doc = nlp(sentence)

                sentence_entities = []
                for token in doc:
                    # Determine entity role
                    if token.dep_ == 'nsubj':
                        role = 'S'  # Subject
                    elif token.dep_ == 'dobj' or token.dep_ == 'pobj':
                        role = 'O'  # Object
                    elif token.ent_type_:
                        role = 'X'  # Other entity
                    else:
                        role = '-'  # None

                    sentence_entities.append(role)

                entities_by_sentence[i] = sentence_entities

            return entities_by_sentence

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return self._fallback_entity_extraction(sentences)

    def _fallback_entity_extraction(self, sentences: List[str]) -> Dict[str, List[str]]:
        """Fallback entity extraction when spaCy is not available."""
        # Simple rule-based extraction
        entities_by_sentence = {}

        for i, sentence in enumerate(sentences):
            words = sentence.split()
            roles = []

            for word in words:
                word_lower = word.lower().strip('.,!?;:')
                # Simple heuristics for entity roles
                if word_lower in ['the', 'a', 'an']:
                    roles.append('O')  # Determiner -> object
                elif word_lower.endswith('ing') or word_lower.endswith('ed'):
                    roles.append('S')  # Verb form -> subject
                elif len(word_lower) > 6 and not word_lower.endswith(('ing', 'ed', 'ly')):
                    roles.append('X')  # Long word -> entity
                else:
                    roles.append('-')  # Other

            entities_by_sentence[i] = roles

        return entities_by_sentence

    def calculate_entity_continuity(self, entities_by_sentence: Dict[int, List[str]]) -> float:
        """Calculate entity continuity score using transition probabilities."""
        if not entities_by_sentence:
            return 0.0

        # This is a simplified version of entity grid analysis
        # In practice, would use the full Barzilay-Lapata algorithm

        continuity_scores = []

        # Look at entity transitions between sentences
        sentence_indices = sorted(entities_by_sentence.keys())

        for i in range(len(sentence_indices) - 1):
            current_sent = entities_by_sentence[sentence_indices[i]]
            next_sent = entities_by_sentence[sentence_indices[i + 1]]

            # Count entity continuations
            continuations = 0
            total_entities = 0

            for role in self.entity_roles:
                if role in current_sent:
                    total_entities += 1
                    if role in next_sent:
                        continuations += 1

            if total_entities > 0:
                continuity = continuations / total_entities
                continuity_scores.append(continuity)

        return np.mean(continuity_scores) if continuity_scores else 0.0

    def calculate_embedding_drift(self, sentences: List[str]) -> float:
        """Calculate drift between adjacent sentence embeddings."""
        if len(sentences) < 2:
            return 0.0

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Get embeddings for all sentences
            embeddings = model.encode(sentences)

            # Calculate cosine similarities between adjacent sentences
            drifts = []
            for i in range(len(embeddings) - 1):
                emb1 = embeddings[i]
                emb2 = embeddings[i + 1]

                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

                # Convert to drift (lower similarity = higher drift)
                drift = 1.0 - similarity
                drifts.append(drift)

            return np.mean(drifts) if drifts else 0.0

        except Exception as e:
            logger.error(f"Error calculating embedding drift: {e}")
            return 0.0

    def detect_coherence_breaks(self, sentences: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Detect coherence breaks between sentences."""
        if len(sentences) < 2:
            return []

        breaks = []
        drift = self.calculate_embedding_drift(sentences)

        # Find positions where drift is high
        for i in range(len(sentences) - 1):
            if drift > threshold:
                # Estimate character position
                char_pos = sum(len(sentences[j]) + 1 for j in range(i + 1))
                breaks.append({
                    "start": char_pos,
                    "end": char_pos + 1,
                    "type": "coherence_break",
                    "note": f"High topic drift (drift: {drift:.3f})",
                })

        return breaks


def extract_features(text: str, sentences: List[str], tokens: List[str]) -> Dict[str, Any]:
    """Extract all coherence-related features."""
    try:
        analyzer = EntityGridCoherence()

        # Extract entities and roles
        entities_by_sentence = analyzer.extract_entities(sentences)

        # Calculate entity continuity
        entity_continuity = analyzer.calculate_entity_continuity(entities_by_sentence)

        # Calculate embedding drift
        embedding_drift = analyzer.calculate_embedding_drift(sentences)

        # Detect coherence breaks
        coherence_spans = analyzer.detect_coherence_breaks(sentences)

        # Calculate overall coherence score
        # Higher continuity and lower drift = higher coherence
        coherence_score = (
            entity_continuity * 0.6 +  # Entity continuity
            (1.0 - embedding_drift) * 0.4  # Low drift = high coherence
        )

        return {
            "entity_continuity": entity_continuity,
            "embedding_drift": embedding_drift,
            "coherence_score": coherence_score,
            "coherence_spans": coherence_spans,
        }

    except Exception as e:
        logger.error(f"Error in coherence feature extraction: {e}")
        return {
            "entity_continuity": 0.0,
            "embedding_drift": 1.0,
            "coherence_score": 0.0,
            "coherence_spans": [],
        }
