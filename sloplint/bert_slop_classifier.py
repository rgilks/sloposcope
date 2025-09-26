"""
BERT-based Slop Detection Classifier

This module provides a focused BERT-based approach for exceptional English slop detection.
Uses fine-tuned BERT models to directly classify text as slop or high-quality content.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from datasets import Dataset

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    Dataset = None

logger = logging.getLogger(__name__)


@dataclass
class SlopDetectionResult:
    """Result from BERT-based slop detection."""

    is_slop: bool
    confidence: float
    slop_score: float
    quality_score: float
    explanation: str


class BERTSlopClassifier:
    """BERT-based classifier for exceptional English slop detection."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """Initialize BERT-based slop classifier."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for BERT classifier")

        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.is_fitted = False

        # Load pre-trained components
        self._load_pretrained_components()

    def _load_pretrained_components(self):
        """Load pre-trained tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: slop vs quality
                problem_type="single_label_classification",
            )
            self.model.to(self.device)
            logger.info(f"Loaded BERT model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise

    def _prepare_training_data(self, texts: List[str], labels: List[int]) -> Dataset:
        """Prepare training data for BERT fine-tuning."""
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create dataset
        dataset = Dataset.from_dict(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
            }
        )

        return dataset

    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        output_dir: str = "./bert_slop_model",
    ):
        """Fine-tune BERT model on slop detection task."""
        logger.info(f"Fine-tuning BERT model on {len(train_texts)} training samples")

        # Prepare training data
        train_dataset = self._prepare_training_data(train_texts, train_labels)

        # Prepare validation data if provided
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = self._prepare_training_data(val_texts, val_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=500 if val_dataset else None,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            learning_rate=learning_rate,
            report_to=None,  # Disable wandb/tensorboard
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            if val_dataset
            else None,
        )

        # Fine-tune the model
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        self.is_fitted = True
        logger.info(f"BERT model fine-tuned and saved to {output_dir}")

    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned BERT model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.is_fitted = True
            logger.info(f"Loaded fine-tuned BERT model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise

    def predict(self, text: str) -> SlopDetectionResult:
        """Predict if text is slop using BERT classifier."""
        if not self.is_fitted:
            logger.warning("Model not fine-tuned, using pre-trained weights")

        # Tokenize input text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Extract results
        slop_prob = probabilities[0][1].item()  # Probability of being slop
        quality_prob = probabilities[0][0].item()  # Probability of being quality

        is_slop = slop_prob > 0.5
        confidence = max(slop_prob, quality_prob)

        # Generate explanation
        explanation = self._generate_explanation(text, slop_prob, is_slop)

        return SlopDetectionResult(
            is_slop=is_slop,
            confidence=confidence,
            slop_score=slop_prob,
            quality_score=quality_prob,
            explanation=explanation,
        )

    def _generate_explanation(self, text: str, slop_prob: float, is_slop: bool) -> str:
        """Generate explanation for the prediction."""
        if is_slop:
            if slop_prob > 0.8:
                return f"High confidence slop detected ({slop_prob:.2f}). Text shows strong indicators of AI generation."
            elif slop_prob > 0.6:
                return f"Moderate confidence slop detected ({slop_prob:.2f}). Text contains several slop indicators."
            else:
                return f"Low confidence slop detected ({slop_prob:.2f}). Text shows some slop characteristics."
        else:
            if slop_prob < 0.2:
                return f"High quality text ({slop_prob:.2f}). No significant slop indicators detected."
            elif slop_prob < 0.4:
                return f"Good quality text ({slop_prob:.2f}). Minimal slop indicators."
            else:
                return f"Acceptable quality text ({slop_prob:.2f}). Some minor slop characteristics present."

    def batch_predict(self, texts: List[str]) -> List[SlopDetectionResult]:
        """Predict slop for multiple texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


class SlopDetectionPipeline:
    """Complete pipeline combining BERT classifier with feature-based analysis."""

    def __init__(
        self,
        bert_model_path: Optional[str] = None,
        use_feature_fallback: bool = True,
    ):
        """Initialize the complete slop detection pipeline."""
        self.bert_classifier = BERTSlopClassifier()
        self.use_feature_fallback = use_feature_fallback

        # Load fine-tuned model if provided
        if bert_model_path:
            self.bert_classifier.load_fine_tuned_model(bert_model_path)

        # Import feature-based components for fallback
        if use_feature_fallback:
            try:
                from .optimized_feature_extractor import OptimizedFeatureExtractor
                from .combine import combine_features

                self.feature_extractor = OptimizedFeatureExtractor(
                    use_transformer=False
                )
                self.feature_fallback_available = True
            except ImportError:
                logger.warning("Feature-based fallback not available")
                self.feature_fallback_available = False
        else:
            self.feature_fallback_available = False

    def analyze(self, text: str) -> Dict[str, Any]:
        """Comprehensive slop analysis using BERT + features."""
        # Primary BERT-based analysis
        bert_result = self.bert_classifier.predict(text)

        result = {
            "text": text,
            "bert_analysis": {
                "is_slop": bert_result.is_slop,
                "confidence": bert_result.confidence,
                "slop_score": bert_result.slop_score,
                "quality_score": bert_result.quality_score,
                "explanation": bert_result.explanation,
            },
            "method": "bert_classifier",
            "model_name": self.bert_classifier.model_name,
        }

        # Add feature-based analysis as fallback/validation
        if self.feature_fallback_available:
            try:
                features = self.feature_extractor.extract_all_features(text)
                combined_score = self._calculate_feature_score(features)

                result["feature_analysis"] = {
                    "slop_score": combined_score,
                    "features": features,
                }

                # Compare BERT and feature-based results
                bert_slop = bert_result.is_slop
                feature_slop = combined_score > 0.5

                result["consensus"] = {
                    "bert_says_slop": bert_slop,
                    "features_say_slop": feature_slop,
                    "agreement": bert_slop == feature_slop,
                    "confidence_difference": abs(bert_result.confidence - 0.5),
                }

            except Exception as e:
                logger.warning(f"Feature analysis failed: {e}")
                result["feature_analysis"] = {"error": str(e)}

        return result

    def _calculate_feature_score(self, features: Dict[str, Any]) -> float:
        """Calculate combined slop score from features."""
        # Simple weighted combination of key features
        weights = {
            "repetition_score": 0.2,
            "templated_score": 0.2,
            "coherence_score": 0.15,
            "density_score": 0.15,
            "tone_score": 0.1,
            "verbosity_score": 0.1,
            "complexity_score": 0.1,
        }

        total_score = 0.0
        total_weight = 0.0

        for feature_name, weight in weights.items():
            if feature_name in features:
                total_score += features[feature_name] * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts."""
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results


def create_training_data() -> Tuple[List[str], List[int]]:
    """Create high-quality training data for BERT fine-tuning."""

    # High-quality English text examples
    quality_texts = [
        "The implementation uses a distributed architecture with microservices that communicate through REST APIs.",
        "Each service is containerized using Docker and deployed on Kubernetes clusters for scalability.",
        "The system implements event-driven patterns with message queues for asynchronous communication.",
        "Security is handled through OAuth 2.0 authentication and JWT tokens for secure access.",
        "The database uses PostgreSQL with read replicas for improved performance and reliability.",
        "Machine learning models are trained on large datasets to achieve state-of-the-art performance.",
        "The research methodology follows established protocols for data collection and analysis.",
        "Economic indicators suggest a gradual recovery in the manufacturing sector.",
        "Climate change poses significant challenges for agricultural productivity worldwide.",
        "The novel explores themes of identity and belonging in contemporary society.",
        "Scientific evidence supports the hypothesis that exercise improves cognitive function.",
        "The company's quarterly earnings exceeded analyst expectations by 15%.",
        "Educational institutions are adapting to new technologies for enhanced learning outcomes.",
        "The legal framework provides clear guidelines for intellectual property protection.",
        "Urban planning initiatives aim to reduce traffic congestion and improve air quality.",
    ]

    # AI-generated slop examples
    slop_texts = [
        "In today's rapidly evolving digital landscape, it's absolutely crucial to understand that leveraging cutting-edge technologies can truly revolutionize your business operations and unlock unprecedented opportunities for growth and success.",
        "As an AI, I can confidently say that this revolutionary approach will undoubtedly transform the way we think about innovation and drive meaningful change across industries.",
        "Here are 5 amazing ways to boost your productivity: First, prioritize your tasks effectively. Second, eliminate distractions completely. Third, take regular breaks. Fourth, stay organized. Fifth, maintain focus.",
        "The implementation is very important and provides many benefits. The system is very useful and helps users. The features are very helpful and improve efficiency. The solution is very effective and saves time.",
        "In conclusion, it's clear that this innovative solution represents a paradigm shift in how we approach complex challenges and create value for stakeholders.",
        "This comprehensive analysis reveals that the data suggests a strong correlation between various factors that contribute to the overall success of the initiative.",
        "The methodology employed in this study demonstrates that the results indicate a significant improvement in performance metrics across multiple dimensions.",
        "It's important to note that the findings suggest that the approach taken provides a solid foundation for future research and development efforts.",
        "The research shows that the implementation of these strategies can lead to substantial improvements in key performance indicators.",
        "This approach offers a unique opportunity to address the challenges faced by organizations in today's competitive environment.",
        "The solution provides a comprehensive framework for addressing the complex issues that arise in modern business operations.",
        "It's worth mentioning that the results demonstrate the effectiveness of the proposed methodology in achieving desired outcomes.",
        "The analysis reveals that the data supports the hypothesis that the approach taken is both innovative and practical.",
        "This innovative solution represents a significant advancement in the field and offers numerous benefits for users.",
        "The implementation of these strategies can help organizations achieve their goals and improve their competitive position.",
    ]

    # Combine and label
    all_texts = quality_texts + slop_texts
    labels = [0] * len(quality_texts) + [1] * len(slop_texts)  # 0 = quality, 1 = slop

    return all_texts, labels


def train_bert_slop_classifier(
    output_dir: str = "./bert_slop_model",
    epochs: int = 5,
    batch_size: int = 8,
) -> BERTSlopClassifier:
    """Train a BERT classifier for slop detection."""
    logger.info("Creating training data for BERT slop classifier")

    # Create training data
    train_texts, train_labels = create_training_data()

    # Split into train/validation
    split_idx = int(0.8 * len(train_texts))
    train_texts_split = train_texts[:split_idx]
    train_labels_split = train_labels[:split_idx]
    val_texts_split = train_texts[split_idx:]
    val_labels_split = train_labels[split_idx:]

    # Initialize classifier
    classifier = BERTSlopClassifier()

    # Fine-tune the model
    classifier.fine_tune(
        train_texts=train_texts_split,
        train_labels=train_labels_split,
        val_texts=val_texts_split,
        val_labels=val_labels_split,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    logger.info(f"BERT slop classifier trained and saved to {output_dir}")
    return classifier
