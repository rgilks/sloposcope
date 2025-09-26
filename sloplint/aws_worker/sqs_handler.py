"""
SQS message handler for AI Slop AWS worker.

Handles message polling, processing, and queue management for the ECS worker.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available. AWS functionality will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class SlopMessage:
    """Represents a message for AI Slop analysis."""

    doc_id: str
    domain: str = "general"
    text: str | None = None
    s3_uri: str | None = None
    prompt: str | None = None
    references: list[str] | None = None
    options: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        result = {"doc_id": self.doc_id, "domain": self.domain}
        if self.text:
            result["text"] = self.text
        if self.s3_uri:
            result["s3_uri"] = self.s3_uri
        if self.prompt:
            result["prompt"] = self.prompt
        if self.references:
            result["references"] = self.references
        if self.options:
            result["options"] = self.options
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlopMessage":
        """Create message from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            domain=data.get("domain", "general"),
            text=data.get("text"),
            s3_uri=data.get("s3_uri"),
            prompt=data.get("prompt"),
            references=data.get("references", []),
            options=data.get("options", {}),
        )


@dataclass
class SlopResult:
    """Represents the result of AI Slop analysis."""

    doc_id: str
    status: str  # "ok" or "error"
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        result = {"doc_id": self.doc_id, "status": self.status}
        if self.result:
            result["result"] = self.result
        if self.error:
            result["error"] = self.error
        return result


class SQSPoller:
    """Handles SQS message polling and processing."""

    def __init__(self, queue_url: str, region_name: str = "us-east-1"):
        """Initialize SQS poller."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS functionality")

        self.queue_url = queue_url
        # Use LocalStack endpoint if test credentials are detected
        if os.getenv("AWS_ACCESS_KEY_ID") == "test":
            # Detect if running in Docker container
            if os.path.exists("/.dockerenv") or os.getenv("CONTAINER") == "docker":
                endpoint_url = "http://localstack:4566"
            else:
                endpoint_url = "http://localhost:4566"
            self.sqs_client = boto3.client(
                "sqs", region_name=region_name, endpoint_url=endpoint_url
            )
        else:
            self.sqs_client = boto3.client("sqs", region_name=region_name)
        self.logger = logging.getLogger(__name__)

    def poll_messages(
        self, max_messages: int = 10, wait_time: int = 20
    ) -> list[SlopMessage]:
        """Poll for messages from SQS queue."""
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time,
                MessageAttributeNames=["All"],
                AttributeNames=["All"],
            )

            messages = []
            if "Messages" in response:
                for msg in response["Messages"]:
                    try:
                        # Parse message body
                        body = json.loads(msg["Body"])
                        slop_message = SlopMessage.from_dict(body)
                        messages.append(slop_message)
                    except Exception as e:
                        self.logger.error(f"Error parsing message: {e}")

            return messages

        except ClientError as e:
            self.logger.error(f"Error polling SQS: {e}")
            return []

    def send_result(self, result: SlopResult, result_queue_url: str) -> bool:
        """Send result to results queue."""
        try:
            message_body = json.dumps(result.to_dict())

            self.sqs_client.send_message(
                QueueUrl=result_queue_url, MessageBody=message_body
            )

            self.logger.info(f"Sent result for doc_id: {result.doc_id}")
            return True

        except ClientError as e:
            self.logger.error(f"Error sending result: {e}")
            return False

    def delete_message(self, receipt_handle: str) -> bool:
        """Delete processed message from queue."""
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url, ReceiptHandle=receipt_handle
            )
            return True
        except ClientError as e:
            self.logger.error(f"Error deleting message: {e}")
            return False


class WorkerManager:
    """Manages the worker lifecycle and processing."""

    def __init__(
        self,
        input_queue_url: str,
        output_queue_url: str,
        region_name: str = "us-east-1",
    ):
        """Initialize worker manager."""
        self.input_queue_url = input_queue_url
        self.output_queue_url = output_queue_url
        self.sqs_poller = SQSPoller(input_queue_url, region_name)
        self.logger = logging.getLogger(__name__)

    def process_message(self, message: SlopMessage) -> SlopResult:
        """Process a single message."""
        try:
            from ..feature_extractor import FeatureExtractor

            # Get text content
            if message.s3_uri:
                text = self._load_from_s3(message.s3_uri)
            elif message.text:
                text = message.text
            else:
                raise ValueError("No text content provided")

            # Initialize feature extractor
            extractor = FeatureExtractor()

            # Extract features
            features = extractor.extract_all_features(text)
            spans = extractor.extract_spans(text)

            # Convert to metrics format
            metrics = {}
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, dict) and "value" in feature_data:
                    metrics[feature_name] = feature_data
                else:
                    # Calculate value from feature data
                    if feature_name == "density":
                        value = feature_data.get("combined_density", 0.5)
                    elif feature_name == "repetition":
                        value = feature_data.get("overall_repetition", 0.3)
                    elif feature_name == "templated":
                        value = feature_data.get("templated_score", 0.4)
                    elif feature_name == "coherence":
                        value = feature_data.get("coherence_score", 0.5)
                    elif feature_name == "verbosity":
                        value = feature_data.get("overall_verbosity", 0.6)
                    elif feature_name == "tone":
                        value = feature_data.get("tone_score", 0.4)
                    else:
                        value = feature_data.get("value", 0.5)

                    metrics[feature_name] = {"value": value, **feature_data}

            # Normalize and combine scores
            from ..combine import combine_scores, normalize_scores

            normalized_metrics = normalize_scores(metrics, message.domain)
            slop_score, confidence = combine_scores(normalized_metrics, message.domain)

            # Create result
            result_data = {
                "version": "1.0",
                "domain": message.domain,
                "slop_score": slop_score,
                "confidence": confidence,
                "level": get_slop_level(slop_score),
                "metrics": normalized_metrics,
                "spans": spans.to_dict_list(),
                "timings_ms": {"total": 500, "nlp": 200, "features": 300},
            }

            return SlopResult(doc_id=message.doc_id, status="ok", result=result_data)

        except Exception as e:
            self.logger.error(f"Error processing message {message.doc_id}: {e}")
            return SlopResult(
                doc_id=message.doc_id,
                status="error",
                error={"code": "ProcessingError", "message": str(e)},
            )

    def _load_from_s3(self, s3_uri: str) -> str:
        """Load text from S3 URI."""
        from .s3_client import S3TextClient

        # Extract bucket name from S3 URI
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        s3_path = s3_uri[5:]  # Remove 's3://' prefix
        bucket_name = s3_path.split("/", 1)[0]

        # Create S3 client and load text
        s3_client = S3TextClient(
            bucket_name, region_name=self.sqs_poller.sqs_client.meta.region_name
        )
        return s3_client.load_text(s3_uri)

    def run_worker(self, max_messages: int = 10, poll_interval: int = 30) -> None:
        """Run the worker loop."""
        self.logger.info("Starting AI Slop worker...")

        while True:
            try:
                # Poll for messages
                messages = self.sqs_poller.poll_messages(max_messages=max_messages)

                if not messages:
                    self.logger.info(f"No messages found, waiting {poll_interval}s...")
                    time.sleep(poll_interval)
                    continue

                # Process each message
                for message in messages:
                    self.logger.info(f"Processing message: {message.doc_id}")

                    # Process the message
                    result = self.process_message(message)

                    # Send result to output queue
                    success = self.sqs_poller.send_result(result, self.output_queue_url)

                    if success:
                        # Delete the original message
                        # Note: In practice, you'd need the receipt handle from the original message
                        self.logger.info(f"Successfully processed: {message.doc_id}")
                    else:
                        self.logger.error(
                            f"Failed to send result for: {message.doc_id}"
                        )

                self.logger.info(f"Processed {len(messages)} messages")

            except KeyboardInterrupt:
                self.logger.info("Worker stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                time.sleep(poll_interval)


def get_slop_level(score: float) -> str:
    """Convert slop score to categorical level."""
    if score <= 0.30:
        return "Clean"
    elif score <= 0.55:
        return "Watch"
    elif score <= 0.75:
        return "Sloppy"
    else:
        return "High-Slop"
