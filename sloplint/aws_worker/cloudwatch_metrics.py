"""
CloudWatch metrics for AI Slop AWS worker.

Handles metrics collection and publishing to CloudWatch.
"""

import logging
import os
import time
from typing import Any

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available. AWS functionality will be limited.")

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and publishes metrics to CloudWatch."""

    def __init__(
        self, namespace: str = "AI-Slop-Worker", region_name: str = "us-east-1"
    ):
        """Initialize CloudWatch metrics client."""
        if not BOTO3_AVAILABLE:
            self.cloudwatch = None
            logger.warning("CloudWatch metrics disabled - boto3 not available")
            return

        self.namespace = namespace
        # Use LocalStack endpoint if test credentials are detected
        if os.getenv("AWS_ACCESS_KEY_ID") == "test":
            # Detect if running in Docker container
            if os.path.exists("/.dockerenv") or os.getenv("CONTAINER") == "docker":
                endpoint_url = "http://localstack:4566"
            else:
                endpoint_url = "http://localhost:4566"
            self.cloudwatch = boto3.client(
                "cloudwatch", region_name=region_name, endpoint_url=endpoint_url
            )
        else:
            self.cloudwatch = boto3.client("cloudwatch", region_name=region_name)
        self.logger = logging.getLogger(__name__)

    def publish_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "Count",
        dimensions: dict[str, str] | None = None,
    ) -> bool:
        """Publish a single metric to CloudWatch."""
        if not self.cloudwatch:
            return False

        try:
            # Prepare metric data
            metric_data = [
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": unit,
                    "Timestamp": time.time(),
                }
            ]

            # Add dimensions if provided
            if dimensions:
                metric_data[0]["Dimensions"] = [
                    {"Name": k, "Value": v} for k, v in dimensions.items()
                ]

            # Publish to CloudWatch
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace, MetricData=metric_data
            )

            logger.debug(f"Published metric: {metric_name} = {value} {unit}")
            return True

        except ClientError as e:
            logger.error(f"Error publishing metric {metric_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing metric {metric_name}: {e}")
            return False

    def publish_batch_metrics(
        self,
        metrics: dict[str, float],
        unit: str = "Count",
        dimensions: dict[str, str] | None = None,
    ) -> bool:
        """Publish multiple metrics in a single batch."""
        if not self.cloudwatch:
            return False

        try:
            # Prepare metric data
            metric_data = []
            for metric_name, value in metrics.items():
                metric_data.append(
                    {
                        "MetricName": metric_name,
                        "Value": value,
                        "Unit": unit,
                        "Timestamp": time.time(),
                    }
                )

                # Add dimensions if provided
                if dimensions:
                    metric_data[-1]["Dimensions"] = [
                        {"Name": k, "Value": v} for k, v in dimensions.items()
                    ]

            # Publish to CloudWatch
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace, MetricData=metric_data
            )

            logger.debug(f"Published {len(metrics)} metrics")
            return True

        except ClientError as e:
            logger.error(f"Error publishing batch metrics: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing batch metrics: {e}")
            return False


class WorkerMetrics:
    """Specialized metrics for the AI Slop worker."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize worker metrics."""
        self.collector = metrics_collector
        self.logger = logging.getLogger(__name__)

    def record_message_processed(
        self, processing_time: float, success: bool = True
    ) -> None:
        """Record that a message was processed."""
        # Processing time
        self.collector.publish_metric(
            "MessageProcessingTime", processing_time, "Milliseconds"
        )

        # Success/failure count
        if success:
            self.collector.publish_metric("MessagesProcessed", 1, "Count")
        else:
            self.collector.publish_metric("MessagesFailed", 1, "Count")

    def record_feature_extraction_time(
        self, feature_name: str, extraction_time: float
    ) -> None:
        """Record feature extraction timing."""
        metric_name = f"FeatureExtractionTime/{feature_name}"
        self.collector.publish_metric(metric_name, extraction_time, "Milliseconds")

    def record_slop_score(self, score: float, domain: str) -> None:
        """Record slop score metrics."""
        # Overall score
        self.collector.publish_metric("SlopScore", score, "None")

        # Score by domain
        dimensions = {"Domain": domain}
        self.collector.publish_metric("SlopScoreByDomain", score, "None", dimensions)

    def record_confidence_score(self, confidence: float) -> None:
        """Record confidence score."""
        self.collector.publish_metric("AnalysisConfidence", confidence, "None")

    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        # Count errors by type
        dimensions = {"ErrorType": error_type}
        self.collector.publish_metric("WorkerErrors", 1, "Count", dimensions)

    def record_queue_depth(self, queue_depth: int, queue_name: str) -> None:
        """Record queue depth for monitoring."""
        dimensions = {"QueueName": queue_name}
        self.collector.publish_metric("QueueDepth", queue_depth, "Count", dimensions)

    def record_memory_usage(self, memory_mb: float) -> None:
        """Record memory usage."""
        self.collector.publish_metric("MemoryUsage", memory_mb, "Megabytes")

    def record_cpu_usage(self, cpu_percent: float) -> None:
        """Record CPU usage."""
        self.collector.publish_metric("CPUUsage", cpu_percent, "Percent")

    def publish_batch_analysis(self, batch_metrics: dict[str, Any]) -> None:
        """Publish comprehensive batch analysis metrics."""
        # Extract metrics from batch
        slop_scores = batch_metrics.get("slop_scores", [])
        confidence_scores = batch_metrics.get("confidence_scores", [])
        processing_times = batch_metrics.get("processing_times", [])
        errors = batch_metrics.get("errors", {})

        # Publish aggregated metrics
        if slop_scores:
            avg_slop = sum(slop_scores) / len(slop_scores)
            self.collector.publish_metric("AverageSlopScore", avg_slop, "None")

        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            self.collector.publish_metric("AverageConfidence", avg_confidence, "None")

        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            self.collector.publish_metric(
                "AverageProcessingTime", avg_time, "Milliseconds"
            )

        # Publish error counts
        for error_type, count in errors.items():
            dimensions = {"ErrorType": error_type}
            self.collector.publish_metric("BatchErrors", count, "Count", dimensions)
