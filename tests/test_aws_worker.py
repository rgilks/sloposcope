"""
Tests for AWS Worker components using LocalStack.
"""

import json
import os

import boto3
import pytest
from botocore.exceptions import ClientError

from sloplint.aws_worker.cloudwatch_metrics import MetricsCollector, WorkerMetrics
from sloplint.aws_worker.s3_client import S3TextClient
from sloplint.aws_worker.sqs_handler import (
    SlopMessage,
    SlopResult,
    SQSPoller,
    WorkerManager,
)


@pytest.fixture(scope="session")
def localstack_endpoint():
    """Get LocalStack endpoint URL."""
    return "http://localhost:4566"


@pytest.fixture(scope="session")
def aws_credentials():
    """Set up test AWS credentials."""
    return {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": "us-east-1",
    }


@pytest.fixture(scope="session")
def localstack_resources(localstack_endpoint, aws_credentials):
    """Set up LocalStack resources for testing."""
    # Set environment variables
    for key, value in aws_credentials.items():
        os.environ[key] = value

    # Create SQS client
    sqs = boto3.client("sqs", endpoint_url=localstack_endpoint, region_name="us-east-1")

    # Create S3 client
    s3 = boto3.client("s3", endpoint_url=localstack_endpoint, region_name="us-east-1")

    # Create queues
    input_queue = sqs.create_queue(QueueName="test-input-queue")
    output_queue = sqs.create_queue(QueueName="test-output-queue")

    # Create S3 bucket
    bucket_name = "test-bucket"
    s3.create_bucket(Bucket=bucket_name)

    # Create test file in S3
    test_content = "This is a test document for AI slop analysis. It contains some repetitive content that should trigger detection."
    s3.put_object(
        Bucket=bucket_name, Key="test-document.txt", Body=test_content.encode("utf-8")
    )

    yield {
        "input_queue_url": input_queue["QueueUrl"],
        "output_queue_url": output_queue["QueueUrl"],
        "bucket_name": bucket_name,
        "test_s3_uri": f"s3://{bucket_name}/test-document.txt",
        "test_content": test_content,
    }

    # Cleanup
    try:
        sqs.delete_queue(QueueUrl=input_queue["QueueUrl"])
        sqs.delete_queue(QueueUrl=output_queue["QueueUrl"])
        s3.delete_object(Bucket=bucket_name, Key="test-document.txt")
        s3.delete_bucket(Bucket=bucket_name)
    except ClientError:
        pass  # Ignore cleanup errors


class TestSlopMessage:
    """Test SlopMessage data class."""

    def test_message_creation(self):
        """Test creating a SlopMessage."""
        message = SlopMessage(
            doc_id="test-001",
            domain="general",
            text="Test content",
            s3_uri="s3://bucket/file.txt",
            options={"detailed_analysis": True},
        )

        assert message.doc_id == "test-001"
        assert message.domain == "general"
        assert message.text == "Test content"
        assert message.s3_uri == "s3://bucket/file.txt"
        assert message.options == {"detailed_analysis": True}

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        message = SlopMessage(doc_id="test-001", domain="general", text="Test content")

        data = message.to_dict()
        assert data["doc_id"] == "test-001"
        assert data["domain"] == "general"
        assert data["text"] == "Test content"

    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "doc_id": "test-001",
            "domain": "general",
            "text": "Test content",
            "s3_uri": "s3://bucket/file.txt",
        }

        message = SlopMessage.from_dict(data)
        assert message.doc_id == "test-001"
        assert message.domain == "general"
        assert message.text == "Test content"
        assert message.s3_uri == "s3://bucket/file.txt"


class TestSlopResult:
    """Test SlopResult data class."""

    def test_success_result(self):
        """Test creating a success result."""
        result_data = {"slop_score": 0.5, "confidence": 0.8, "level": "Watch"}

        result = SlopResult(doc_id="test-001", status="ok", result=result_data)

        assert result.doc_id == "test-001"
        assert result.status == "ok"
        assert result.result == result_data
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        error_data = {"code": "ProcessingError", "message": "Test error"}

        result = SlopResult(doc_id="test-001", status="error", error=error_data)

        assert result.doc_id == "test-001"
        assert result.status == "error"
        assert result.error == error_data
        assert result.result is None


class TestS3TextClient:
    """Test S3TextClient functionality."""

    def test_s3_client_creation(self, localstack_resources):
        """Test creating S3 client."""
        client = S3TextClient(
            bucket_name=localstack_resources["bucket_name"], region_name="us-east-1"
        )

        assert client.bucket_name == localstack_resources["bucket_name"]

    def test_load_text_from_s3(self, localstack_resources):
        """Test loading text from S3."""
        client = S3TextClient(
            bucket_name=localstack_resources["bucket_name"], region_name="us-east-1"
        )

        text = client.load_text(localstack_resources["test_s3_uri"])
        assert text == localstack_resources["test_content"]

    def test_save_text_to_s3(self, localstack_resources):
        """Test saving text to S3."""
        client = S3TextClient(
            bucket_name=localstack_resources["bucket_name"], region_name="us-east-1"
        )

        test_text = "This is a test document for saving to S3."
        test_uri = f"s3://{localstack_resources['bucket_name']}/test-save.txt"

        success = client.save_text(test_text, test_uri)
        assert success is True

        # Verify the text was saved
        loaded_text = client.load_text(test_uri)
        assert loaded_text == test_text

    def test_object_exists(self, localstack_resources):
        """Test checking if S3 object exists."""
        client = S3TextClient(
            bucket_name=localstack_resources["bucket_name"], region_name="us-east-1"
        )

        # Test existing object
        assert client.object_exists(localstack_resources["test_s3_uri"]) is True

        # Test non-existing object
        non_existing_uri = (
            f"s3://{localstack_resources['bucket_name']}/non-existing.txt"
        )
        assert client.object_exists(non_existing_uri) is False


class TestSQSPoller:
    """Test SQS message polling functionality."""

    def test_sqs_poller_creation(self, localstack_resources):
        """Test creating SQS poller."""
        poller = SQSPoller(
            queue_url=localstack_resources["input_queue_url"], region_name="us-east-1"
        )

        assert poller.queue_url == localstack_resources["input_queue_url"]

    def test_send_and_receive_message(self, localstack_resources):
        """Test sending and receiving messages."""
        poller = SQSPoller(
            queue_url=localstack_resources["input_queue_url"], region_name="us-east-1"
        )

        # Create test message
        message = SlopMessage(
            doc_id="test-001", domain="general", text="Test message content"
        )

        # Send message
        sqs = boto3.client(
            "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
        )
        sqs.send_message(
            QueueUrl=localstack_resources["input_queue_url"],
            MessageBody=json.dumps(message.to_dict()),
        )

        # Receive message
        messages = poller.poll_messages(max_messages=1, wait_time=5)
        assert len(messages) == 1
        assert messages[0].doc_id == "test-001"
        assert messages[0].text == "Test message content"

    def test_send_result(self, localstack_resources):
        """Test sending result to output queue."""
        poller = SQSPoller(
            queue_url=localstack_resources["input_queue_url"], region_name="us-east-1"
        )

        result = SlopResult(
            doc_id="test-001",
            status="ok",
            result={"slop_score": 0.5, "confidence": 0.8},
        )

        success = poller.send_result(result, localstack_resources["output_queue_url"])
        assert success is True

        # Verify result was sent
        sqs = boto3.client(
            "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
        )
        response = sqs.receive_message(
            QueueUrl=localstack_resources["output_queue_url"], MaxNumberOfMessages=1
        )

        assert "Messages" in response
        received_result = json.loads(response["Messages"][0]["Body"])
        assert received_result["doc_id"] == "test-001"
        assert received_result["status"] == "ok"


class TestWorkerManager:
    """Test WorkerManager functionality."""

    def test_worker_manager_creation(self, localstack_resources):
        """Test creating worker manager."""
        manager = WorkerManager(
            input_queue_url=localstack_resources["input_queue_url"],
            output_queue_url=localstack_resources["output_queue_url"],
            region_name="us-east-1",
        )

        assert manager.input_queue_url == localstack_resources["input_queue_url"]
        assert manager.output_queue_url == localstack_resources["output_queue_url"]

    def test_process_message_with_text(self, localstack_resources):
        """Test processing a message with text content."""
        manager = WorkerManager(
            input_queue_url=localstack_resources["input_queue_url"],
            output_queue_url=localstack_resources["output_queue_url"],
            region_name="us-east-1",
        )

        message = SlopMessage(
            doc_id="test-001",
            domain="general",
            text="This is a test document for AI slop analysis.",
        )

        result = manager.process_message(message)

        assert result.doc_id == "test-001"
        assert result.status == "ok"
        assert "result" in result.to_dict()
        assert "slop_score" in result.result
        assert "confidence" in result.result

    def test_process_message_with_s3(self, localstack_resources):
        """Test processing a message with S3 URI."""
        manager = WorkerManager(
            input_queue_url=localstack_resources["input_queue_url"],
            output_queue_url=localstack_resources["output_queue_url"],
            region_name="us-east-1",
        )

        message = SlopMessage(
            doc_id="test-001",
            domain="general",
            s3_uri=localstack_resources["test_s3_uri"],
        )

        result = manager.process_message(message)

        assert result.doc_id == "test-001"
        assert result.status == "ok"
        assert "result" in result.to_dict()
        assert "slop_score" in result.result
        assert "confidence" in result.result

    def test_process_message_error(self, localstack_resources):
        """Test processing a message with invalid content."""
        manager = WorkerManager(
            input_queue_url=localstack_resources["input_queue_url"],
            output_queue_url=localstack_resources["output_queue_url"],
            region_name="us-east-1",
        )

        message = SlopMessage(
            doc_id="test-001",
            domain="general",
            # No text or s3_uri provided
        )

        result = manager.process_message(message)

        assert result.doc_id == "test-001"
        assert result.status == "error"
        assert "error" in result.to_dict()


class TestMetricsCollector:
    """Test CloudWatch metrics functionality."""

    def test_metrics_collector_creation(self):
        """Test creating metrics collector."""
        collector = MetricsCollector(
            namespace="Test-Namespace", region_name="us-east-1"
        )

        assert collector.namespace == "Test-Namespace"

    def test_publish_metric(self):
        """Test publishing a single metric."""
        collector = MetricsCollector(
            namespace="Test-Namespace", region_name="us-east-1"
        )

        success = collector.publish_metric(
            metric_name="TestMetric", value=42.0, unit="Count"
        )

        # Should succeed even with LocalStack
        assert success is True

    def test_publish_batch_metrics(self):
        """Test publishing multiple metrics."""
        collector = MetricsCollector(
            namespace="Test-Namespace", region_name="us-east-1"
        )

        metrics = {"Metric1": 10.0, "Metric2": 20.0, "Metric3": 30.0}

        success = collector.publish_batch_metrics(metrics)

        # Should succeed even with LocalStack
        assert success is True


class TestWorkerMetrics:
    """Test WorkerMetrics functionality."""

    def test_worker_metrics_creation(self):
        """Test creating worker metrics."""
        collector = MetricsCollector(
            namespace="Test-Namespace", region_name="us-east-1"
        )

        worker_metrics = WorkerMetrics(collector)
        assert worker_metrics.collector == collector

    def test_record_message_processed(self):
        """Test recording message processing metrics."""
        collector = MetricsCollector(
            namespace="Test-Namespace", region_name="us-east-1"
        )

        worker_metrics = WorkerMetrics(collector)

        # Should not raise any exceptions
        worker_metrics.record_message_processed(processing_time=100.0, success=True)
        worker_metrics.record_message_processed(processing_time=200.0, success=False)

    def test_record_slop_score(self):
        """Test recording slop score metrics."""
        collector = MetricsCollector(
            namespace="Test-Namespace", region_name="us-east-1"
        )

        worker_metrics = WorkerMetrics(collector)

        # Should not raise any exceptions
        worker_metrics.record_slop_score(score=0.5, domain="general")
        worker_metrics.record_confidence_score(confidence=0.8)


@pytest.mark.integration
class TestEndToEndIntegration:
    """Test end-to-end integration with LocalStack."""

    def test_full_workflow(self, localstack_resources):
        """Test complete workflow from message to result."""
        # Create worker manager
        manager = WorkerManager(
            input_queue_url=localstack_resources["input_queue_url"],
            output_queue_url=localstack_resources["output_queue_url"],
            region_name="us-east-1",
        )

        # Send test message
        sqs = boto3.client(
            "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
        )

        message_data = {
            "doc_id": "integration-test-001",
            "domain": "general",
            "s3_uri": localstack_resources["test_s3_uri"],
            "options": {"detailed_analysis": True},
        }

        sqs.send_message(
            QueueUrl=localstack_resources["input_queue_url"],
            MessageBody=json.dumps(message_data),
        )

        # Process message
        messages = manager.sqs_poller.poll_messages(max_messages=1, wait_time=5)
        assert len(messages) == 1

        message = messages[0]
        assert message.doc_id == "integration-test-001"
        assert message.s3_uri == localstack_resources["test_s3_uri"]

        # Process the message
        result = manager.process_message(message)

        assert result.doc_id == "integration-test-001"
        assert result.status == "ok"
        assert "slop_score" in result.result
        assert "confidence" in result.result
        assert "metrics" in result.result

        # Send result to output queue
        success = manager.sqs_poller.send_result(
            result, localstack_resources["output_queue_url"]
        )
        assert success is True

        # Verify result in output queue
        response = sqs.receive_message(
            QueueUrl=localstack_resources["output_queue_url"], MaxNumberOfMessages=1
        )

        assert "Messages" in response
        received_result = json.loads(response["Messages"][0]["Body"])
        assert received_result["doc_id"] == "integration-test-001"
        assert received_result["status"] == "ok"
        assert received_result["result"]["slop_score"] > 0
        assert received_result["result"]["confidence"] > 0
