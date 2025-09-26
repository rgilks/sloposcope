# Terraform configuration for AI Slop Worker ECS deployment

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
}

variable "input_queue_url" {
  description = "Input SQS queue URL"
  type        = string
}

variable "output_queue_url" {
  description = "Output SQS queue URL"
  type        = string
}

# ECR Repository
resource "aws_ecr_repository" "sloplint" {
  name                 = "sloplint-worker"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "sloplint-worker"
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "sloplint" {
  name = "sloplint-worker-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name        = "sloplint-worker"
    Environment = var.environment
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "sloplint" {
  family                   = "sloplint-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "sloplint-worker"
      image = "${aws_ecr_repository.sloplint.repository_url}:latest"

      environment = [
        {
          name  = "INPUT_QUEUE_URL"
          value = var.input_queue_url
        },
        {
          name  = "OUTPUT_QUEUE_URL"
          value = var.output_queue_url
        },
        {
          name  = "AWS_REGION"
          value = var.region
        },
        {
          name  = "LOG_LEVEL"
          value = "INFO"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.sloplint.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command = [
          "CMD-SHELL",
          "python -c \"import sys; sys.exit(0)\""
        ]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Name        = "sloplint-worker"
    Environment = var.environment
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "sloplint" {
  name              = "/ecs/sloplint-worker"
  retention_in_days = 30

  tags = {
    Name        = "sloplint-worker"
    Environment = var.environment
  }
}

# ECS Service
resource "aws_ecs_service" "sloplint" {
  name            = "sloplint-worker-service"
  cluster         = aws_ecs_cluster.sloplint.id
  task_definition = aws_ecs_task_definition.sloplint.arn
  desired_count   = 2

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 100
  }

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.sloplint.id]
    assign_public_ip = true
  }

  tags = {
    Name        = "sloplint-worker"
    Environment = var.environment
  }
}

# Security Group
resource "aws_security_group" "sloplint" {
  name_prefix = "sloplint-worker-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "sloplint-worker"
    Environment = var.environment
  }
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_execution_role" {
  name = "sloplint-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
  ]

  tags = {
    Name        = "sloplint-worker"
    Environment = var.environment
  }
}

# IAM Role for ECS Task
resource "aws_iam_role" "ecs_task_role" {
  name = "sloplint-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  inline_policy {
    name = "sloplint-worker-policy"
    policy = jsonencode({
      Version = "2012-10-17"
      Statement = [
        {
          Action = [
            "sqs:SendMessage",
            "sqs:ReceiveMessage",
            "sqs:DeleteMessage",
            "sqs:GetQueueAttributes"
          ]
          Effect   = "Allow"
          Resource = "*"
        },
        {
          Action = [
            "s3:GetObject",
            "s3:PutObject",
            "s3:ListBucket"
          ]
          Effect   = "Allow"
          Resource = "*"
        },
        {
          Action = [
            "cloudwatch:PutMetricData"
          ]
          Effect   = "Allow"
          Resource = "*"
        },
        {
          Action = [
            "logs:CreateLogStream",
            "logs:PutLogEvents"
          ]
          Effect   = "Allow"
          Resource = "*"
        }
      ]
    })
  }

  tags = {
    Name        = "sloplint-worker"
    Environment = var.environment
  }
}

# Outputs
output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.sloplint.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.sloplint.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.sloplint.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.sloplint.name
}
