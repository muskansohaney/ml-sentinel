variable "aws_region" {
  description = "AWS region for this environment."
  type        = string
  default     = "ap-southeast-2"
}

variable "environment" {
  description = "Environment name."
  type        = string

  validation {
    condition     = var.environment == "staging"
    error_message = "The staging root must use environment = \"staging\"."
  }
}

variable "bucket_name_prefix" {
  description = "Lowercase globally unique prefix for S3 bucket names; customize before apply."
  type        = string
}
