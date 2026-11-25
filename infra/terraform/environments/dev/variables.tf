variable "aws_region" {
  description = "AWS region for this environment."
  type        = string
  default     = "ap-southeast-2"
}

variable "environment" {
  description = "Environment name."
  type        = string

  validation {
    condition     = var.environment == "dev"
    error_message = "The dev root must use environment = \"dev\"."
  }
}

variable "bucket_name_prefix" {
  description = "Lowercase globally unique prefix for S3 bucket names; customize before apply."
  type        = string
}
