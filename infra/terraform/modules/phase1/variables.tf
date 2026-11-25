variable "environment" {
  description = "Deployment environment name."
  type        = string
}

variable "bucket_name_prefix" {
  description = "Lowercase prefix used to form globally unique S3 bucket names."
  type        = string
}

variable "tags" {
  description = "Tags applied to Phase 1 resources."
  type        = map(string)
  default     = {}
}
