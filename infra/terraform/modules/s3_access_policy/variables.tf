variable "policy_name" {
  description = "Name of the managed IAM policy."
  type        = string
}

variable "mlflow_artifacts_bucket_arn" {
  description = "ARN of the MLflow artifact bucket."
  type        = string
}

variable "data_bucket_arn" {
  description = "ARN of the data bucket."
  type        = string
}

variable "tags" {
  description = "Tags applied to the policy."
  type        = map(string)
  default     = {}
}
