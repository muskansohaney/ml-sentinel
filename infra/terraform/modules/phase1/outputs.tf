output "mlflow_artifacts_bucket_name" {
  description = "MLflow artifact bucket name."
  value       = module.mlflow_artifacts.bucket_name
}

output "data_bucket_name" {
  description = "Data bucket name."
  value       = module.data.bucket_name
}

output "s3_access_policy_arn" {
  description = "ARN of the least-privilege S3 access policy."
  value       = module.s3_access_policy.policy_arn
}
