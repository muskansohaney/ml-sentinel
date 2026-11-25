output "mlflow_artifacts_bucket_name" {
  description = "MLflow artifact bucket name."
  value       = module.phase1.mlflow_artifacts_bucket_name
}

output "data_bucket_name" {
  description = "Data bucket name."
  value       = module.phase1.data_bucket_name
}

output "s3_access_policy_arn" {
  description = "ARN of the least-privilege S3 access policy."
  value       = module.phase1.s3_access_policy_arn
}
