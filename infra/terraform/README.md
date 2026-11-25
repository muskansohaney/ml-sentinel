# Terraform: Phase 1

This scaffold provisions two private S3 buckets per environment: one for MLflow artifacts and one for data. It also creates a standalone IAM managed policy scoped to those buckets. The policy is not attached to a role; workload identity and trust relationships are intentionally left for a later design.

The `environments/dev`, `environments/staging`, and `environments/prod` directories are independent Terraform roots and state boundaries. Each uses `ap-southeast-2` and the shared modules.

Before any future apply, set `bucket_name_prefix` in that environment's `terraform.tfvars` to a lowercase, globally unique S3-compatible prefix. No backend is configured yet. Keep local development independent of AWS.

The S3 buckets have public access blocked, server-side encryption enabled, and a policy denying insecure transport. Versioning is omitted to avoid ongoing storage costs in this initial scaffold. The IAM policy grants MLflow artifact read/write/delete and multipart-upload permissions, and data bucket read/write permissions, limited to the two bucket ARNs.
