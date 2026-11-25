module "mlflow_artifacts" {
  source = "../s3_bucket"

  bucket_name = "${var.bucket_name_prefix}-${var.environment}-mlflow-artifacts"
  tags        = var.tags
}

module "data" {
  source = "../s3_bucket"

  bucket_name = "${var.bucket_name_prefix}-${var.environment}-data"
  tags        = var.tags
}

module "s3_access_policy" {
  source = "../s3_access_policy"

  policy_name                 = "${var.bucket_name_prefix}-${var.environment}-s3-access"
  mlflow_artifacts_bucket_arn = module.mlflow_artifacts.bucket_arn
  data_bucket_arn             = module.data.bucket_arn
  tags                        = var.tags
}
