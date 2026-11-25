module "phase1" {
  source = "../../modules/phase1"

  environment        = var.environment
  bucket_name_prefix = var.bucket_name_prefix
  tags = {
    Project     = "ml-sentinel"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
