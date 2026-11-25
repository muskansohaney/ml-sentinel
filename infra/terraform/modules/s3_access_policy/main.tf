data "aws_iam_policy_document" "this" {
  statement {
    sid       = "ListMlflowArtifactBucket"
    actions   = ["s3:GetBucketLocation", "s3:ListBucket"]
    resources = [var.mlflow_artifacts_bucket_arn]
  }

  statement {
    sid = "ManageMlflowArtifacts"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:DeleteObject",
      "s3:GetObject",
      "s3:ListMultipartUploadParts",
      "s3:PutObject",
    ]
    resources = ["${var.mlflow_artifacts_bucket_arn}/*"]
  }

  statement {
    sid       = "ListDataBucket"
    actions   = ["s3:GetBucketLocation", "s3:ListBucket"]
    resources = [var.data_bucket_arn]
  }

  statement {
    sid       = "ReadWriteDataObjects"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${var.data_bucket_arn}/*"]
  }
}

resource "aws_iam_policy" "this" {
  name        = var.policy_name
  description = "Scoped S3 access for ML Sentinel storage."
  policy      = data.aws_iam_policy_document.this.json
  tags        = var.tags
}
