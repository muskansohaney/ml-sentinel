output "policy_arn" {
  description = "ARN of the scoped S3 access policy."
  value       = aws_iam_policy.this.arn
}
