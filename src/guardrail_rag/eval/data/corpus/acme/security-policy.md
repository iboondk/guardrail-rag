# Acme Security Policy

All production data is encrypted at rest with AES-256 and in transit with TLS 1.3. Encryption keys are rotated every 90 days by the key management service, and every rotation event is logged.

Employees must use hardware security keys for every administrative console. Password logins are disabled for production systems. Access reviews run every quarter, and any access that has not been used in 45 days is revoked automatically.

Customer data is stored in the EU region by default. Backups are kept for 35 days in an isolated account that no engineer can modify without a second approval.

Vulnerabilities can be reported through the responsible disclosure form. Acme acknowledges reports within one business day and aims to ship fixes for critical issues within 14 days. An external firm carries out penetration tests twice a year.

Security training is required for every new hire within their first 30 days, and it is repeated each year with a short assessment.
