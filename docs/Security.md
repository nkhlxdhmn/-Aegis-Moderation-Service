# Security

Aegis Moderation is designed for local or self-hosted multimodal moderation without a database or credentials.

## Ingestion Protections

- Image uploads are limited to JPEG, PNG, WEBP, and GIF.
- Image uploads are limited to 10 MB by default.
- Video uploads are limited to common browser video formats and 250 MB.
- PDF and DOCX uploads are limited to 25 MB.
- PDF processing rejects password-protected, corrupted, and over-limit documents.
- DOCX processing rejects corrupted files, zip bombs, and embedded executable content.
- Image dimensions are validated before analysis.
- URL analysis requires HTTPS.
- URL credentials are rejected.
- Hostnames are resolved and private, loopback, link-local, and otherwise non-public IPs are rejected.
- Redirect targets are revalidated.

## Document Checks

PDF and DOCX moderation extracts text, metadata, links, embedded image counts, and document-specific risk signals. Current checks include phishing language, executable links, sensitive-document terms, financial information terms, identity-document terms, copyright notices, QR/barcode hints, PII detection, spam, scam, toxic text, and hate speech.

Scanned PDF OCR and embedded-image vision analysis are extension points in this build; machine-readable text extraction and document security validation are implemented.

## Secret Hygiene

The repository should not contain `.env` files, API keys, cloud credentials, database credentials, private URLs, local paths, model weights, logs, caches, or training artifacts.

Suggested scan:

```bash
rg -n -uu -i "(api[_-]?key|secret|password|token|bearer|private key|service_role|BEGIN RSA|BEGIN OPENSSH|supabase\.co|redis://|postgresql://|localhost:5432|127\.0\.0\.1|c:\\Users|/home/)" .
```

Review matches before release; documentation examples may intentionally mention placeholder patterns.
