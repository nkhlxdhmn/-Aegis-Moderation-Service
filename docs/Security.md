# Security

Aegis Moderation is designed for local or self-hosted image moderation without a database or credentials.

## Ingestion Protections

- Only JPEG, PNG, WEBP, and GIF inputs are accepted.
- Uploads are limited to 10 MB by default.
- Image dimensions are validated before analysis.
- URL analysis requires HTTPS.
- URL credentials are rejected.
- Hostnames are resolved and private, loopback, link-local, and otherwise non-public IPs are rejected.
- Redirect targets are revalidated.

## Secret Hygiene

The repository should not contain `.env` files, API keys, cloud credentials, database credentials, private URLs, local paths, model weights, logs, caches, or training artifacts.

Suggested scan:

```bash
rg -n -uu -i "(api[_-]?key|secret|password|token|bearer|private key|service_role|BEGIN RSA|BEGIN OPENSSH|supabase\.co|redis://|postgresql://|localhost:5432|127\.0\.0\.1|c:\\Users|/home/)" .
```

Review matches before release; documentation examples may intentionally mention placeholder patterns.
