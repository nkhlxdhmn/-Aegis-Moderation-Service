# Security Policy

Aegis Moderation is designed for local or self-hosted multimodal moderation without a database or credentials.

## Threat Model & Protections

- **Server-Side Request Forgery (SSRF)**
  - Remote image/document downloads only accept `https://` URLs.
  - DNS resolution explicitly rejects RFC-1918 (private/local) IP blocks (e.g., `127.0.0.1`, `10.0.0.0/8`).
  - URL redirects are followed manually up to 4 times and validated at each step.
  - URL credentials are rejected.

- **Denial of Service (DoS)**
  - Image payloads are limited to 10 MB and 40,000,000 pixels.
  - Documents (PDF/DOCX) are limited to 25 MB.
  - PDF processing rejects password-protected, corrupted, and over-limit documents.
  - DOCX zip-bomb protection limits decompression ratios to 100x and caps uncompressed buffers at 80 MB.
  - FAISS embedding limits prevent unbounded memory growth.

- **Path Traversal & Execution**
  - All uploads and downloads are saved using `tempfile.NamedTemporaryFile` with randomized names.
  - Files are automatically unlinked upon request completion.
  - DOCX extraction actively scans for executable extensions (`.exe`, `.dll`, `.ps1`, `.bat`) and halts processing immediately if found.

- **MIME & Format Validation**
  - Strict content-type checking (`ALLOWED_CONTENT_TYPES`) via headers and magic bytes.
  - Images are parsed with `Pillow.verify()` before model loading.

## Secret Hygiene

The repository should not contain `.env` files, API keys, cloud credentials, database credentials, private URLs, local paths, model weights, logs, caches, or training artifacts.

No credentials, tokens, or secrets are required at runtime.

## Reporting a Vulnerability

If you discover a security vulnerability, please open an issue in this repository. Ensure you omit any private hostnames or proprietary logic from the report.
