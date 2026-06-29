# Security Policy

## Supported Versions

| Version | Security fixes |
|---|---|
| `1.0.x` (main branch) | Yes |
| Older | No |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

Open a [private security advisory](https://github.com/your-org/aegis-moderation/security/advisories/new) on GitHub. Include:

1. A short description of the vulnerability and its impact.
2. Steps to reproduce (curl command, code snippet, or test file).
3. The version or commit where you found it.

We aim to respond within 5 business days and to publish a fix or mitigation within 30 days of confirmed impact.

---

## Security Controls

### Input Validation

| Control | Implementation |
|---|---|
| SSRF protection | `ipaddress` + `socket.getaddrinfo` — RFC-1918, loopback, and link-local IPs blocked at DNS time |
| URL scheme allowlist | Only `https://` accepted for remote URLs |
| File extension + MIME type check | Checked against an allow-list for every upload endpoint |
| File size limit | Configurable via `MAX_IMAGE_SIZE_MB` / `MAX_DOC_SIZE_MB`; enforced before writing to disk |
| Path traversal | All uploads written via `tempfile.NamedTemporaryFile`; user-controlled names are never used |
| Zip bomb protection | DOCX decompression ratio capped at 100×, max 80 MB uncompressed |
| Embedded executable detection | DOCX files containing `.exe`, `.dll`, `.ps1` etc. rejected immediately |

### Response Security

| Header | Value |
|---|---|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |

### Deployment Security

- The application has **no authentication layer** — it is designed for deployment behind a reverse proxy or API gateway that handles auth and rate-limiting.
- The CORS policy defaults to `allow_origins=["*"]`; restrict this in production by setting a proxy-level allowlist.
- No credentials, tokens, or secrets are required at runtime.

---

## Out of Scope

- Denial-of-service through excessively large CPU workloads on legitimate content (inference is intentionally expensive).
- Vulnerabilities in upstream model weights hosted on HuggingFace Hub.
- AI jailbreaking / adversarial content that bypasses classifiers (these are model accuracy issues, not security vulnerabilities).
