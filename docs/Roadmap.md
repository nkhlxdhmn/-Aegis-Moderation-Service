# Roadmap

This document outlines planned features and improvements for Aegis Moderation.

## Planned Features

- **Audio Moderation**: Native support for `.mp3` and `.wav` files (currently audio is only processed if embedded in a video container).
- **WebHooks**: Ability to register a webhook URL to receive asynchronous callbacks, allowing very large videos to be processed without holding the HTTP connection open.
- **Custom Rule Engine UI**: A dedicated page in the monitoring dashboard to tweak category thresholds without modifying code.
- **Support for Llama 3 / Qwen**: Optional integration for smaller, quantized LLMs to replace the rule-engine for complex context analysis.

## Known Limitations

- **Language Support**: OCR is heavily biased towards English and Latin scripts. The EasyOCR fallback provides basic Indic support, but CJK characters are currently unoptimized.
- **Throughput**: Since the application is a monolith running synchronous pipeline inference, throughput is heavily bottlenecked by GPU compute.
