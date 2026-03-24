# Changelog

## v0.2.1 - 2026-03-24

### Added

- Container images now automatically built and pushed to `ghcr.io/philhem/stt-server` on version tags
- Images tagged with full semver (`v0.2.1`), major.minor (`0.2`), and commit SHA

## v0.2.0 - 2026-03-24

### Added

- Prometheus metrics endpoint at `/metrics` with request counts, latency histograms, inference timing, audio duration, in-progress gauge, and model info
- OpenTelemetry tracing with W3C `traceparent` propagation — spans for `transcribe`, `audio.decode`, and `model.inference` exported via OTLP gRPC
- Request ID propagation across services (`X-Request-ID`, `X-Litellm-Call-Id`, or auto-generated UUID) logged on every request for cross-service correlation
- LiteLLM integration: `verbose_json` response format with `duration` field for cost tracking, `/v1/models` endpoint for health checks and model discovery
- Native systemd journal logging mode (`--log-format journal`) with first-class structured fields queryable via `journalctl`
- Safety limits: `--max-concurrent` (default 4), `--max-file-size` (100 MB), `--max-audio-duration` (600s), `--request-timeout` (300s)
- Graceful shutdown on SIGTERM/SIGINT with 30-second drain for in-flight requests
- Audio format validation via magic bytes before ffmpeg (WAV, MP3, FLAC, OGG, WebM, M4A, AAC)
- Docker Compose stack in `e2e/` for local validation with LiteLLM, VictoriaMetrics, and VictoriaTraces
- Pre-commit hook enforcing hexagonal dependency direction (`scripts/check-deps.sh`)

### Changed

- Container image reduced from 580 MB to 129 MB by building ffmpeg from source with audio-only codecs (3.5 MB static binary)
- Dockerfile uses Go 1.26 base image
- Codebase restructured into hexagonal architecture: 6 internal packages with enforced dependency direction (no adapter-to-adapter imports)
- Error responses use static messages — internal details (ffmpeg stderr, Go errors) logged at debug level only
- Filenames and request IDs sanitized before logging/tracing to prevent log injection and stored XSS

### Fixed

- ffmpeg processes now killed on request timeout via `exec.CommandContext` (previously orphaned indefinitely)
- ffmpeg stdout capped at 25 MB to prevent memory bombs from small compressed files decompressing to gigabytes
- ffmpeg stderr capped at 64 KB to prevent unbounded memory from verbose error output
- Inference timeout works via goroutine+select pattern — CGo calls no longer block the HTTP handler indefinitely
- Recognizer cleanup waits for in-flight inference goroutines before destroying C objects (prevents use-after-free on shutdown)
- `ReadHeaderTimeout` set to 10s to prevent slowloris attacks; `IdleTimeout` set to 60s
- Multipart temp files cleaned up via `defer RemoveAll`
- Model name validated against `^[a-zA-Z0-9][a-zA-Z0-9._-]*$` to prevent URL injection
- Tar extraction capped at 2 GB total / 1 GB per file, file modes masked to 0o777
- Download client enforces HTTPS-only redirects (max 5) with 10-minute timeout
- ONNX model files verified via protobuf magic byte check
- Partial model extractions cleaned up on failure
- OTel baggage propagator removed to prevent arbitrary header reflection
- Concurrency semaphore checked before any resource allocation (no OTel spans on 503 rejection)

## v0.1.0 - 2026-03-24

### Added

- OpenAI-compatible HTTP server with `POST /v1/audio/transcriptions` and `GET /health`
- NVIDIA Parakeet V3 (TDT 0.6B, int8 ONNX) via sherpa-onnx Go bindings — 25 European languages
- Auto-download models by name with persistent cache (`--model sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`)
- Accepts any audio format via ffmpeg (MP3, FLAC, OGG, WAV, M4A, WebM, AAC)
- Structured logging with `--log-format text` (default) or `--log-format json`
- Container-first deployment: model mounted as volume or auto-downloaded to cache
- Configurable via CLI flags and environment variables
