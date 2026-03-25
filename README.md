# stt-server

OpenAI-compatible speech-to-text HTTP server using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for offline inference. Runs NVIDIA Parakeet, Whisper, SenseVoice, Paraformer, and NeMo CTC models on CPU or CUDA.

## Quick start

```bash
docker run -p 8000:8000 -v stt-cache:/opt/stt/cache \
  -e STT_MODEL=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 \
  ghcr.io/philhem/stt-server:latest
```

The model is downloaded automatically on first start and cached for subsequent runs.

## API

All endpoints follow the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio) format. See [ADR-001](docs/adr/001-openai-api-compatibility.md) for compatibility scope and [ADR-002](docs/adr/002-unsupported-openai-params.md) for parameters that are accepted but ignored.

### `POST /v1/audio/transcriptions`

Transcribe an audio file. Multipart form upload.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | Audio file (see [Audio formats](#audio-formats)) |
| `model` | string | no | Accepted for compatibility, single model served |
| `language` | string | no | ISO 639-1 hint; used as fallback when model detection returns empty |
| `response_format` | string | no | `json` (default), `text`, `verbose_json`, `srt`, `vtt` |

**json** (default):
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3

# {"text":"Hello world."}
```

**verbose_json** — includes duration, language, word and segment timestamps:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 -F response_format=verbose_json

# {
#   "task": "transcribe",
#   "text": "Hello world.",
#   "language": "en",
#   "duration": 2.5,
#   "words": [
#     {"word": "Hello", "start": 0.0, "end": 0.32},
#     {"word": "world.", "start": 0.32, "end": 0.64}
#   ],
#   "segments": [
#     {"id": 0, "start": 0.0, "end": 0.64, "text": "Hello world."}
#   ]
# }
```

**text** — plain text, no JSON wrapper:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 -F response_format=text

# Hello world.
```

**srt** / **vtt** — subtitle formats:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 -F response_format=srt
```

Error responses use OpenAI's format:
```json
{"error": {"message": "...", "type": "invalid_request_error", "code": "400"}}
```

When all processing slots and the queue are full, the server returns `429` with a `Retry-After: 5` header. OpenAI SDKs and LiteLLM retry this automatically. See [ADR-003](docs/adr/003-queue-and-rate-limiting.md).

### `GET /health`

```json
{"status": "ok", "version": "v0.3.1"}
```

### `GET /v1/models`

OpenAI-compatible model listing for LiteLLM health checks and model discovery.

```json
{"object": "list", "data": [{"id": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", "object": "model", "owned_by": "local"}]}
```

### `GET /metrics`

Prometheus metrics endpoint. See [Observability](#observability).

### Request headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Propagated to response and logs. Falls back to `X-Litellm-Call-Id`, then auto-generated UUID. |
| `traceparent` | W3C trace context propagated through to OTel spans. |

## Configuration

Every flag has a corresponding environment variable. Env vars take precedence when no flag is passed.

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--model` | `STT_MODEL` | *(required)* | Model name (auto-downloaded) or path to local directory |
| `--cache-dir` | `STT_CACHE_DIR` | `/opt/stt/cache` | Directory for downloaded models |
| `--port` | `STT_PORT` | `8000` | HTTP listen port |
| `--num-threads` | `STT_NUM_THREADS` | `runtime.NumCPU()` | ONNX Runtime inference threads |
| `--provider` | `STT_PROVIDER` | `cpu` | ONNX Runtime execution provider: `cpu` or `cuda` |
| `--max-concurrent` | `STT_MAX_CONCURRENT` | `4` | Max concurrent transcription requests |
| `--max-queue` | `STT_MAX_QUEUE` | `8` | Max requests waiting for a processing slot |
| `--max-file-size` | `STT_MAX_FILE_SIZE_MB` | `100` | Max upload file size in MB |
| `--max-audio-duration` | `STT_MAX_AUDIO_DURATION` | `600` | Max audio duration in seconds |
| `--request-timeout` | `STT_REQUEST_TIMEOUT` | `300` | Per-request timeout in seconds (queue wait + processing) |
| `--log-format` | `STT_LOG_FORMAT` | `text` | `text`, `json`, or `journal` |
| `--log-level` | `STT_LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error` |
| `--otel-endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` | *(disabled)* | OTLP gRPC endpoint for traces |
| `--version` | | | Print version and exit |

## Models

The server auto-detects the model type from files present in the model directory. Detection order:

| Type | Files required | Example models |
|------|---------------|----------------|
| NeMo Transducer | `encoder.onnx` + `decoder.onnx` + `joiner.onnx` | Parakeet TDT, Zipformer |
| Whisper | `encoder.onnx` + `decoder.onnx` (no joiner) | Whisper tiny/base/small/medium/large |
| SenseVoice | `model.onnx` + directory name contains `sense_voice` | SenseVoice Small/Large |
| Paraformer | `model.onnx` + directory name contains `paraformer` | Paraformer |
| NeMo CTC | `model.onnx` (generic fallback) | Conformer CTC, Citrinet |

All models must include a `tokens.txt` file. Both full-precision (`.onnx`) and quantized (`.int8.onnx`) weights are supported.

**Auto-download by name** — pass a [sherpa-onnx model name](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models):
```bash
stt-server --model sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
```

**Local directory** — mount or point to an existing model:
```bash
stt-server --model /path/to/model
```

## Audio formats

Input is decoded to 16 kHz mono PCM via ffmpeg. The following formats are recognized by magic bytes before being passed to ffmpeg:

- WAV
- MP3 (ID3 tag or raw frame sync)
- FLAC
- OGG (Vorbis, Opus)
- WebM
- M4A / MP4
- AAC (ADTS)

Minimum audio duration: 100 ms. Maximum: configurable via `--max-audio-duration` (default 600 s).

## Languages

Parakeet V3 supports 25 European languages (auto-detected per utterance):

| | | | | |
|---|---|---|---|---|
| Bulgarian (bg) | Croatian (hr) | Czech (cs) | Danish (da) | Dutch (nl) |
| English (en) | Estonian (et) | Finnish (fi) | French (fr) | German (de) |
| Greek (el) | Hungarian (hu) | Italian (it) | Latvian (lv) | Lithuanian (lt) |
| Maltese (mt) | Polish (pl) | Portuguese (pt) | Romanian (ro) | Russian (ru) |
| Slovak (sk) | Slovenian (sl) | Spanish (es) | Swedish (sv) | Ukrainian (uk) |

Whisper and SenseVoice models support their own language sets. Use the `language` parameter to hint when needed.

## Observability

### Metrics

Prometheus metrics are served at `GET /metrics`.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `stt_requests_total` | counter | `status`, `lang` | Total requests by HTTP status and detected language |
| `stt_audio_bytes_total` | counter | | Total bytes of audio uploaded |
| `stt_requests_in_progress` | gauge | | Requests currently being processed |
| `stt_requests_queued` | gauge | | Requests waiting for a processing slot |
| `stt_model_info` | gauge | `model`, `provider`, `threads` | Static model metadata (value always 1) |
| `stt_build_info` | gauge | `version`, `commit`, `build_time` | Build metadata (value always 1) |
| `stt_request_duration_seconds` | histogram | | End-to-end latency (upload + decode + inference) |
| `stt_decode_duration_seconds` | histogram | | ffmpeg audio decode time |
| `stt_inference_duration_seconds` | histogram | | Model inference time only |
| `stt_audio_duration_seconds` | histogram | | Duration of audio processed per request |

### Tracing

OpenTelemetry traces are exported via OTLP gRPC when `--otel-endpoint` is set. Three spans per request:

- `transcribe` — full request lifecycle
- `audio.decode` — ffmpeg format conversion
- `model.inference` — sherpa-onnx model execution

W3C `traceparent` headers are propagated from upstream (e.g., LiteLLM) and injected into responses.

### Logging

Three output modes via `--log-format`:

| Format | Output | Use case |
|--------|--------|----------|
| `text` | `key=value` to stderr | Local development |
| `json` | Structured JSON to stderr | Loki, Promtail, ELK |
| `journal` | Native systemd journal fields | `journalctl` queries with structured fields |

Each transcription request logs: `request_id`, `file`, `size_bytes`, `duration_s`, `elapsed_ms`, `decode_ms`, `inference_ms`, `lang`, `trace_id`, `span_id`.

## Deployment

### Docker

```bash
# Auto-download model, persist cache
docker run -p 8000:8000 -v stt-cache:/opt/stt/cache \
  -e STT_MODEL=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 \
  ghcr.io/philhem/stt-server:latest

# Mount local model directory
docker run -p 8000:8000 -v /path/to/model:/model:ro \
  -e STT_MODEL=/model \
  ghcr.io/philhem/stt-server:latest
```

The container image is ~129 MB. It includes a minimal static ffmpeg built from source with audio-only codecs. The process runs as a non-root `stt` user.

### GPU (CUDA)

```bash
docker run --gpus all -p 8000:8000 -v stt-cache:/opt/stt/cache \
  -e STT_MODEL=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 \
  -e STT_PROVIDER=cuda \
  ghcr.io/philhem/stt-server:latest
```

### LiteLLM integration

stt-server works as a backend for [LiteLLM](https://github.com/BerryDev/litellm) audio transcription proxy. LiteLLM config:

```yaml
model_list:
  - model_name: parakeet
    litellm_params:
      model: whisper/parakeet
      api_base: http://stt-server:8000
      api_key: unused
      drop_params: true
```

The `e2e/` directory contains a full Docker Compose stack with LiteLLM, VictoriaMetrics, and VictoriaTraces for local testing:

```bash
cd e2e && docker compose up
```

### E2E test stack

```bash
# Validate the live server with the test script
./scripts/test-e2e.sh
```

## Building from source

Requires Go 1.25+, CGO enabled (for sherpa-onnx bindings), and ffmpeg in `PATH`.

```bash
# Download a model for local testing
./download-model.sh

# Build and run
go build -o stt-server ./cmd/stt-server/
./stt-server --model models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
```

With version info:

```bash
go build -ldflags "-X github.com/PhilHem/stt-server/internal/config.Version=v0.3.1 \
  -X github.com/PhilHem/stt-server/internal/config.Commit=$(git rev-parse --short HEAD) \
  -X github.com/PhilHem/stt-server/internal/config.BuildTime=$(date -u +%FT%TZ)" \
  -o stt-server ./cmd/stt-server/
```

### Docker build

```bash
docker build -t stt-server:local \
  --build-arg VERSION=v0.3.1 \
  --build-arg COMMIT=$(git rev-parse --short HEAD) \
  --build-arg BUILD_TIME=$(date -u +%FT%TZ) .
```

## License

See [LICENSE](LICENSE) file.
