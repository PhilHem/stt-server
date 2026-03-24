# ADR-001: OpenAI API Compatibility Scope

## Status
Accepted

## Context
stt-server exposes an OpenAI-compatible `/v1/audio/transcriptions` endpoint for integration with LiteLLM and the OpenAI Python SDK. The OpenAI API has evolved to include features specific to their proprietary models (GPT-4o Transcribe, Whisper) that may not map to NVIDIA Parakeet V3.

## Decision
We implement the subset of the OpenAI API that:
1. Is required by LiteLLM's audio_transcription proxy
2. Works with the OpenAI Python SDK's `client.audio.transcriptions.create()`
3. Maps meaningfully to Parakeet V3's capabilities

### Supported
- `POST /v1/audio/transcriptions` with multipart file upload
- `GET /v1/models` for model discovery
- `GET /health` for readiness probes
- `response_format`: json, text, verbose_json, srt, vtt
- `model` parameter (accepted, single model served)
- `language` parameter (accepted, Parakeet auto-detects)
- `verbose_json` returns `text`, `task`, `language`, `duration`, `words`, `segments`
- 429 + Retry-After for rate limiting (OpenAI SDK auto-retries)
- Error responses in OpenAI format (`{"error": {"message": ..., "type": ..., "code": ...}}`)

### Not supported (by design)
- `prompt` parameter (Whisper-specific conditioning)
- `temperature` parameter (Parakeet uses greedy decoding)
- `timestamp_granularities` parameter (we return all available by default)
- `diarize` / `known_speaker_names` (GPT-4o Transcribe feature)
- `include` parameter (logprobs — model-specific)
- Streaming transcription
- Translation (`/v1/audio/translations`)

## Consequences
- LiteLLM's `drop_params: true` safely ignores unsupported parameters
- OpenAI SDK works without modification for transcription workflows
- Features requiring model changes (diarization, streaming) are out of scope until Parakeet or sherpa-onnx supports them
