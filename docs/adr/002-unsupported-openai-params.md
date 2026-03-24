# ADR-002: Unsupported OpenAI Parameters

## Status
Accepted

## Context
The OpenAI Audio API includes parameters that are specific to their proprietary models and cannot be meaningfully implemented with NVIDIA Parakeet V3 via sherpa-onnx.

## Decision

### `prompt` — Not applicable
Whisper uses `prompt` to condition the model with prior context (e.g., previous transcript segment). Parakeet's TDT architecture doesn't have an equivalent conditioning mechanism. Accepting and ignoring it is safe.

### `temperature` — Not applicable
Parakeet uses greedy decoding (argmax at each step). There is no sampling step where temperature would apply. The decoding method is hardcoded to `greedy_search` in the recognizer config.

### `timestamp_granularities` — Implicit
OpenAI uses this to control whether word-level or segment-level timestamps are returned. Our implementation returns both by default in verbose_json. The parameter is accepted and ignored.

### `diarize` / `known_speaker_names` — Not supported
Speaker diarization is a GPT-4o Transcribe feature that requires a separate model component. Parakeet V3 is a single-speaker ASR model. Adding diarization would require integrating a separate diarization model (e.g., Pyannote). This is a potential future enhancement but not in current scope.

### `include` (logprobs) — Not supported
Log probabilities are model-internal data that sherpa-onnx's Go bindings don't expose. Would require changes to the sherpa-onnx C API.

## Consequences
- All unsupported parameters are safely dropped by LiteLLM (`drop_params: true`)
- No error is returned for unsupported parameters (matches OpenAI behavior — unknown params are ignored)
- If Parakeet or sherpa-onnx adds support for any of these features in the future, we can implement them without API changes
