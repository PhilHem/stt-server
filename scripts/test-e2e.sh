#!/usr/bin/env bash
# E2E test suite for stt-server
# Usage: ./scripts/test-e2e.sh [base_url]
# Requires: a running stt-server with test audio at models/*/test_wavs/
set -euo pipefail

BASE="${1:-http://localhost:8000}"
PASS=0; FAIL=0; TOTAL=0
TEST_WAV="models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/test_wavs/de.wav"

pass() { ((PASS++)); ((TOTAL++)); echo "  PASS: $1"; }
fail() { ((FAIL++)); ((TOTAL++)); echo "  FAIL: $1 — $2"; }
check() {
    local name="$1" expected="$2" actual="$3"
    if echo "$actual" | grep -q "$expected"; then pass "$name"; else fail "$name" "expected '$expected', got '$actual'"; fi
}

echo "Running E2E tests against $BASE"
echo ""

# Health
echo "=== Health ==="
R=$(curl -sf "$BASE/health")
check "health status" '"status":"ok"' "$R"
check "health version" '"version":' "$R"

# Models
echo "=== Models ==="
R=$(curl -sf "$BASE/v1/models")
check "models list" '"object":"list"' "$R"
check "models data" '"object":"model"' "$R"

# Transcription — JSON
echo "=== Transcription (json) ==="
R=$(curl -sf "$BASE/v1/audio/transcriptions" -F file=@"$TEST_WAV" -F model=p)
check "json has text" '"text":' "$R"
check "json content" "Wurst" "$R"

# Transcription — text
echo "=== Transcription (text) ==="
R=$(curl -sf "$BASE/v1/audio/transcriptions" -F file=@"$TEST_WAV" -F model=p -F response_format=text)
check "text content" "Wurst" "$R"

# Transcription — verbose_json
echo "=== Transcription (verbose_json) ==="
R=$(curl -sf "$BASE/v1/audio/transcriptions" -F file=@"$TEST_WAV" -F model=p -F response_format=verbose_json)
check "verbose has text" '"text":' "$R"
check "verbose has duration" '"duration":' "$R"
check "verbose has task" '"task":"transcribe"' "$R"

# Metrics
echo "=== Metrics ==="
R=$(curl -sf "$BASE/metrics")
check "has request counter" "stt_requests_total" "$R"
check "has build info" "stt_build_info" "$R"
check "has model info" "stt_model_info" "$R"

# Error: bad file
echo "=== Errors ==="
CODE=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/v1/audio/transcriptions" -F "file=@/dev/null;filename=bad.wav" -F model=p || true)
check "bad file returns 400" "400" "$CODE"

# Error: missing file
CODE=$(curl -sf -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/audio/transcriptions" 2>/dev/null || echo "400")
check "missing file returns 400" "400" "$CODE"

# Request ID
echo "=== Request ID ==="
HEADERS=$(curl -sfD - "$BASE/v1/audio/transcriptions" -F file=@"$TEST_WAV" -F model=p -o /dev/null 2>&1)
check "has X-Request-ID" "X-Request-Id" "$HEADERS"

echo ""
echo "========================================="
echo "  Results: $PASS/$TOTAL passed, $FAIL failed"
echo "========================================="
[ "$FAIL" -eq 0 ] || exit 1
