package server

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/PhilHem/stt-server/internal/config"
	"github.com/PhilHem/stt-server/internal/observe"
)

var testCfg = config.Config{
	MaxConcurrent:    4,
	MaxFileSizeMB:    100,
	MaxAudioDuration: 600 * time.Second,
	RequestTimeout:   300 * time.Second,
}

var testMetrics = observe.NewMetrics()
var testSem = make(chan struct{}, testCfg.MaxConcurrent)
var testMaxBody = int64(testCfg.MaxFileSizeMB) << 20

func TestHealthEndpoint(t *testing.T) {
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	handleHealth(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var body map[string]string
	json.NewDecoder(resp.Body).Decode(&body)
	if body["status"] != "ok" {
		t.Errorf("expected status=ok, got %v", body["status"])
	}
}

func TestTranscriptionEndpoint_MissingFile(t *testing.T) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.Close()

	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	w := httptest.NewRecorder()

	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testMaxBody)
	handler(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	var errResp map[string]any
	json.Unmarshal(body, &errResp)

	errObj, ok := errResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error object in response, got: %s", body)
	}
	if errObj["type"] != "invalid_request_error" {
		t.Errorf("expected type=invalid_request_error, got %v", errObj["type"])
	}
}

func TestRequestID_Generated(t *testing.T) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.Close()

	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testMaxBody)
	handler(w, req)

	reqID := w.Header().Get("X-Request-ID")
	if reqID == "" {
		t.Error("expected X-Request-ID in response headers")
	}
	if len(reqID) != 36 {
		t.Errorf("expected UUID format, got: %s", reqID)
	}
}

func TestRequestID_Propagated(t *testing.T) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.Close()

	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("X-Request-ID", "upstream-id-123")

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testMaxBody)
	handler(w, req)

	reqID := w.Header().Get("X-Request-ID")
	if reqID != "upstream-id-123" {
		t.Errorf("expected propagated request ID 'upstream-id-123', got: %s", reqID)
	}
}

func TestRequestID_LiteLLMCallID(t *testing.T) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.Close()

	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("X-Litellm-Call-Id", "litellm-call-456")

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testMaxBody)
	handler(w, req)

	reqID := w.Header().Get("X-Request-ID")
	if reqID != "litellm-call-456" {
		t.Errorf("expected LiteLLM call ID 'litellm-call-456', got: %s", reqID)
	}
}

func TestConcurrencyLimit(t *testing.T) {
	// Fill the semaphore completely
	fullSem := make(chan struct{}, 1)
	fullSem <- struct{}{} // slot taken

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.Close()

	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	w := httptest.NewRecorder()
	cfg := testCfg
	cfg.MaxConcurrent = 1
	handler := handleTranscription(nil, cfg, testMetrics, fullSem, testMaxBody)
	handler(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 when concurrency limit hit, got %d", resp.StatusCode)
	}
}
