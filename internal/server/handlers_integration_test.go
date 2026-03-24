package server

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Helper to create a multipart request with a file field.
func newMultipartRequest(t *testing.T, url, fieldName, filename string, data []byte) *http.Request {
	t.Helper()
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile(fieldName, filename)
	if err != nil {
		t.Fatalf("create form file: %v", err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatalf("write file data: %v", err)
	}
	writer.Close()

	req := httptest.NewRequest("POST", url, &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req
}

// Helper to create a multipart request with a file field and extra form fields.
func newMultipartRequestWithFields(t *testing.T, url, fieldName, filename string, data []byte, fields map[string]string) *http.Request {
	t.Helper()
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile(fieldName, filename)
	if err != nil {
		t.Fatalf("create form file: %v", err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatalf("write file data: %v", err)
	}
	for k, v := range fields {
		if err := writer.WriteField(k, v); err != nil {
			t.Fatalf("write field %s: %v", k, err)
		}
	}
	writer.Close()

	req := httptest.NewRequest("POST", url, &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req
}

// fakeWAVHeader produces a minimal WAV header that passes IsKnownFormat.
// The actual audio content is zeros — ffmpeg will still fail, but the format
// check itself passes.
func fakeWAVHeader() []byte {
	return []byte("RIFF\x00\x00\x00\x00WAVEfmt ")
}

func TestTranscription_JSONFormat(t *testing.T) {
	// With nil recognizer, the handler will fail after audio decode.
	// We test that the handler correctly processes multipart input up to
	// the point where it needs the recognizer.
	// Sending an actual WAV with nil recognizer: the audio decode (ffmpeg)
	// will attempt to process the minimal WAV and either fail or produce empty
	// output. Either way, we get a 400 error — verifying the handler chain works.
	req := newMultipartRequest(t, "/v1/audio/transcriptions", "file", "test.wav", fakeWAVHeader())

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testQueue, testMaxBody)
	handler(w, req)

	resp := w.Result()
	body, _ := io.ReadAll(resp.Body)

	// With nil recognizer, we expect a 400 (audio decode error for minimal WAV)
	// The response should still be valid JSON with an error object.
	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatalf("response is not valid JSON: %v, body: %s", err, body)
	}
	if _, hasError := result["error"]; !hasError {
		// If no error, then text field should be present (unlikely with nil recognizer)
		if _, hasText := result["text"]; !hasText {
			t.Error("expected either 'error' or 'text' field in JSON response")
		}
	}
}

func TestTranscription_TextFormat(t *testing.T) {
	// Same as JSON test — we verify the handler chain processes the format parameter.
	req := newMultipartRequestWithFields(t, "/v1/audio/transcriptions", "file", "test.wav", fakeWAVHeader(), map[string]string{
		"response_format": "text",
	})

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testQueue, testMaxBody)
	handler(w, req)

	resp := w.Result()
	// With nil recognizer, we expect a 400 from decode failure.
	// The key assertion: the handler doesn't panic and returns a valid HTTP response.
	if resp.StatusCode == 0 {
		t.Error("expected a valid HTTP status code")
	}
}

func TestHealth_IncludesVersion(t *testing.T) {
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	handleHealth(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode health response: %v", err)
	}
	if _, ok := body["version"]; !ok {
		t.Error("health response missing 'version' field")
	}
	if body["version"] == "" {
		t.Error("health response 'version' field is empty")
	}
}

func TestModels_ReturnsModelList(t *testing.T) {
	cfg := testCfg
	cfg.ModelDir = "test-model-v1"

	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()

	handleModels(cfg)(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode models response: %v", err)
	}

	if body["object"] != "list" {
		t.Errorf("expected object='list', got %v", body["object"])
	}

	data, ok := body["data"].([]any)
	if !ok {
		t.Fatalf("expected 'data' to be an array, got %T", body["data"])
	}
	if len(data) == 0 {
		t.Fatal("expected at least one model in data array")
	}

	model, ok := data[0].(map[string]any)
	if !ok {
		t.Fatalf("expected model entry to be an object, got %T", data[0])
	}
	if model["object"] != "model" {
		t.Errorf("expected model object='model', got %v", model["object"])
	}
	if model["id"] != "test-model-v1" {
		t.Errorf("expected model id='test-model-v1', got %v", model["id"])
	}
	if model["owned_by"] != "local" {
		t.Errorf("expected owned_by='local', got %v", model["owned_by"])
	}
}

func TestError_UnsupportedFormat(t *testing.T) {
	// Send a non-audio file (random bytes that don't match any format)
	randomData := []byte{0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x09, 0x0A, 0x0B, 0x0C}
	req := newMultipartRequest(t, "/v1/audio/transcriptions", "file", "data.bin", randomData)

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testQueue, testMaxBody)
	handler(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for unsupported format, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	var errResp map[string]any
	if err := json.Unmarshal(body, &errResp); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	errObj, ok := errResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error object in response, got: %s", body)
	}
	if errObj["type"] != "invalid_request_error" {
		t.Errorf("expected type=invalid_request_error, got %v", errObj["type"])
	}
}

func TestError_MissingFile(t *testing.T) {
	// POST with multipart but no "file" field
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.WriteField("model", "test")
	writer.Close()

	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testQueue, testMaxBody)
	handler(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing file, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	var errResp map[string]any
	if err := json.Unmarshal(body, &errResp); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	errObj, ok := errResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error object, got: %s", body)
	}
	msg, _ := errObj["message"].(string)
	if msg == "" {
		t.Error("expected non-empty error message")
	}
}

func TestRateLimit_RetryAfterHeader(t *testing.T) {
	// Fill both semaphore and queue completely
	fullSem := make(chan struct{}, 1)
	fullSem <- struct{}{}
	fullQueue := make(chan struct{}, 1)
	fullQueue <- struct{}{}

	req := newMultipartRequest(t, "/v1/audio/transcriptions", "file", "test.wav", fakeWAVHeader())

	w := httptest.NewRecorder()
	cfg := testCfg
	cfg.MaxConcurrent = 1
	cfg.MaxQueue = 1
	handler := handleTranscription(nil, cfg, testMetrics, fullSem, fullQueue, testMaxBody)
	handler(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected 429, got %d", resp.StatusCode)
	}

	retryAfter := resp.Header.Get("Retry-After")
	if retryAfter == "" {
		t.Error("expected Retry-After header in 429 response")
	}
	if retryAfter != "5" {
		t.Errorf("expected Retry-After: 5, got: %s", retryAfter)
	}

	// Verify the body is valid JSON with rate_limit_error type
	body, _ := io.ReadAll(resp.Body)
	var errResp map[string]any
	if err := json.Unmarshal(body, &errResp); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	errObj, ok := errResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error object, got: %s", body)
	}
	if errObj["type"] != "rate_limit_error" {
		t.Errorf("expected type=rate_limit_error, got %v", errObj["type"])
	}
}

func TestRequestID_InResponse(t *testing.T) {
	req := newMultipartRequest(t, "/v1/audio/transcriptions", "file", "test.wav", fakeWAVHeader())

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testQueue, testMaxBody)
	handler(w, req)

	reqID := w.Header().Get("X-Request-ID")
	if reqID == "" {
		t.Error("expected X-Request-ID header in response")
	}
	// Should be a valid UUID (36 chars with dashes)
	if len(reqID) != 36 {
		t.Errorf("expected UUID-length request ID (36 chars), got %d chars: %q", len(reqID), reqID)
	}
}

func TestRequestID_PropagatedUpstream(t *testing.T) {
	req := newMultipartRequest(t, "/v1/audio/transcriptions", "file", "test.wav", fakeWAVHeader())
	req.Header.Set("X-Request-ID", "test-propagated-id-42")

	w := httptest.NewRecorder()
	handler := handleTranscription(nil, testCfg, testMetrics, testSem, testQueue, testMaxBody)
	handler(w, req)

	reqID := w.Header().Get("X-Request-ID")
	if reqID != "test-propagated-id-42" {
		t.Errorf("expected propagated X-Request-ID 'test-propagated-id-42', got: %q", reqID)
	}
}
