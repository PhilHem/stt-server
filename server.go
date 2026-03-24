package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const requestIDHeader = "X-Request-ID"

func newServer(rec *Recognizer, port int) *http.Server {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", handleHealth)
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.HandleFunc("POST /v1/audio/transcriptions", handleTranscription(rec))

	return &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      mux,
		ReadTimeout:  300 * time.Second, // large audio uploads
		WriteTimeout: 300 * time.Second,
	}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleTranscription(rec *Recognizer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		requestsInProgress.Inc()
		defer requestsInProgress.Dec()

		// Propagate or generate request ID for cross-service correlation.
		// LiteLLM sends x-litellm-call-id; Caddy can forward X-Request-ID.
		reqID := r.Header.Get(requestIDHeader)
		if reqID == "" {
			reqID = r.Header.Get("X-Litellm-Call-Id")
		}
		if reqID == "" {
			reqID = uuid.NewString()
		}
		w.Header().Set(requestIDHeader, reqID)

		// Parse multipart form (max 100 MB)
		if err := r.ParseMultipartForm(100 << 20); err != nil {
			requestsTotal.WithLabelValues("400", "").Inc()
			httpError(w, r, reqID, http.StatusBadRequest, "invalid multipart form: %v", err)
			return
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			requestsTotal.WithLabelValues("400", "").Inc()
			httpError(w, r, reqID, http.StatusBadRequest, "missing 'file' field: %v", err)
			return
		}
		defer file.Close()

		audioData, err := io.ReadAll(file)
		if err != nil {
			requestsTotal.WithLabelValues("500", "").Inc()
			httpError(w, r, reqID, http.StatusInternalServerError, "failed to read file: %v", err)
			return
		}

		// Convert to 16kHz mono PCM via ffmpeg
		decodeStart := time.Now()
		samples, sampleRate, err := decodeAudio(audioData, header.Filename)
		decodeElapsed := time.Since(decodeStart)
		if err != nil {
			requestsTotal.WithLabelValues("400", "").Inc()
			httpError(w, r, reqID, http.StatusBadRequest, "audio decode failed: %v", err)
			return
		}
		decodeDuration.Observe(decodeElapsed.Seconds())

		result := rec.Transcribe(samples, sampleRate)

		elapsed := time.Since(start)
		lang := result.Language
		if lang == "" {
			lang = "unknown"
		}

		// Record metrics
		requestsTotal.WithLabelValues("200", lang).Inc()
		requestDuration.Observe(elapsed.Seconds())
		inferenceDuration.Observe(result.InferenceTime.Seconds())
		audioDuration.Observe(float64(result.Duration))
		audioBytesTotal.Add(float64(len(audioData)))

		slog.Info("transcribed",
			"request_id", reqID,
			"file", header.Filename,
			"size_bytes", len(audioData),
			"duration_s", fmt.Sprintf("%.1f", result.Duration),
			"elapsed_ms", elapsed.Milliseconds(),
			"decode_ms", decodeElapsed.Milliseconds(),
			"inference_ms", result.InferenceTime.Milliseconds(),
			"lang", lang,
		)

		// Response format (OpenAI-compatible)
		responseFormat := r.FormValue("response_format")
		if responseFormat == "text" {
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			fmt.Fprint(w, result.Text)
			return
		}

		// Default: JSON
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"text": result.Text,
		})
	}
}

func httpError(w http.ResponseWriter, r *http.Request, reqID string, code int, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	slog.Warn("request error",
		"request_id", reqID,
		"status", code,
		"method", r.Method,
		"path", r.URL.Path,
		"error", msg,
	)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "invalid_request_error",
			"code":    strconv.Itoa(code),
		},
	})
}
