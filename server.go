package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

func newServer(rec *Recognizer, port int) *http.Server {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", handleHealth)
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

		// Parse multipart form (max 100 MB)
		if err := r.ParseMultipartForm(100 << 20); err != nil {
			httpError(w, r, http.StatusBadRequest, "invalid multipart form: %v", err)
			return
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			httpError(w, r, http.StatusBadRequest, "missing 'file' field: %v", err)
			return
		}
		defer file.Close()

		audioData, err := io.ReadAll(file)
		if err != nil {
			httpError(w, r, http.StatusInternalServerError, "failed to read file: %v", err)
			return
		}

		// Convert to 16kHz mono PCM via ffmpeg
		samples, sampleRate, err := decodeAudio(audioData, header.Filename)
		if err != nil {
			httpError(w, r, http.StatusBadRequest, "audio decode failed: %v", err)
			return
		}

		result := rec.Transcribe(samples, sampleRate)

		elapsed := time.Since(start)
		slog.Info("transcribed",
			"file", header.Filename,
			"size_bytes", len(audioData),
			"duration_s", fmt.Sprintf("%.1f", result.Duration),
			"elapsed_ms", elapsed.Milliseconds(),
			"lang", result.Language,
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

func httpError(w http.ResponseWriter, r *http.Request, code int, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	slog.Warn("request error",
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
			"code":    nil,
		},
	})
}
