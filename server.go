package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
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
			httpError(w, http.StatusBadRequest, "invalid multipart form: %v", err)
			return
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			httpError(w, http.StatusBadRequest, "missing 'file' field: %v", err)
			return
		}
		defer file.Close()

		audioData, err := io.ReadAll(file)
		if err != nil {
			httpError(w, http.StatusInternalServerError, "failed to read file: %v", err)
			return
		}

		// Convert to 16kHz mono PCM via ffmpeg
		samples, sampleRate, err := decodeAudio(audioData, header.Filename)
		if err != nil {
			httpError(w, http.StatusBadRequest, "audio decode failed: %v", err)
			return
		}

		result := rec.Transcribe(samples, sampleRate)

		elapsed := time.Since(start)
		log.Printf("transcribed %.1fs audio in %s (%s)", result.Duration, elapsed.Round(time.Millisecond), header.Filename)

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

func httpError(w http.ResponseWriter, code int, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	log.Printf("HTTP %d: %s", code, msg)
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
