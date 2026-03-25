package server

import (
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/PhilHem/stt-server/internal/config"
	"github.com/PhilHem/stt-server/internal/observe"
	"github.com/PhilHem/stt-server/internal/recognizer"
)

// New creates an HTTP server with all routes wired up.
func New(pool *recognizer.Pool, cfg config.Config, m *observe.Metrics) *http.Server {
	sem := make(chan struct{}, cfg.MaxConcurrent)
	queue := make(chan struct{}, cfg.MaxQueue) // queue depth limit
	maxBodyBytes := int64(cfg.MaxFileSizeMB) << 20

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handleHealth)
	mux.HandleFunc("GET /ready", handleReady(pool))
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.HandleFunc("GET /v1/models", handleModels(cfg))
	mux.HandleFunc("POST /v1/audio/transcriptions", handleTranscription(pool, cfg, m, sem, queue, maxBodyBytes))

	return &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.Port),
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       cfg.RequestTimeout + 10*time.Second,
		WriteTimeout:      cfg.RequestTimeout + 10*time.Second,
		IdleTimeout:       60 * time.Second,
	}
}
