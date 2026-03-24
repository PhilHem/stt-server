package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"
)

type Config struct {
	ModelDir          string
	Port              int
	NumThreads        int
	Provider          string // "cpu" or "cuda"
	MaxConcurrent     int
	MaxFileSizeMB     int
	MaxAudioDuration  time.Duration
	RequestTimeout    time.Duration
}

func main() {
	var model, cacheDir, logFormat, logLevel, otelEndpoint string
	var maxAudioSec, requestTimeoutSec int
	cfg := Config{}

	flag.StringVar(&model, "model", envOr("STT_MODEL", ""), "Model name or path (env: STT_MODEL)")
	flag.StringVar(&cacheDir, "cache-dir", envOr("STT_CACHE_DIR", ""), "Model cache directory (env: STT_CACHE_DIR)")
	flag.IntVar(&cfg.Port, "port", envInt("STT_PORT", 8000), "HTTP listen port (env: STT_PORT)")
	flag.IntVar(&cfg.NumThreads, "num-threads", envInt("STT_NUM_THREADS", runtime.NumCPU()), "Inference threads (env: STT_NUM_THREADS)")
	flag.StringVar(&cfg.Provider, "provider", envOr("STT_PROVIDER", "cpu"), "ONNX Runtime provider: cpu or cuda (env: STT_PROVIDER)")
	flag.StringVar(&logFormat, "log-format", envOr("STT_LOG_FORMAT", "text"), "Log format: text, json, or journal (env: STT_LOG_FORMAT)")
	flag.StringVar(&logLevel, "log-level", envOr("STT_LOG_LEVEL", "info"), "Log level: debug, info, warn, error (env: STT_LOG_LEVEL)")
	flag.StringVar(&otelEndpoint, "otel-endpoint", envOr("OTEL_EXPORTER_OTLP_ENDPOINT", ""), "OTLP gRPC endpoint for traces (env: OTEL_EXPORTER_OTLP_ENDPOINT)")
	flag.IntVar(&cfg.MaxConcurrent, "max-concurrent", envInt("STT_MAX_CONCURRENT", 4), "Max concurrent transcription requests (env: STT_MAX_CONCURRENT)")
	flag.IntVar(&cfg.MaxFileSizeMB, "max-file-size", envInt("STT_MAX_FILE_SIZE_MB", 100), "Max upload file size in MB (env: STT_MAX_FILE_SIZE_MB)")
	flag.IntVar(&maxAudioSec, "max-audio-duration", envInt("STT_MAX_AUDIO_DURATION", 600), "Max audio duration in seconds (env: STT_MAX_AUDIO_DURATION)")
	flag.IntVar(&requestTimeoutSec, "request-timeout", envInt("STT_REQUEST_TIMEOUT", 300), "Request timeout in seconds (env: STT_REQUEST_TIMEOUT)")
	flag.Parse()

	cfg.MaxAudioDuration = time.Duration(maxAudioSec) * time.Second
	cfg.RequestTimeout = time.Duration(requestTimeoutSec) * time.Second

	if err := setupLogging(logFormat, logLevel); err != nil {
		slog.Error("invalid logging config", "error", err)
		os.Exit(1)
	}

	// Initialize tracing (noop if endpoint is empty)
	ctx := context.Background()
	shutdownTracing, err := setupTracing(ctx, otelEndpoint, "stt-server")
	if err != nil {
		slog.Error("failed to setup tracing", "error", err)
		os.Exit(1)
	}
	defer shutdownTracing(ctx)

	if model == "" {
		slog.Error("--model or STT_MODEL is required")
		os.Exit(1)
	}

	modelDir, err := resolveModel(model, cacheDir)
	if err != nil {
		slog.Error("failed to resolve model", "error", err)
		os.Exit(1)
	}
	cfg.ModelDir = modelDir

	recognizer, err := newRecognizer(cfg)
	if err != nil {
		slog.Error("failed to load model", "error", err)
		os.Exit(1)
	}
	defer recognizer.Close()

	// Publish static model info as Prometheus gauge
	modelInfo.WithLabelValues(
		filepath.Base(cfg.ModelDir),
		cfg.Provider,
		fmt.Sprintf("%d", cfg.NumThreads),
	).Set(1)

	slog.Info("model loaded",
		"path", cfg.ModelDir,
		"threads", cfg.NumThreads,
		"provider", cfg.Provider,
		"max_concurrent", cfg.MaxConcurrent,
		"max_file_size_mb", cfg.MaxFileSizeMB,
		"max_audio_duration_s", int(cfg.MaxAudioDuration.Seconds()),
		"request_timeout_s", int(cfg.RequestTimeout.Seconds()),
	)

	srv := newServer(recognizer, cfg)

	// Graceful shutdown: drain in-flight requests on SIGTERM/SIGINT
	shutdownCtx, stop := signal.NotifyContext(ctx, syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	go func() {
		<-shutdownCtx.Done()
		slog.Info("shutting down, draining in-flight requests...")
		drainCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := srv.Shutdown(drainCtx); err != nil {
			slog.Error("shutdown error", "error", err)
		}
	}()

	slog.Info("listening", "port", cfg.Port)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		slog.Error("server stopped", "error", err)
		os.Exit(1)
	}
	slog.Info("server stopped gracefully")
}
