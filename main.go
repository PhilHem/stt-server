package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
)

type Config struct {
	ModelDir   string
	Port       int
	NumThreads int
	Provider   string // "cpu" or "cuda"
}

func main() {
	var model, cacheDir, logFormat, logLevel, otelEndpoint string
	cfg := Config{}

	flag.StringVar(&model, "model", envOr("STT_MODEL", ""), "Model name or path (env: STT_MODEL)")
	flag.StringVar(&cacheDir, "cache-dir", envOr("STT_CACHE_DIR", ""), "Model cache directory (env: STT_CACHE_DIR)")
	flag.IntVar(&cfg.Port, "port", envInt("STT_PORT", 8000), "HTTP listen port (env: STT_PORT)")
	flag.IntVar(&cfg.NumThreads, "num-threads", envInt("STT_NUM_THREADS", runtime.NumCPU()), "Inference threads (env: STT_NUM_THREADS)")
	flag.StringVar(&cfg.Provider, "provider", envOr("STT_PROVIDER", "cpu"), "ONNX Runtime provider: cpu or cuda (env: STT_PROVIDER)")
	flag.StringVar(&logFormat, "log-format", envOr("STT_LOG_FORMAT", "text"), "Log format: text, json, or journal (env: STT_LOG_FORMAT)")
	flag.StringVar(&logLevel, "log-level", envOr("STT_LOG_LEVEL", "info"), "Log level: debug, info, warn, error (env: STT_LOG_LEVEL)")
	flag.StringVar(&otelEndpoint, "otel-endpoint", envOr("OTEL_EXPORTER_OTLP_ENDPOINT", ""), "OTLP gRPC endpoint for traces, e.g. localhost:4317 (env: OTEL_EXPORTER_OTLP_ENDPOINT)")
	flag.Parse()

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
	)

	srv := newServer(recognizer, cfg.Port, filepath.Base(cfg.ModelDir))
	slog.Info("listening", "port", cfg.Port)
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("server stopped", "error", err)
		os.Exit(1)
	}
}
