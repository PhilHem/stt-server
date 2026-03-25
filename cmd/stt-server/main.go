package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/PhilHem/stt-server/internal/config"
	"github.com/PhilHem/stt-server/internal/model"
	"github.com/PhilHem/stt-server/internal/observe"
	"github.com/PhilHem/stt-server/internal/recognizer"
	"github.com/PhilHem/stt-server/internal/server"
)

func main() {
	var modelName, cacheDir, logFormat, logLevel, otelEndpoint string
	var maxAudioSec, requestTimeoutSec int
	var showVersion bool
	cfg := config.Config{}

	flag.BoolVar(&showVersion, "version", false, "Print version and exit")
	flag.StringVar(&modelName, "model", config.EnvOr("STT_MODEL", ""), "Model name or path (env: STT_MODEL)")
	flag.StringVar(&cacheDir, "cache-dir", config.EnvOr("STT_CACHE_DIR", ""), "Model cache directory (env: STT_CACHE_DIR)")
	flag.IntVar(&cfg.Port, "port", config.EnvInt("STT_PORT", 8000), "HTTP listen port (env: STT_PORT)")
	flag.IntVar(&cfg.NumThreads, "num-threads", config.EnvInt("STT_NUM_THREADS", runtime.NumCPU()), "Inference threads (env: STT_NUM_THREADS)")
	flag.StringVar(&cfg.Provider, "provider", config.EnvOr("STT_PROVIDER", "cpu"), "ONNX Runtime provider: cpu or cuda (env: STT_PROVIDER)")
	flag.StringVar(&logFormat, "log-format", config.EnvOr("STT_LOG_FORMAT", "text"), "Log format: text, json, or journal (env: STT_LOG_FORMAT)")
	flag.StringVar(&logLevel, "log-level", config.EnvOr("STT_LOG_LEVEL", "info"), "Log level: debug, info, warn, error (env: STT_LOG_LEVEL)")
	flag.StringVar(&otelEndpoint, "otel-endpoint", config.EnvOr("OTEL_EXPORTER_OTLP_ENDPOINT", ""), "OTLP gRPC endpoint for traces (env: OTEL_EXPORTER_OTLP_ENDPOINT)")
	flag.IntVar(&cfg.MaxConcurrent, "max-concurrent", config.EnvInt("STT_MAX_CONCURRENT", 4), "Max concurrent transcription requests (env: STT_MAX_CONCURRENT)")
	flag.IntVar(&cfg.MaxQueue, "max-queue", config.EnvInt("STT_MAX_QUEUE", 8), "Max queued requests waiting for a slot (env: STT_MAX_QUEUE)")
	flag.IntVar(&cfg.MaxFileSizeMB, "max-file-size", config.EnvInt("STT_MAX_FILE_SIZE_MB", 100), "Max upload file size in MB (env: STT_MAX_FILE_SIZE_MB)")
	flag.IntVar(&maxAudioSec, "max-audio-duration", config.EnvInt("STT_MAX_AUDIO_DURATION", 600), "Max audio duration in seconds (env: STT_MAX_AUDIO_DURATION)")
	flag.IntVar(&requestTimeoutSec, "request-timeout", config.EnvInt("STT_REQUEST_TIMEOUT", 300), "Request timeout in seconds (env: STT_REQUEST_TIMEOUT)")
	flag.Parse()

	cfg.MaxAudioDuration = time.Duration(maxAudioSec) * time.Second
	cfg.RequestTimeout = time.Duration(requestTimeoutSec) * time.Second

	if showVersion {
		fmt.Printf("stt-server %s (commit: %s, built: %s)\n", config.Version, config.Commit, config.BuildTime)
		os.Exit(0)
	}

	if err := observe.SetupLogging(logFormat, logLevel); err != nil {
		slog.Error("invalid logging config", "error", err)
		os.Exit(1)
	}

	// Verify ffmpeg is available (required for audio decoding)
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		slog.Error("ffmpeg not found in PATH (required for audio decoding)")
		os.Exit(1)
	}

	// Initialize tracing (noop if endpoint is empty)
	ctx := context.Background()
	shutdownTracing, err := observe.SetupTracing(ctx, otelEndpoint, "stt-server")
	if err != nil {
		slog.Error("failed to setup tracing", "error", err)
		os.Exit(1)
	}
	defer shutdownTracing(ctx)

	if modelName == "" {
		slog.Error("--model or STT_MODEL is required")
		os.Exit(1)
	}

	modelDir, err := model.Resolve(modelName, cacheDir)
	if err != nil {
		slog.Error("failed to resolve model", "error", err)
		os.Exit(1)
	}
	cfg.ModelDir = modelDir

	rec, err := recognizer.New(recognizer.Config{
		ModelDir:   cfg.ModelDir,
		NumThreads: cfg.NumThreads,
		Provider:   cfg.Provider,
	})
	if err != nil {
		slog.Error("failed to load model", "error", err)
		os.Exit(1)
	}
	defer rec.Close()

	// Publish static info as Prometheus gauges
	metrics := observe.NewMetrics()
	metrics.BuildInfo.WithLabelValues(config.Version, config.Commit, config.BuildTime).Set(1)
	metrics.ModelInfo.WithLabelValues(
		filepath.Base(cfg.ModelDir),
		cfg.Provider,
		fmt.Sprintf("%d", cfg.NumThreads),
	).Set(1)

	slog.Info("model loaded",
		"version", config.Version,
		"model_type", rec.ModelType,
		"path", cfg.ModelDir,
		"threads", cfg.NumThreads,
		"provider", cfg.Provider,
		"max_concurrent", cfg.MaxConcurrent,
		"max_queue", cfg.MaxQueue,
		"max_file_size_mb", cfg.MaxFileSizeMB,
		"max_audio_duration_s", int(cfg.MaxAudioDuration.Seconds()),
		"request_timeout_s", int(cfg.RequestTimeout.Seconds()),
	)

	srv := server.New(rec, cfg, metrics)

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
