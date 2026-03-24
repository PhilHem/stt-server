package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const requestIDHeader = "X-Request-ID"

var tracer = otel.Tracer("stt-server")

func newServer(rec *Recognizer, cfg Config) *http.Server {
	sem := make(chan struct{}, cfg.MaxConcurrent)
	maxBodyBytes := int64(cfg.MaxFileSizeMB) << 20

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handleHealth)
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.HandleFunc("GET /v1/models", handleModels(cfg))
	mux.HandleFunc("POST /v1/audio/transcriptions", handleTranscription(rec, cfg, sem, maxBodyBytes))

	return &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      mux,
		ReadTimeout:  cfg.RequestTimeout + 10*time.Second, // slightly more than request timeout
		WriteTimeout: cfg.RequestTimeout + 10*time.Second,
	}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleModels(cfg Config) http.HandlerFunc {
	resp := map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id":       cfg.ModelDir,
			"object":   "model",
			"owned_by": "local",
		}},
	}
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func handleTranscription(rec *Recognizer, cfg Config, sem chan struct{}, maxBodyBytes int64) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Request timeout
		ctx, cancel := context.WithTimeout(r.Context(), cfg.RequestTimeout)
		defer cancel()

		// Extract W3C traceparent from upstream, or start a new trace
		ctx = otel.GetTextMapPropagator().Extract(ctx, propagatorCarrier(r.Header))
		ctx, span := tracer.Start(ctx, "transcribe",
			trace.WithSpanKind(trace.SpanKindServer),
		)
		defer span.End()

		start := time.Now()

		// Propagate or generate request ID
		reqID := r.Header.Get(requestIDHeader)
		if reqID == "" {
			reqID = r.Header.Get("X-Litellm-Call-Id")
		}
		if reqID == "" {
			reqID = uuid.NewString()
		}
		w.Header().Set(requestIDHeader, reqID)
		span.SetAttributes(attribute.String("request.id", reqID))

		// Inject trace context into response
		otel.GetTextMapPropagator().Inject(ctx, propagatorCarrier(w.Header()))

		// Concurrency limit — reject with 503 if all slots full
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
		default:
			requestsTotal.WithLabelValues("503", "").Inc()
			span.SetStatus(codes.Error, "too many requests")
			httpError(w, r, ctx, reqID, http.StatusServiceUnavailable,
				"server busy: %d concurrent requests in progress", cfg.MaxConcurrent)
			return
		}

		requestsInProgress.Inc()
		defer requestsInProgress.Dec()

		// Limit request body size
		r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)

		// Parse multipart form
		if err := r.ParseMultipartForm(maxBodyBytes); err != nil {
			requestsTotal.WithLabelValues("400", "").Inc()
			span.SetStatus(codes.Error, "invalid multipart form")
			httpError(w, r, ctx, reqID, http.StatusBadRequest, "invalid multipart form: %v", err)
			return
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			requestsTotal.WithLabelValues("400", "").Inc()
			span.SetStatus(codes.Error, "missing file field")
			httpError(w, r, ctx, reqID, http.StatusBadRequest, "missing 'file' field: %v", err)
			return
		}
		defer file.Close()

		audioData, err := io.ReadAll(file)
		if err != nil {
			requestsTotal.WithLabelValues("400", "").Inc()
			span.SetStatus(codes.Error, "read file failed")
			httpError(w, r, ctx, reqID, http.StatusBadRequest, "file too large or read error: %v", err)
			return
		}

		span.SetAttributes(
			attribute.String("audio.filename", header.Filename),
			attribute.Int("audio.size_bytes", len(audioData)),
		)

		// Check context before expensive operations
		if ctx.Err() != nil {
			requestsTotal.WithLabelValues("408", "").Inc()
			span.SetStatus(codes.Error, "request timeout")
			httpError(w, r, ctx, reqID, http.StatusRequestTimeout, "request timed out")
			return
		}

		// Convert to 16kHz mono PCM via ffmpeg
		_, decodeSpan := tracer.Start(ctx, "audio.decode")
		decodeStart := time.Now()
		samples, sampleRate, err := decodeAudio(audioData, header.Filename)
		decodeElapsed := time.Since(decodeStart)
		decodeSpan.End()

		if err != nil {
			requestsTotal.WithLabelValues("400", "").Inc()
			span.SetStatus(codes.Error, "audio decode failed")
			httpError(w, r, ctx, reqID, http.StatusBadRequest, "audio decode failed: %v", err)
			return
		}
		decodeDuration.Observe(decodeElapsed.Seconds())

		// Enforce max audio duration
		audioDur := float32(len(samples)) / float32(sampleRate)
		if audioDur > float32(cfg.MaxAudioDuration.Seconds()) {
			requestsTotal.WithLabelValues("413", "").Inc()
			span.SetStatus(codes.Error, "audio too long")
			httpError(w, r, ctx, reqID, http.StatusRequestEntityTooLarge,
				"audio duration %.0fs exceeds limit of %ds", audioDur, int(cfg.MaxAudioDuration.Seconds()))
			return
		}

		// Check context before inference
		if ctx.Err() != nil {
			requestsTotal.WithLabelValues("408", "").Inc()
			span.SetStatus(codes.Error, "request timeout")
			httpError(w, r, ctx, reqID, http.StatusRequestTimeout, "request timed out before inference")
			return
		}

		// Inference
		_, inferSpan := tracer.Start(ctx, "model.inference")
		result := rec.Transcribe(samples, sampleRate)
		inferSpan.End()

		elapsed := time.Since(start)
		lang := result.Language
		if lang == "" {
			lang = "unknown"
		}

		span.SetAttributes(
			attribute.Float64("audio.duration_s", float64(result.Duration)),
			attribute.Int64("elapsed_ms", elapsed.Milliseconds()),
			attribute.Int64("inference_ms", result.InferenceTime.Milliseconds()),
			attribute.Int64("decode_ms", decodeElapsed.Milliseconds()),
			attribute.String("lang", lang),
		)
		span.SetStatus(codes.Ok, "")

		// Record metrics
		requestsTotal.WithLabelValues("200", lang).Inc()
		requestDuration.Observe(elapsed.Seconds())
		inferenceDuration.Observe(result.InferenceTime.Seconds())
		audioDuration.Observe(float64(result.Duration))
		audioBytesTotal.Add(float64(len(audioData)))

		// Log with trace context
		logAttrs := append([]any{
			"request_id", reqID,
			"file", header.Filename,
			"size_bytes", len(audioData),
			"duration_s", fmt.Sprintf("%.1f", result.Duration),
			"elapsed_ms", elapsed.Milliseconds(),
			"decode_ms", decodeElapsed.Milliseconds(),
			"inference_ms", result.InferenceTime.Milliseconds(),
			"lang", lang,
		}, traceAttrs(ctx)...)
		slog.Info("transcribed", logAttrs...)

		// Response format (OpenAI-compatible)
		responseFormat := r.FormValue("response_format")
		switch responseFormat {
		case "text":
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			fmt.Fprint(w, result.Text)
		case "verbose_json":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"text":     result.Text,
				"language": lang,
				"duration": result.Duration,
			})
		default:
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{
				"text": result.Text,
			})
		}
	}
}

func httpError(w http.ResponseWriter, r *http.Request, ctx context.Context, reqID string, code int, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	logAttrs := append([]any{
		"request_id", reqID,
		"status", code,
		"method", r.Method,
		"path", r.URL.Path,
		"error", msg,
	}, traceAttrs(ctx)...)
	slog.Warn("request error", logAttrs...)

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

// propagatorCarrier adapts http.Header for OTel text map propagation.
type propagatorCarrier http.Header

func (c propagatorCarrier) Get(key string) string { return http.Header(c).Get(key) }
func (c propagatorCarrier) Set(key, value string) { http.Header(c).Set(key, value) }
func (c propagatorCarrier) Keys() []string {
	keys := make([]string, 0, len(c))
	for k := range c {
		keys = append(keys, k)
	}
	return keys
}
