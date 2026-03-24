package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/PhilHem/stt-server/internal/audio"
	"github.com/PhilHem/stt-server/internal/config"
	"github.com/PhilHem/stt-server/internal/observe"
	"github.com/PhilHem/stt-server/internal/recognizer"
)

const requestIDHeader = "X-Request-ID"

var tracer = otel.Tracer("stt-server")

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "ok",
		"version": config.Version,
	})
}

func handleModels(cfg config.Config) http.HandlerFunc {
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

func handleTranscription(rec *recognizer.Recognizer, cfg config.Config, m *observe.Metrics, sem chan struct{}, queue chan struct{}, maxBodyBytes int64) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Request timeout — applies to queue wait + processing
		ctx, cancel := context.WithTimeout(r.Context(), cfg.RequestTimeout)
		defer cancel()

		// Try to get a processing slot immediately
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
		default:
			// All slots busy — try to enter the queue
			select {
			case queue <- struct{}{}:
				// In queue — wait for a slot
				m.RequestsQueued.Inc()
				select {
				case sem <- struct{}{}:
					<-queue // leave queue, enter processing
					m.RequestsQueued.Dec()
					defer func() { <-sem }()
				case <-ctx.Done():
					<-queue
					m.RequestsQueued.Dec()
					m.RequestsTotal.WithLabelValues("408", "").Inc()
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusRequestTimeout)
					json.NewEncoder(w).Encode(map[string]any{
						"error": map[string]any{
							"message": "request timed out while queued",
							"type":    "server_error",
							"code":    "408",
						},
					})
					return
				}
			default:
				// Queue full — reject with 429 + Retry-After
				m.RequestsTotal.WithLabelValues("429", "").Inc()
				w.Header().Set("Retry-After", "5")
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusTooManyRequests)
				json.NewEncoder(w).Encode(map[string]any{
					"error": map[string]any{
						"message": fmt.Sprintf("server busy: %d requests processing, %d queued", cfg.MaxConcurrent, cfg.MaxQueue),
						"type":    "rate_limit_error",
						"code":    "429",
					},
				})
				return
			}
		}

		// Extract W3C traceparent from upstream, or start a new trace
		ctx = otel.GetTextMapPropagator().Extract(ctx, propagatorCarrier(r.Header))
		ctx, span := tracer.Start(ctx, "transcribe",
			trace.WithSpanKind(trace.SpanKindServer),
		)
		defer span.End()

		start := time.Now()
		m.RequestsInProgress.Inc()
		defer m.RequestsInProgress.Dec()

		// Propagate or generate request ID
		reqID := r.Header.Get(requestIDHeader)
		if reqID == "" {
			reqID = r.Header.Get("X-Litellm-Call-Id")
		}
		if reqID == "" {
			reqID = uuid.NewString()
		}
		reqID = sanitizeRequestID(reqID)
		w.Header().Set(requestIDHeader, reqID)
		span.SetAttributes(attribute.String("request.id", reqID))

		// Inject trace context into response
		otel.GetTextMapPropagator().Inject(ctx, propagatorCarrier(w.Header()))

		// Limit request body size
		r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)

		// Parse multipart form
		if err := r.ParseMultipartForm(maxBodyBytes); err != nil {
			m.RequestsTotal.WithLabelValues("400", "").Inc()
			span.SetStatus(codes.Error, "invalid multipart form")
			slog.Debug("multipart parse failed", "error", err)
			httpError(w, r, ctx, reqID, http.StatusBadRequest, observe.TraceAttrs, "invalid request body")
			return
		}
		if r.MultipartForm != nil {
			defer r.MultipartForm.RemoveAll()
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			m.RequestsTotal.WithLabelValues("400", "").Inc()
			span.SetStatus(codes.Error, "missing file field")
			httpError(w, r, ctx, reqID, http.StatusBadRequest, observe.TraceAttrs, "missing 'file' field")
			return
		}
		defer file.Close()

		audioData, err := io.ReadAll(file)
		if err != nil {
			m.RequestsTotal.WithLabelValues("400", "").Inc()
			span.SetStatus(codes.Error, "read file failed")
			slog.Debug("file read failed", "error", err)
			httpError(w, r, ctx, reqID, http.StatusBadRequest, observe.TraceAttrs, "file too large or unreadable")
			return
		}

		safeFilename := sanitizeRequestID(header.Filename) // reuse same sanitizer
		span.SetAttributes(
			attribute.String("audio.filename", safeFilename),
			attribute.Int("audio.size_bytes", len(audioData)),
		)

		// Convert to 16kHz mono PCM via ffmpeg
		_, decodeSpan := tracer.Start(ctx, "audio.decode")
		decodeStart := time.Now()
		samples, sampleRate, err := audio.Decode(ctx, audioData, header.Filename)
		decodeElapsed := time.Since(decodeStart)
		decodeSpan.End()

		if err != nil {
			m.RequestsTotal.WithLabelValues("400", "").Inc()
			span.SetStatus(codes.Error, "audio decode failed")
			httpError(w, r, ctx, reqID, http.StatusBadRequest, observe.TraceAttrs, "audio decode failed: %v", err)
			return
		}
		m.DecodeDuration.Observe(decodeElapsed.Seconds())

		// Enforce max audio duration
		audioDur := float32(len(samples)) / float32(sampleRate)
		if audioDur > float32(cfg.MaxAudioDuration.Seconds()) {
			m.RequestsTotal.WithLabelValues("413", "").Inc()
			span.SetStatus(codes.Error, "audio too long")
			httpError(w, r, ctx, reqID, http.StatusRequestEntityTooLarge, observe.TraceAttrs,
				"audio duration %.0fs exceeds limit of %ds", audioDur, int(cfg.MaxAudioDuration.Seconds()))
			return
		}

		// Inference
		_, inferSpan := tracer.Start(ctx, "model.inference")
		result, err := rec.Transcribe(ctx, samples, sampleRate)
		inferSpan.End()

		if err != nil {
			m.RequestsTotal.WithLabelValues("408", "").Inc()
			span.SetStatus(codes.Error, "inference failed")
			slog.Debug("transcription failed", "error", err)
			httpError(w, r, ctx, reqID, http.StatusRequestTimeout, observe.TraceAttrs, "transcription timed out")
			return
		}

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
		m.RequestsTotal.WithLabelValues("200", lang).Inc()
		m.RequestDuration.Observe(elapsed.Seconds())
		m.InferenceDuration.Observe(result.InferenceTime.Seconds())
		m.AudioDuration.Observe(float64(result.Duration))
		m.AudioBytesTotal.Add(float64(len(audioData)))

		// Log with trace context
		logAttrs := append([]any{
			"request_id", reqID,
			"file", safeFilename,
			"size_bytes", len(audioData),
			"duration_s", fmt.Sprintf("%.1f", result.Duration),
			"elapsed_ms", elapsed.Milliseconds(),
			"decode_ms", decodeElapsed.Milliseconds(),
			"inference_ms", result.InferenceTime.Milliseconds(),
			"lang", lang,
		}, observe.TraceAttrs(ctx)...)
		slog.Info("transcribed", logAttrs...)

		// Build word and segment timing data from tokens/timestamps
		var words []Word
		var segments []Segment
		if len(result.Tokens) > 0 && len(result.Timestamps) > 0 {
			starts, ends := tokenTimesToStartEnd(result.Timestamps)
			words = buildWords(result.Tokens, starts, ends)
			segments = buildSegments(words)
		}

		// Response format (OpenAI-compatible)
		responseFormat := r.FormValue("response_format")
		switch responseFormat {
		case "text":
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			fmt.Fprint(w, result.Text)
		case "verbose_json":
			w.Header().Set("Content-Type", "application/json")
			resp := map[string]any{
				"task":     "transcribe",
				"text":     result.Text,
				"language": lang,
				"duration": result.Duration,
			}
			if len(words) > 0 {
				resp["words"] = words
				resp["segments"] = segments
			}
			json.NewEncoder(w).Encode(resp)
		case "srt":
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			fmt.Fprint(w, formatSRT(segments))
		case "vtt":
			w.Header().Set("Content-Type", "text/vtt; charset=utf-8")
			fmt.Fprint(w, formatVTT(segments))
		default:
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{
				"text": result.Text,
			})
		}
	}
}
