package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// --- Counters ---

	requestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "stt_requests_total",
		Help: "Total transcription requests by status.",
	}, []string{"status"})

	audioBytesTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "stt_audio_bytes_total",
		Help: "Total bytes of audio uploaded.",
	})

	// --- Gauges ---

	requestsInProgress = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "stt_requests_in_progress",
		Help: "Number of transcription requests currently being processed.",
	})

	modelInfo = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "stt_model_info",
		Help: "Static model metadata. Value is always 1.",
	}, []string{"model", "provider", "threads"})

	// --- Histograms ---

	requestDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "stt_request_duration_seconds",
		Help:    "End-to-end request latency (upload + decode + inference).",
		Buckets: []float64{0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30},
	})

	decodeDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "stt_decode_duration_seconds",
		Help:    "Audio decode time (ffmpeg format conversion).",
		Buckets: []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5},
	})

	inferenceDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "stt_inference_duration_seconds",
		Help:    "Model inference time only (excluding audio decode).",
		Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
	})

	audioDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "stt_audio_duration_seconds",
		Help:    "Duration of audio processed per request.",
		Buckets: []float64{1, 5, 10, 30, 60, 120, 300, 600},
	})
)
