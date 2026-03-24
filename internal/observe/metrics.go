package observe

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all Prometheus metrics for the STT server.
// Passed as a dependency rather than using package-level globals.
type Metrics struct {
	// Counters
	RequestsTotal  *prometheus.CounterVec
	AudioBytesTotal prometheus.Counter

	// Gauges
	RequestsInProgress prometheus.Gauge
	RequestsQueued     prometheus.Gauge
	ModelInfo          *prometheus.GaugeVec
	BuildInfo          *prometheus.GaugeVec

	// Histograms
	RequestDuration   prometheus.Histogram
	DecodeDuration    prometheus.Histogram
	InferenceDuration prometheus.Histogram
	AudioDuration     prometheus.Histogram
}

// NewMetrics creates and registers all Prometheus metrics.
func NewMetrics() *Metrics {
	return &Metrics{
		RequestsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "stt_requests_total",
			Help: "Total transcription requests by status and detected language.",
		}, []string{"status", "lang"}),

		AudioBytesTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "stt_audio_bytes_total",
			Help: "Total bytes of audio uploaded.",
		}),

		RequestsInProgress: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "stt_requests_in_progress",
			Help: "Number of transcription requests currently being processed.",
		}),

		RequestsQueued: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "stt_requests_queued",
			Help: "Number of requests waiting for a processing slot.",
		}),

		ModelInfo: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "stt_model_info",
			Help: "Static model metadata. Value is always 1.",
		}, []string{"model", "provider", "threads"}),

		BuildInfo: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "stt_build_info",
			Help: "Build information. Value is always 1.",
		}, []string{"version", "commit", "build_time"}),

		RequestDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "stt_request_duration_seconds",
			Help:    "End-to-end request latency (upload + decode + inference).",
			Buckets: []float64{0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30},
		}),

		DecodeDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "stt_decode_duration_seconds",
			Help:    "Audio decode time (ffmpeg format conversion).",
			Buckets: []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5},
		}),

		InferenceDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "stt_inference_duration_seconds",
			Help:    "Model inference time only (excluding audio decode).",
			Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
		}),

		AudioDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "stt_audio_duration_seconds",
			Help:    "Duration of audio processed per request.",
			Buckets: []float64{1, 5, 10, 30, 60, 120, 300, 600},
		}),
	}
}
