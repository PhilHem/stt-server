package recognizer

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// TranscriptionResult holds the output of a transcription.
type TranscriptionResult struct {
	Text          string
	Language      string
	Duration      float32       // audio duration in seconds
	InferenceTime time.Duration // model inference time
	Tokens        []string      // per-token text
	Timestamps    []float32     // per-token start time in seconds
}

// Config holds the parameters needed to create a Recognizer.
type Config struct {
	ModelDir   string
	NumThreads int
	Provider   string // "cpu" or "cuda"
	Language   string // ISO-639-1 hint (used by Whisper/SenseVoice, ignored by Parakeet)
}

// Recognizer wraps sherpa-onnx offline recognition with mutex serialization.
// A WaitGroup tracks in-flight inference goroutines so Close() can wait
// for them before destroying the C object (prevents use-after-free).
type Recognizer struct {
	inner     *sherpa.OfflineRecognizer
	mu        sync.Mutex
	wg        sync.WaitGroup
	closed    bool
	ModelType string // detected model type (e.g. "nemo_transducer", "whisper")
}

// New creates a Recognizer from the given configuration.
func New(cfg Config) (*Recognizer, error) {
	config := sherpa.OfflineRecognizerConfig{}
	config.FeatConfig.SampleRate = 16000
	config.FeatConfig.FeatureDim = 80
	config.DecodingMethod = "greedy_search"
	config.ModelConfig.NumThreads = cfg.NumThreads
	config.ModelConfig.Provider = cfg.Provider

	// Auto-detect model type from files present in the directory
	modelType, err := detectModel(&config, cfg.ModelDir, cfg.Language)
	if err != nil {
		return nil, err
	}

	r := sherpa.NewOfflineRecognizer(&config)
	if r == nil {
		return nil, fmt.Errorf("sherpa-onnx failed to create recognizer (check model files)")
	}

	return &Recognizer{inner: r, ModelType: modelType}, nil
}

// Transcribe runs speech recognition on the given audio samples.
func (r *Recognizer) Transcribe(ctx context.Context, samples []float32, sampleRate int) (*TranscriptionResult, error) {
	if len(samples) == 0 {
		return &TranscriptionResult{Duration: 0}, nil
	}

	type result struct {
		res *TranscriptionResult
	}
	ch := make(chan result, 1)

	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		r.mu.Lock()
		defer r.mu.Unlock()

		if r.closed {
			ch <- result{res: &TranscriptionResult{}}
			return
		}

		stream := sherpa.NewOfflineStream(r.inner)
		defer sherpa.DeleteOfflineStream(stream)

		stream.AcceptWaveform(sampleRate, samples)

		inferStart := time.Now()
		r.inner.Decode(stream)
		inferElapsed := time.Since(inferStart)

		out := stream.GetResult()
		var text, lang string
		var tokens []string
		var timestamps []float32
		if out != nil {
			text = strings.TrimSpace(out.Text)
			lang = out.Lang
			if len(out.Tokens) > 0 {
				tokens = out.Tokens
			}
			if len(out.Timestamps) > 0 {
				timestamps = out.Timestamps
			}
		}
		ch <- result{res: &TranscriptionResult{
			Text:          text,
			Language:      lang,
			Duration:      float32(len(samples)) / float32(sampleRate),
			InferenceTime: inferElapsed,
			Tokens:        tokens,
			Timestamps:    timestamps,
		}}
	}()

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("inference timed out: %w", ctx.Err())
	case r := <-ch:
		return r.res, nil
	}
}

// Close waits for all in-flight inference goroutines to finish before
// destroying the C object. If goroutines are still inside CGo after the
// timeout, the C memory is intentionally leaked to avoid use-after-free.
// This is safe because Close is only called during shutdown and the OS
// will reclaim the memory when the process exits.
func (r *Recognizer) Close() {
	done := make(chan struct{})
	go func() {
		r.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// All goroutines finished — safe to free
		sherpa.DeleteOfflineRecognizer(r.inner)
	case <-time.After(30 * time.Second):
		// Goroutines still running inside CGo. Freeing r.inner would cause
		// use-after-free. Intentionally leak the C memory — the process is
		// shutting down and the OS will reclaim it.
		r.mu.Lock()
		r.closed = true
		r.mu.Unlock()
		slog.Warn("shutdown: leaked recognizer C object (goroutines still in CGo)")
	}
}
