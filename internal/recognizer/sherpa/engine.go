package sherpa

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/PhilHem/stt-server/internal/recognizer"
)

// Engine wraps sherpa-onnx offline recognition with mutex serialization.
// A WaitGroup tracks in-flight inference goroutines so Close() can wait
// for them before destroying the C object (prevents use-after-free).
type Engine struct {
	inner     *sherpa.OfflineRecognizer
	mu        sync.Mutex
	wg        sync.WaitGroup
	closed    bool
	modelType string
}

// New creates a sherpa-onnx Engine from the given configuration.
func New(cfg recognizer.Config) (recognizer.Engine, error) {
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

	return &Engine{inner: r, modelType: modelType}, nil
}

// Transcribe runs speech recognition on the given audio samples.
func (e *Engine) Transcribe(ctx context.Context, samples []float32, sampleRate int) (*recognizer.TranscriptionResult, error) {
	if len(samples) == 0 {
		return &recognizer.TranscriptionResult{Duration: 0}, nil
	}

	type result struct {
		res *recognizer.TranscriptionResult
	}
	ch := make(chan result, 1)

	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		e.mu.Lock()
		defer e.mu.Unlock()

		if e.closed {
			ch <- result{res: &recognizer.TranscriptionResult{}}
			return
		}

		stream := sherpa.NewOfflineStream(e.inner)
		defer sherpa.DeleteOfflineStream(stream)

		stream.AcceptWaveform(sampleRate, samples)

		inferStart := time.Now()
		e.inner.Decode(stream)
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
		ch <- result{res: &recognizer.TranscriptionResult{
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
func (e *Engine) Close() {
	done := make(chan struct{})
	go func() {
		e.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// All goroutines finished — safe to free
		sherpa.DeleteOfflineRecognizer(e.inner)
	case <-time.After(30 * time.Second):
		// Goroutines still running inside CGo. Freeing e.inner would cause
		// use-after-free. Intentionally leak the C memory — the process is
		// shutting down and the OS will reclaim it.
		e.mu.Lock()
		e.closed = true
		e.mu.Unlock()
		slog.Warn("shutdown: leaked recognizer C object (goroutines still in CGo)")
	}
}

// ModelType returns the detected model type.
func (e *Engine) ModelType() string {
	return e.modelType
}
