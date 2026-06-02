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

		// Long audio is split into windows; the offline recognizer crashes on a
		// single very long utterance. Each window is decoded on its own stream
		// and the texts are joined. Short audio yields one full-length window.
		windows := splitWindows(samples, sampleRate, chunkTargetSeconds, chunkSearchSeconds)

		var (
			texts      []string
			tokens     []string
			timestamps []float32
			lang       string
			inferTotal time.Duration
		)
		for _, w := range windows {
			if ctx.Err() != nil { // cancelled or timed out — stop early
				break
			}

			stream := sherpa.NewOfflineStream(e.inner)
			stream.AcceptWaveform(sampleRate, samples[w.start:w.end])

			inferStart := time.Now()
			e.inner.Decode(stream)
			inferTotal += time.Since(inferStart)

			out := stream.GetResult()
			if out != nil {
				if t := strings.TrimSpace(out.Text); t != "" {
					texts = append(texts, t)
				}
				if lang == "" {
					lang = out.Lang
				}
				tokens = append(tokens, out.Tokens...)
				// Window timestamps are relative to the window; shift them back
				// onto the original timeline.
				offset := float32(w.start) / float32(sampleRate)
				for _, ts := range out.Timestamps {
					timestamps = append(timestamps, ts+offset)
				}
			}
			sherpa.DeleteOfflineStream(stream)
		}

		ch <- result{res: &recognizer.TranscriptionResult{
			Text:          strings.Join(texts, " "),
			Language:      lang,
			Duration:      float32(len(samples)) / float32(sampleRate),
			InferenceTime: inferTotal,
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
