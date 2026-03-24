package recognizer

import (
	"context"
	"fmt"
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
}

// Config holds the parameters needed to create a Recognizer.
type Config struct {
	ModelDir   string
	NumThreads int
	Provider   string // "cpu" or "cuda"
}

// Recognizer wraps sherpa-onnx offline recognition with mutex serialization.
// A WaitGroup tracks in-flight inference goroutines so Close() can wait
// for them before destroying the C object (prevents use-after-free).
type Recognizer struct {
	inner *sherpa.OfflineRecognizer
	mu    sync.Mutex
	wg    sync.WaitGroup
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
	if err := detectModel(&config, cfg.ModelDir); err != nil {
		return nil, err
	}

	r := sherpa.NewOfflineRecognizer(&config)
	if r == nil {
		return nil, fmt.Errorf("sherpa-onnx failed to create recognizer (check model files)")
	}

	return &Recognizer{inner: r}, nil
}

// Transcribe runs speech recognition on the given audio samples.
func (r *Recognizer) Transcribe(ctx context.Context, samples []float32, sampleRate int) (*TranscriptionResult, error) {
	type result struct {
		res *TranscriptionResult
	}
	ch := make(chan result, 1)

	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		r.mu.Lock()
		defer r.mu.Unlock()

		stream := sherpa.NewOfflineStream(r.inner)
		defer sherpa.DeleteOfflineStream(stream)

		stream.AcceptWaveform(sampleRate, samples)

		inferStart := time.Now()
		r.inner.Decode(stream)
		inferElapsed := time.Since(inferStart)

		out := stream.GetResult()
		ch <- result{res: &TranscriptionResult{
			Text:          strings.TrimSpace(out.Text),
			Language:      out.Lang,
			Duration:      float32(len(samples)) / float32(sampleRate),
			InferenceTime: inferElapsed,
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
// destroying the C object. Prevents use-after-free on shutdown.
func (r *Recognizer) Close() {
	r.wg.Wait()
	sherpa.DeleteOfflineRecognizer(r.inner)
}
