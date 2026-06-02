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
	vad       *sherpa.VoiceActivityDetector // nil unless a VAD model is configured
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

	eng := &Engine{inner: r, modelType: modelType}

	// Optional voice-activity detector: when present, long audio is split on
	// real speech boundaries instead of by fixed windows.
	if cfg.VadModel != "" {
		vad := newVAD(cfg.VadModel)
		if vad == nil {
			sherpa.DeleteOfflineRecognizer(r)
			return nil, fmt.Errorf("failed to create VAD from %s", cfg.VadModel)
		}
		eng.vad = vad
		slog.Info("VAD segmentation enabled", "model", cfg.VadModel)
	}

	return eng, nil
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

		// Long audio is segmented before recognition; the offline recognizer
		// crashes on a single very long utterance. With a VAD the cuts land on
		// real speech boundaries (silence dropped); otherwise we fall back to
		// fixed windows cut at the quietest point. Segments are then decoded in
		// batches so the encoder runs on the GPU with full occupancy.
		var segs []segment
		if e.vad != nil {
			segs = segmentByVAD(e.vad, samples)
		} else {
			for _, w := range splitWindows(samples, sampleRate, chunkTargetSeconds, chunkSearchSeconds) {
				segs = append(segs, segment{start: w.start, samples: samples[w.start:w.end]})
			}
		}

		res := e.decodeBatched(ctx, segs, sampleRate)
		res.Duration = float32(len(samples)) / float32(sampleRate)
		ch <- result{res: res}
	}()

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("inference timed out: %w", ctx.Err())
	case r := <-ch:
		return r.res, nil
	}
}

// maxBatch bounds how many segments are decoded in one batched call. The batch
// pads to its longest stream, so this caps peak GPU memory.
const maxBatch = 16

// decodeBatched decodes the segments in batches via DecodeStreams (the encoder
// runs once per batch on the GPU) and joins the results in chronological order.
// Per-segment token timestamps are shifted onto the original timeline.
func (e *Engine) decodeBatched(ctx context.Context, segs []segment, sampleRate int) *recognizer.TranscriptionResult {
	var (
		texts      []string
		tokens     []string
		timestamps []float32
		lang       string
		inferTotal time.Duration
	)

	for _, b := range batchBounds(len(segs), maxBatch) {
		if ctx.Err() != nil { // cancelled or timed out — stop early
			break
		}
		batch := segs[b[0]:b[1]]

		streams := make([]*sherpa.OfflineStream, len(batch))
		for i, sg := range batch {
			st := sherpa.NewOfflineStream(e.inner)
			st.AcceptWaveform(sampleRate, sg.samples)
			streams[i] = st
		}

		inferStart := time.Now()
		e.inner.DecodeStreams(streams)
		inferTotal += time.Since(inferStart)

		for i, st := range streams {
			out := st.GetResult()
			if out != nil {
				if t := strings.TrimSpace(out.Text); t != "" {
					texts = append(texts, t)
				}
				if lang == "" {
					lang = out.Lang
				}
				tokens = append(tokens, out.Tokens...)
				offset := float32(batch[i].start) / float32(sampleRate)
				for _, ts := range out.Timestamps {
					timestamps = append(timestamps, ts+offset)
				}
			}
			sherpa.DeleteOfflineStream(st)
		}
	}

	return &recognizer.TranscriptionResult{
		Text:          strings.Join(texts, " "),
		Language:      lang,
		InferenceTime: inferTotal,
		Tokens:        tokens,
		Timestamps:    timestamps,
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
		if e.vad != nil {
			sherpa.DeleteVoiceActivityDetector(e.vad)
		}
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
