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
	diar      *httpDiarizer                 // nil unless a diarization service is configured
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

	// Optional speaker diarization via an external GPU service. When configured,
	// a request can opt in (see Transcribe's diarize flag).
	if cfg.DiarizeURL != "" {
		eng.diar = newHTTPDiarizer(cfg.DiarizeURL)
		slog.Info("speaker diarization available", "url", cfg.DiarizeURL)
	}

	return eng, nil
}

// Transcribe runs speech recognition on the given audio samples.
func (e *Engine) Transcribe(ctx context.Context, samples []float32, sampleRate int, diarize bool) (*recognizer.TranscriptionResult, error) {
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
		// crashes on a single very long utterance. Diarization (when configured)
		// splits on per-speaker turns and tags each segment with its speaker; a
		// VAD splits on speech boundaries; otherwise we fall back to fixed
		// windows cut at the quietest point. Segments are then decoded in
		// batches so the encoder runs on the GPU with full occupancy.
		var segs []segment
		diarized := false
		if diarize && e.diar != nil {
			if ds, err := e.diar.segments(ctx, samples, sampleRate); err != nil {
				// Don't fail the whole request on a diarization hiccup — fall back
				// to a plain transcript without speaker labels.
				slog.Warn("diarization failed; transcribing without speakers", "error", err)
			} else {
				segs = ds
				diarized = true
			}
		}
		switch {
		case diarized:
			// segs already built (per-speaker turns)
		case e.vad != nil:
			segs = segmentByVAD(e.vad, samples)
			for i := range segs {
				segs[i].speaker = -1
			}
		default:
			for _, w := range splitWindows(samples, sampleRate, chunkTargetSeconds, chunkSearchSeconds) {
				segs = append(segs, segment{start: w.start, samples: samples[w.start:w.end], speaker: -1})
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
// pads to its longest stream and the fp32 encoder's activations scale with
// batch×length, so this caps peak GPU memory — kept small because the GPU is
// shared with other models and a failed allocation aborts the process.
const maxBatch = 6

// decodeBatched decodes the segments in batches via DecodeStreams (the encoder
// runs once per batch on the GPU) and joins the results in chronological order.
// Per-segment token timestamps are shifted onto the original timeline.
func (e *Engine) decodeBatched(ctx context.Context, segs []segment, sampleRate int) *recognizer.TranscriptionResult {
	type decoded struct {
		speaker    int
		start, end float32
		text       string
	}
	var (
		texts      []string
		tokens     []string
		timestamps []float32
		results    []decoded
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
				sg := batch[i]
				text := strings.TrimSpace(out.Text)
				if text != "" {
					texts = append(texts, text)
				}
				if lang == "" {
					lang = out.Lang
				}
				tokens = append(tokens, out.Tokens...)
				offset := float32(sg.start) / float32(sampleRate)
				for _, ts := range out.Timestamps {
					timestamps = append(timestamps, ts+offset)
				}
				results = append(results, decoded{
					speaker: sg.speaker,
					start:   offset,
					end:     float32(sg.start+len(sg.samples)) / float32(sampleRate),
					text:    text,
				})
			}
			sherpa.DeleteOfflineStream(st)
		}
	}

	// Merge consecutive same-speaker segments into turns. Only when diarization
	// ran (speaker >= 0); otherwise there are no per-speaker turns to report.
	var turns []recognizer.SpeakerTurn
	for _, d := range results {
		if d.speaker < 0 || d.text == "" {
			continue
		}
		if n := len(turns); n > 0 && turns[n-1].Speaker == d.speaker {
			turns[n-1].Text += " " + d.text
			turns[n-1].End = d.end
		} else {
			turns = append(turns, recognizer.SpeakerTurn{
				Speaker: d.speaker, Start: d.start, End: d.end, Text: d.text,
			})
		}
	}

	return &recognizer.TranscriptionResult{
		Text:          strings.Join(texts, " "),
		Language:      lang,
		InferenceTime: inferTotal,
		Tokens:        tokens,
		Timestamps:    timestamps,
		Segments:      turns,
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
