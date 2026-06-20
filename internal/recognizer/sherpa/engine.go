package sherpa

import (
	"context"
	"fmt"
	"log/slog"
	"math"
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
	fastDiar  *httpDiarizer                 // nil unless a fast (Sortformer) service is configured
	fastMax   float64                       // max audio seconds for the fast diarizer (0 = no cap)
	mu        sync.Mutex
	wg        sync.WaitGroup
	closed    bool
	modelType string
	budget    *Budget // process-global VRAM admission controller; nil = unbounded
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

	eng := &Engine{inner: r, modelType: modelType, budget: sharedBudget()}

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
	// a request can opt in (see Transcribe's diar options). The general-purpose
	// diarizer (pyannote) handles arbitrary speaker counts; the fast diarizer
	// (Sortformer) is used when the caller hints 1..4 speakers.
	if cfg.DiarizeURL != "" {
		eng.diar = newHTTPDiarizer(cfg.DiarizeURL)
		slog.Info("speaker diarization available", "url", cfg.DiarizeURL)
	}
	if cfg.FastDiarizeURL != "" {
		eng.fastDiar = newHTTPDiarizer(cfg.FastDiarizeURL)
		eng.fastMax = float64(cfg.FastDiarizeMaxSeconds)
		slog.Info("fast speaker diarization available", "url", cfg.FastDiarizeURL, "max_seconds", cfg.FastDiarizeMaxSeconds)
	}

	return eng, nil
}

// Transcribe runs speech recognition on the given audio samples.
func (e *Engine) Transcribe(ctx context.Context, samples []float32, sampleRate int, diar recognizer.DiarizeOptions) (*recognizer.TranscriptionResult, error) {
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
		durationSec := float64(len(samples)) / float64(sampleRate)
		if diarizer := e.pickDiarizer(diar, durationSec); diarizer != nil {
			ds, err := diarizer.segments(ctx, samples, sampleRate)
			// The fast diarizer (Sortformer) can OOM on long audio; rather than
			// drop straight to an unlabelled transcript, retry with the
			// general-purpose diarizer (pyannote) which has bounded memory.
			if err != nil && diarizer == e.fastDiar && e.diar != nil {
				slog.Warn("fast diarization failed; retrying with general-purpose diarizer", "error", err)
				ds, err = e.diar.segments(ctx, samples, sampleRate)
			}
			if err != nil {
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
			// Pre-cap the window so a single stream never exceeds the VRAM budget
			// (no-op when unbounded). Long files just yield more, smaller windows.
			target := chunkTargetSeconds
			if e.budget != nil {
				if mw := int(math.Floor(e.budget.maxWindowSeconds())); mw >= 1 && mw < target {
					target = mw
				}
			}
			for _, w := range splitWindows(samples, sampleRate, target, chunkSearchSeconds) {
				segs = append(segs, segment{start: w.start, samples: samples[w.start:w.end], speaker: -1})
			}
		}

		// Sub-split any segment (e.g. a long diarized/VAD turn) that alone would
		// exceed the budget, so every decoded stream fits at batch size 1.
		segs = e.ensureFitsBudget(segs, sampleRate)

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

// pickDiarizer chooses the diarization backend for a request, or nil when
// diarization is not requested/available. A caller-supplied hint of 1..4
// speakers prefers the fast diarizer (Sortformer) — but only for audio short
// enough to fit its quadratic attention memory; longer recordings fall to the
// general-purpose diarizer (pyannote), which also handles arbitrary counts.
func (e *Engine) pickDiarizer(diar recognizer.DiarizeOptions, durationSec float64) *httpDiarizer {
	if !diar.Enabled {
		return nil
	}
	if diar.SpeakerCount >= 1 && diar.SpeakerCount <= 4 && e.fastDiar != nil &&
		(e.fastMax <= 0 || durationSec <= e.fastMax) {
		return e.fastDiar
	}
	return e.diar
}

// maxBatch bounds how many segments are decoded in one batched call. The batch
// pads to its longest stream and the fp32 encoder's activations scale with
// batch×length, so this caps peak GPU memory — kept small because the GPU is
// shared with other models and a failed allocation aborts the process. At a
// 120 s window this puts the peak working set around 11 GB, so long recordings
// decode within a bounded, length-independent VRAM budget that keeps a healthy
// margin on a shared GPU while recovering most of the throughput a larger batch
// would give; the peak is constant regardless of total audio length. This fixed
// cap is only the fallback used when VRAM admission control is disabled
// (STT_VRAM_BUDGET_MB unset); when a budget is set, planBatches sizes every
// batch to the budget instead (see budget.go).
const maxBatch = 4

// grp is a half-open [i,j) range of segments decoded together, carrying the
// predicted VRAM cost (GB) reserved from the budget for that batch.
type grp struct {
	i, j int
	cost float64
}

// planBatches groups segments into decode batches. Unbounded, it falls back to
// fixed maxBatch-sized groups (current behaviour). With a budget set it greedily
// grows each batch while the predicted cost (safety·max(floor, k·n·longestSec −
// base)) stays within budget and within the per-length batch limit — so peak
// VRAM is bounded no matter how many segments a long file produces.
func (e *Engine) planBatches(segs []segment, sampleRate int) []grp {
	var groups []grp
	if e.budget == nil {
		for _, b := range batchBounds(len(segs), maxBatch) {
			groups = append(groups, grp{i: b[0], j: b[1]})
		}
		return groups
	}
	secs := func(s segment) float64 { return float64(len(s.samples)) / float64(sampleRate) }
	i := 0
	for i < len(segs) {
		longest := 0.0
		j := i
		for j < len(segs) {
			cand := math.Max(longest, secs(segs[j]))
			n := j - i + 1
			if n > e.budget.maxBatchFor(cand) {
				break
			}
			if e.budget.cost(n, cand) > e.budget.budgetGB {
				break
			}
			longest = cand
			j++
		}
		if j == i { // always make progress; the window pre-cap guarantees 1 fits
			j = i + 1
			longest = secs(segs[i])
		}
		groups = append(groups, grp{i: i, j: j, cost: e.budget.cost(j-i, longest)})
		i = j
	}
	return groups
}

// ensureFitsBudget sub-splits any segment longer than the budget's max single
// stream so every decoded stream fits at batch size 1. No-op when unbounded; the
// default branch already pre-caps its own windows, so this mainly bounds long
// diarized/VAD turns.
func (e *Engine) ensureFitsBudget(segs []segment, sampleRate int) []segment {
	if e.budget == nil {
		return segs
	}
	maxWin := int(math.Floor(e.budget.maxWindowSeconds()))
	if maxWin < 1 {
		maxWin = 1
	}
	maxSamples := maxWin * sampleRate
	var out []segment
	for _, sg := range segs {
		if len(sg.samples) <= maxSamples {
			out = append(out, sg)
			continue
		}
		for _, w := range splitWindows(sg.samples, sampleRate, maxWin, chunkSearchSeconds) {
			out = append(out, segment{start: sg.start + w.start, samples: sg.samples[w.start:w.end], speaker: sg.speaker})
		}
	}
	return out
}

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

	for _, g := range e.planBatches(segs, sampleRate) {
		if ctx.Err() != nil { // cancelled or timed out — stop early
			break
		}
		// Reserve this batch's predicted VRAM from the global budget; blocks
		// (cancellable) until it frees, so the sum of concurrent decodes never
		// exceeds the budget and the onnxruntime arena's high-water mark stays
		// ≤ budget. No-op when unbounded.
		if err := e.budget.acquire(ctx, g.cost); err != nil {
			break // ctx cancelled/timed out while waiting for budget
		}
		batch := segs[g.i:g.j]

		streams := make([]*sherpa.OfflineStream, len(batch))
		for i, sg := range batch {
			st := sherpa.NewOfflineStream(e.inner)
			st.AcceptWaveform(sampleRate, sg.samples)
			streams[i] = st
		}

		inferStart := time.Now()
		e.inner.DecodeStreams(streams)
		inferTotal += time.Since(inferStart)
		e.budget.release(g.cost)

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
