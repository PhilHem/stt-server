package sherpa

import (
	"context"
	"log/slog"
	"math"
	"os"
	"strconv"
	"sync"

	"golang.org/x/sync/semaphore"
)

// Budget is the process-global GPU-memory admission controller. It is the only
// graceful lever available: the onnxruntime CUDA arena cannot be capped through
// sherpa-onnx, and an over-budget allocation does not throttle — it throws an
// Ort::Exception that crosses the cgo boundary as an uncatchable process abort.
// So the engine must PREVENT any decode (and the sum of concurrent decodes) from
// ever exceeding a configured budget.
//
// The invariant that makes this OOM-proof for ANY audio length: onnxruntime's
// BFC arena grows to its largest single-allocation high-water mark and retains
// it. That peak is governed by the padded batch span (batchCount × longest
// stream seconds), NOT by total audio length — sherpa pads every stream in a
// batch to the longest. So if every DecodeStreams call is predicted (with a
// safety factor) to cost ≤ budget and we refuse anything larger, the arena can
// never be forced past budget. A 4.5 h file just produces more sequential
// batches — more wall-clock, never more peak VRAM.
//
// A nil *Budget means "unbounded" (the cost knobs are unset): every method is a
// no-op and the engine behaves exactly as before. Enable by setting
// STT_VRAM_BUDGET_MB.
type Budget struct {
	sem      *semaphore.Weighted
	capMilli int64 // semaphore capacity in milli-GB == round(budgetGB*1000)
	budgetGB float64
	k        float64 // GB of arena peak per (batch × longest-second)
	base     float64 // GB intercept, subtracted
	safety   float64 // multiplier covering conv-workspace spikes, padding, fragmentation, calibration error
	floorGB  float64
	hardCap  int // absolute batch ceiling regardless of free budget
}

const milliPerGB = 1000.0

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

var (
	budgetOnce sync.Once
	budgetInst *Budget
)

// sharedBudget returns the process-global budget, constructed once from the
// environment. Returns nil (unbounded) when STT_VRAM_BUDGET_MB is unset or ≤ 0.
func sharedBudget() *Budget {
	budgetOnce.Do(func() {
		mb := envInt("STT_VRAM_BUDGET_MB", 0)
		if mb <= 0 {
			return // nil => disabled, current behaviour
		}
		b := &Budget{
			budgetGB: float64(mb) / 1000.0,
			k:        envFloat("STT_VRAM_K_GB_PER_BATCHSEC", 0.0260417),
			base:     envFloat("STT_VRAM_BASE_GB", 1.75),
			safety:   math.Max(1.0, envFloat("STT_VRAM_SAFETY", 1.35)),
			floorGB:  envFloat("STT_VRAM_FLOOR_GB", 0.5),
			hardCap:  envInt("STT_VRAM_BATCH_HARDCAP", 16),
		}
		b.capMilli = int64(math.Round(b.budgetGB * milliPerGB))
		if b.capMilli < 1 {
			b.capMilli = 1
		}
		b.sem = semaphore.NewWeighted(b.capMilli)
		budgetInst = b
		slog.Info("VRAM admission control enabled",
			"budget_gb", b.budgetGB, "k_gb_per_batchsec", b.k, "base_gb", b.base,
			"safety", b.safety, "batch_hardcap", b.hardCap,
			"max_window_s", math.Floor(b.maxWindowSeconds()))
	})
	return budgetInst
}

// cost is the conservative predicted peak VRAM (GB) of decoding n streams whose
// longest is longestSec seconds. Clamped to budgetGB so a single batch can
// always eventually acquire the whole semaphore.
func (b *Budget) cost(n int, longestSec float64) float64 {
	if b == nil {
		return 0
	}
	c := b.safety * math.Max(b.floorGB, b.k*float64(n)*longestSec-b.base)
	if c > b.budgetGB {
		c = b.budgetGB
	}
	return c
}

// maxWindowSeconds is the largest window W for which cost(1, W) == budget, i.e.
// the longest single utterance that fits at batch size 1. Windows are pre-capped
// to this so long files never present an over-budget single stream.
func (b *Budget) maxWindowSeconds() float64 {
	if b == nil {
		return math.Inf(1)
	}
	w := (b.budgetGB/b.safety + b.base) / b.k
	if w < 1 {
		w = 1
	}
	return w
}

// maxBatchFor is the largest batch size whose longest stream is longestSec that
// still fits the budget, clamped to [1, hardCap]. Callers pre-cap window length,
// so batch size 1 always fits.
func (b *Budget) maxBatchFor(longestSec float64) int {
	if b == nil {
		return 1 << 30
	}
	n := int(math.Floor((b.budgetGB/b.safety + b.base) / (b.k * longestSec)))
	if n < 1 {
		n = 1
	}
	if n > b.hardCap {
		n = b.hardCap
	}
	return n
}

// acquire reserves costGB of the global budget, blocking (cancellable) until it
// is free. Honors ctx so a request that cannot get budget within its timeout
// fails cleanly instead of crashing.
func (b *Budget) acquire(ctx context.Context, costGB float64) error {
	if b == nil {
		return nil
	}
	w := int64(math.Round(costGB * milliPerGB))
	if w > b.capMilli {
		w = b.capMilli
	}
	if w < 1 {
		w = 1
	}
	return b.sem.Acquire(ctx, w)
}

func (b *Budget) release(costGB float64) {
	if b == nil {
		return
	}
	w := int64(math.Round(costGB * milliPerGB))
	if w > b.capMilli {
		w = b.capMilli
	}
	if w < 1 {
		w = 1
	}
	b.sem.Release(w)
}
