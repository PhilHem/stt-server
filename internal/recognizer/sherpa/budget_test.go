package sherpa

import (
	"math"
	"testing"

	"golang.org/x/sync/semaphore"
)

func newTestBudget(budgetGB float64) *Budget {
	b := &Budget{
		budgetGB: budgetGB,
		k:        0.0260417,
		base:     1.75,
		safety:   1.35,
		floorGB:  0.5,
		hardCap:  16,
	}
	b.capMilli = int64(math.Round(budgetGB * milliPerGB))
	b.sem = semaphore.NewWeighted(b.capMilli)
	return b
}

// The whole point of the design: no decode batch may ever be predicted to cost
// more than the budget, for ANY audio length. This builds a 4.5 h-equivalent
// segment list (using a tiny sample rate so it stays cheap) and asserts every
// planned batch fits — and that the cost is keyed on the LONGEST stream, the one
// detail whose absence would re-introduce OOM.
func TestPlanBatchesNeverExceedsBudget(t *testing.T) {
	const sr = 100 // seconds-domain math is sample-rate independent; keep memory trivial
	for _, budgetGB := range []float64{1.0, 4.0, 6.0, 12.0} {
		b := newTestBudget(budgetGB)
		e := &Engine{budget: b}

		maxWin := int(math.Floor(b.maxWindowSeconds()))
		if maxWin < 1 {
			t.Fatalf("budget %.1f: maxWindowSeconds %.2f < 1", budgetGB, b.maxWindowSeconds())
		}

		// 4.5 h of audio pre-split into windows of varied length up to maxWin.
		var segs []segment
		total := 0
		const totalSec = 16200
		for total < totalSec {
			lenSec := maxWin
			if total%3 == 0 && maxWin > 5 { // mix in shorter segments
				lenSec = maxWin/2 + 1
			}
			segs = append(segs, segment{start: total * sr, samples: make([]float32, lenSec*sr), speaker: -1})
			total += lenSec
		}

		segs = e.ensureFitsBudget(segs, sr)
		groups := e.planBatches(segs, sr)
		if len(groups) == 0 {
			t.Fatalf("budget %.1f: no batches planned for %d segments", budgetGB, len(segs))
		}

		covered := 0
		for _, g := range groups {
			if g.j <= g.i {
				t.Fatalf("budget %.1f: non-advancing batch [%d,%d)", budgetGB, g.i, g.j)
			}
			covered += g.j - g.i
			// recompute the true cost from the longest stream in the batch
			longest := 0.0
			for k := g.i; k < g.j; k++ {
				if s := float64(len(segs[k].samples)) / float64(sr); s > longest {
					longest = s
				}
			}
			if c := b.cost(g.j-g.i, longest); c > b.budgetGB+1e-9 {
				t.Errorf("budget %.1f: batch [%d,%d) n=%d longest=%.1fs predicted cost %.3f > budget %.3f",
					budgetGB, g.i, g.j, g.j-g.i, longest, c, b.budgetGB)
			}
		}
		if covered != len(segs) {
			t.Errorf("budget %.1f: batches cover %d of %d segments", budgetGB, covered, len(segs))
		}
	}
}

func TestMaxWindowAndBatchFitBudget(t *testing.T) {
	b := newTestBudget(6.0)
	mw := b.maxWindowSeconds()
	// a single stream of exactly maxWindow must fit at batch size 1
	if c := b.cost(1, mw); c > b.budgetGB+1e-9 {
		t.Errorf("cost(1, maxWindow=%.1f)=%.3f exceeds budget %.3f", mw, c, b.budgetGB)
	}
	// maxBatchFor(L) streams of length L must fit
	for _, L := range []float64{10, 30, 60, mw} {
		n := b.maxBatchFor(L)
		if n < 1 {
			t.Errorf("maxBatchFor(%.0f)=%d < 1", L, n)
		}
		if c := b.cost(n, L); c > b.budgetGB+1e-9 {
			t.Errorf("cost(maxBatchFor(%.0f)=%d, %.0f)=%.3f exceeds budget %.3f", L, n, L, c, b.budgetGB)
		}
	}
}

func TestNilBudgetIsNoOp(t *testing.T) {
	var b *Budget
	if b.cost(8, 120) != 0 {
		t.Error("nil budget cost should be 0")
	}
	if !math.IsInf(b.maxWindowSeconds(), 1) {
		t.Error("nil budget maxWindowSeconds should be +Inf")
	}
	if b.maxBatchFor(120) < 1<<20 {
		t.Error("nil budget maxBatchFor should be effectively unbounded")
	}
	if err := b.acquire(nil, 5); err != nil {
		t.Errorf("nil budget acquire should be no-op, got %v", err)
	}
	b.release(5) // must not panic
}
