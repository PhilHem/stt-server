package sherpa

import "testing"

// fill returns n samples all set to v.
func fill(n int, v float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = v
	}
	return s
}

func TestSplitWindows_ShortAudioSingleWindow(t *testing.T) {
	samples := fill(5000, 0.5)
	got := splitWindows(samples, 16000, 120, 5)
	if len(got) != 1 || got[0] != (windowRange{0, 5000}) {
		t.Fatalf("expected one full window {0 5000}, got %v", got)
	}
}

func TestSplitWindows_ContiguousAndComplete(t *testing.T) {
	const sr = 16000
	samples := fill(sr*7, 0.5) // 7 seconds, windowed at 2s
	windows := splitWindows(samples, sr, 2, 1)

	if len(windows) < 2 {
		t.Fatalf("expected multiple windows, got %d", len(windows))
	}
	if windows[0].start != 0 {
		t.Errorf("first window must start at 0, got %d", windows[0].start)
	}
	last := windows[len(windows)-1]
	if last.end != len(samples) {
		t.Errorf("last window must end at %d, got %d", len(samples), last.end)
	}
	for i := 1; i < len(windows); i++ {
		if windows[i].start != windows[i-1].end {
			t.Errorf("windows not contiguous at %d: prev end %d, start %d",
				i, windows[i-1].end, windows[i].start)
		}
		if windows[i].end <= windows[i].start {
			t.Errorf("window %d is empty or backward: %v", i, windows[i])
		}
	}
}

func TestSplitWindows_CutsAtQuietGap(t *testing.T) {
	const sr = 16000
	// 3 seconds loud, with a silent gap straddling the 2s boundary.
	samples := fill(sr*3, 0.8)
	gapStart, gapEnd := sr*2-300, sr*2+300
	for i := gapStart; i < gapEnd; i++ {
		samples[i] = 0
	}

	windows := splitWindows(samples, sr, 2, 1) // target 2s, ±1s search
	if len(windows) < 2 {
		t.Fatalf("expected a split, got %d windows", len(windows))
	}
	cut := windows[0].end
	if cut < gapStart || cut > gapEnd {
		t.Errorf("cut %d did not land in the silent gap [%d, %d]", cut, gapStart, gapEnd)
	}
}

func TestQuietestCut_FallsBackToTargetOnFlatSignal(t *testing.T) {
	const sr = 16000
	samples := fill(sr*3, 0.5) // perfectly flat — no quieter point exists
	target := sr * 2
	cut := quietestCut(samples, target, sr)
	// On a flat signal every frame ties; the running minimum keeps the first
	// candidate, which is the low end of the search window, never past target.
	if cut < target-sr || cut > target {
		t.Errorf("cut %d outside expected search range [%d, %d]", cut, target-sr, target)
	}
}
