package sherpa

// The offline recognizer crashes when handed a single very long utterance, so
// long audio is cut into windows before recognition. Each window is around
// chunkTargetSeconds long and the cut is nudged to the quietest sample within
// ±chunkSearchSeconds of the boundary so the split lands in a pause rather
// than the middle of a word.
const (
	chunkTargetSeconds = 120
	chunkSearchSeconds = 5

	// energyFrameSamples is the width of the amplitude window used to score a
	// candidate cut point (10 ms at 16 kHz).
	energyFrameSamples = 160
)

// windowRange is a half-open [start, end) range of sample indices.
type windowRange struct{ start, end int }

// splitWindows divides samples into consecutive, non-overlapping windows of
// roughly targetSeconds each, moving every interior cut to the lowest-energy
// sample within ±searchSeconds of the nominal boundary. Audio at or below the
// target size yields a single full-length window. The returned ranges are
// contiguous and exactly cover [0, len(samples)).
func splitWindows(samples []float32, sampleRate, targetSeconds, searchSeconds int) []windowRange {
	n := len(samples)
	targetSamples := targetSeconds * sampleRate
	if targetSamples <= 0 || n <= targetSamples {
		return []windowRange{{0, n}}
	}
	searchSamples := searchSeconds * sampleRate

	var windows []windowRange
	start := 0
	for start < n {
		target := start + targetSamples
		if target >= n {
			windows = append(windows, windowRange{start, n})
			break
		}
		end := quietestCut(samples, target, searchSamples)
		if end <= start { // never emit an empty or backward window
			end = target
		}
		windows = append(windows, windowRange{start, end})
		start = end
	}
	return windows
}

// quietestCut returns the sample index near target (within ±search, clamped to
// the signal) whose short surrounding frame has the lowest mean amplitude —
// i.e. the closest thing to a pause. It falls back to target when no quieter
// point is found.
func quietestCut(samples []float32, target, search int) int {
	n := len(samples)
	lo := target - search
	if lo < 1 {
		lo = 1
	}
	hi := target + search
	if hi > n {
		hi = n
	}

	step := energyFrameSamples / 2
	if step < 1 {
		step = 1
	}

	best := target
	bestEnergy := float32(-1)
	for i := lo; i < hi; i += step {
		a := i - energyFrameSamples/2
		if a < 0 {
			a = 0
		}
		b := i + energyFrameSamples/2
		if b > n {
			b = n
		}
		var sum float32
		for _, s := range samples[a:b] {
			if s < 0 {
				s = -s
			}
			sum += s
		}
		energy := sum / float32(b-a)
		if bestEnergy < 0 || energy < bestEnergy {
			bestEnergy = energy
			best = i
		}
	}
	return best
}
