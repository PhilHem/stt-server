package sherpa

import sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

// VAD tuning. The decoder always receives 16 kHz mono audio, so the window and
// durations are expressed against that rate. MaxSpeechDuration caps a single
// speech run so no segment grows long enough to trouble the recognizer, and so
// batched streams stay a bounded size.
const (
	vadSampleRate = 16000
	vadWindowSize = 512 // Silero v5 processes 512-sample frames at 16 kHz
	vadThreshold  = 0.5
	vadMinSilence = 0.5  // seconds of silence that ends a segment
	vadMinSpeech  = 0.25 // seconds; drop blips shorter than this
	vadMaxSpeech  = 20.0 // seconds; force a cut in long monologues
	vadBufferSecs = 30.0 // detector ring-buffer capacity
)

// segment is a span of audio to transcribe, tagged with its sample offset on
// the original timeline so token timestamps can be shifted back. speaker is the
// diarization speaker index, or -1 when diarization did not run.
type segment struct {
	start   int
	samples []float32
	speaker int
}

// newVAD builds a Silero voice-activity detector from a model file. Returns
// nil when sherpa-onnx cannot construct it (e.g. missing/invalid model).
func newVAD(modelPath string) *sherpa.VoiceActivityDetector {
	cfg := &sherpa.VadModelConfig{
		SileroVad: sherpa.SileroVadModelConfig{
			Model:              modelPath,
			Threshold:          vadThreshold,
			MinSilenceDuration: vadMinSilence,
			MinSpeechDuration:  vadMinSpeech,
			WindowSize:         vadWindowSize,
			MaxSpeechDuration:  vadMaxSpeech,
		},
		SampleRate: vadSampleRate,
		NumThreads: 1, // the VAD model is tiny; one thread avoids contending with the recognizer
		Provider:   "cpu",
	}
	return sherpa.NewVoiceActivityDetector(cfg, vadBufferSecs)
}

// segmentByVAD runs the detector over the samples and returns the speech
// segments in chronological order, with silence dropped. The detector is reset
// first so it is safe to reuse across requests.
func segmentByVAD(vad *sherpa.VoiceActivityDetector, samples []float32) []segment {
	vad.Reset()

	var segs []segment
	drain := func() {
		for !vad.IsEmpty() {
			s := vad.Front()
			vad.Pop()
			segs = append(segs, segment{start: s.Start, samples: s.Samples})
		}
	}

	// The detector must be fed one window at a time; it reads past the buffer
	// and crashes on larger blocks. This is the cost of VAD on long audio and
	// is why VAD is opt-in.
	for off := 0; off < len(samples); off += vadWindowSize {
		end := off + vadWindowSize
		if end > len(samples) {
			end = len(samples)
		}
		vad.AcceptWaveform(samples[off:end])
		drain()
	}
	vad.Flush()
	drain()

	return segs
}

// batchBounds splits total items into [start,end) ranges of at most max each.
func batchBounds(total, max int) [][2]int {
	if max < 1 {
		max = 1
	}
	var out [][2]int
	for start := 0; start < total; start += max {
		end := start + max
		if end > total {
			end = total
		}
		out = append(out, [2]int{start, end})
	}
	return out
}
