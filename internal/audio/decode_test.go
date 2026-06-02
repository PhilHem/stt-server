package audio

import (
	"math"
	"testing"
)

func TestPcmToFloat32(t *testing.T) {
	// 16-bit signed LE: 0x0000 = 0, 0xFF7F = 32767, 0x0180 = -32768
	raw := []byte{
		0x00, 0x00, // 0
		0xFF, 0x7F, // 32767 (max positive)
		0x00, 0x80, // -32768 (max negative)
	}

	samples := pcmToFloat32(raw)

	if len(samples) != 3 {
		t.Fatalf("expected 3 samples, got %d", len(samples))
	}

	if samples[0] != 0.0 {
		t.Errorf("sample[0]: expected 0.0, got %f", samples[0])
	}

	// 32767 / 32767 = 1.0
	if math.Abs(float64(samples[1])-1.0) > 0.001 {
		t.Errorf("sample[1]: expected ~1.0, got %f", samples[1])
	}

	// -32768 / 32767 = -1.0
	if math.Abs(float64(samples[2])+1.0) > 0.001 {
		t.Errorf("sample[2]: expected ~-1.0, got %f", samples[2])
	}
}

func TestPcmToFloat32_Empty(t *testing.T) {
	samples := pcmToFloat32([]byte{})
	if len(samples) != 0 {
		t.Errorf("expected 0 samples for empty input, got %d", len(samples))
	}
}

func TestInputExt(t *testing.T) {
	cases := map[string]string{
		"Microphone (2026-03-12 09.25.05).m4a": ".m4a",
		"recording.MP3":                        ".mp3",
		"clip.wav":                             ".wav",
		"no-extension":                         "",
		"":                                     "",
		"weird.name.flac":                      ".flac",
		"trailing.dot.":                        "",
		"too.longextension":                    "", // > 8 chars after the dot
		"bad.ex!t":                             "", // non-alphanumeric
		"archive.tar.gz":                       ".gz",
	}
	for filename, want := range cases {
		if got := inputExt(filename); got != want {
			t.Errorf("inputExt(%q) = %q, want %q", filename, got, want)
		}
	}
}

func TestDecodedByteBudget(t *testing.T) {
	// A positive budget gets one second of headroom, two bytes per sample.
	maxSamples := 600 * TargetSampleRate
	want := (maxSamples + TargetSampleRate) * bytesPerSample
	if got := decodedByteBudget(maxSamples); got != want {
		t.Errorf("decodedByteBudget(%d) = %d, want %d", maxSamples, got, want)
	}

	// Zero or negative falls back to the default cap (plus the same headroom).
	wantFallback := (fallbackMaxSamples + TargetSampleRate) * bytesPerSample
	for _, in := range []int{0, -1} {
		if got := decodedByteBudget(in); got != wantFallback {
			t.Errorf("decodedByteBudget(%d) = %d, want fallback %d", in, got, wantFallback)
		}
	}
}
