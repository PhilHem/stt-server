package main

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

	// 32767 / 32767 ≈ 1.0
	if math.Abs(float64(samples[1])-1.0) > 0.001 {
		t.Errorf("sample[1]: expected ~1.0, got %f", samples[1])
	}

	// -32768 / 32767 ≈ -1.0
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
