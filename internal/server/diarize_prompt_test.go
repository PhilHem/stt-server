package server

import "testing"

func TestParseSpeakerCount(t *testing.T) {
	cases := map[string]int{
		"":                 0,
		"speakers=2":       2,
		"speakers=4":       4,
		"  speakers=3  ":   3,
		"speakers=12":      12, // out-of-range hints are passed through; routing clamps to 1..4
		"speakers=0":       0,
		"foo speakers=1":   1,
		"speakers=":        0,
		"speakers=abc":     0,
		"number=2":         0,
		"two speakers":     0,
		"hint: speakers=2": 2,
	}
	for prompt, want := range cases {
		if got := parseSpeakerCount(prompt); got != want {
			t.Errorf("parseSpeakerCount(%q) = %d, want %d", prompt, got, want)
		}
	}
}
