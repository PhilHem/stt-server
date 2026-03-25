package server

import (
	"strings"
	"testing"
)

func TestBuildWords_Basic(t *testing.T) {
	// SentencePiece tokens: leading space means new word
	tokens := []string{" Hello", " world", " today"}
	starts := []float64{0.0, 0.5, 1.2}
	ends := []float64{0.4, 1.0, 1.8}

	words := buildWords(tokens, starts, ends)

	if len(words) != 3 {
		t.Fatalf("expected 3 words, got %d", len(words))
	}
	if words[0].Word != "Hello" {
		t.Errorf("word[0]: expected 'Hello', got %q", words[0].Word)
	}
	if words[0].Start != 0.0 {
		t.Errorf("word[0].Start: expected 0.0, got %f", words[0].Start)
	}
	if words[0].End != 0.4 {
		t.Errorf("word[0].End: expected 0.4, got %f", words[0].End)
	}
	if words[2].Word != "today" {
		t.Errorf("word[2]: expected 'today', got %q", words[2].Word)
	}
	if words[2].Start != 1.2 {
		t.Errorf("word[2].Start: expected 1.2, got %f", words[2].Start)
	}
	if words[2].End != 1.8 {
		t.Errorf("word[2].End: expected 1.8, got %f", words[2].End)
	}
}

func TestBuildWords_MergesBPETokens(t *testing.T) {
	// "Alles" = " Al" + "les", "hat" = " hat", "ein" = " ein", "Ende." = " En" + "de" + "."
	tokens := []string{" Al", "les", " hat", " ein", " En", "de", "."}
	starts := []float64{0.0, 0.1, 0.3, 0.5, 0.8, 0.9, 1.0}
	ends := []float64{0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 1.05}

	words := buildWords(tokens, starts, ends)

	if len(words) != 4 {
		t.Fatalf("expected 4 words, got %d: %+v", len(words), words)
	}

	// "Alles" merged from " Al" + "les"
	if words[0].Word != "Alles" {
		t.Errorf("word[0]: expected 'Alles', got %q", words[0].Word)
	}
	if words[0].Start != 0.0 {
		t.Errorf("word[0].Start: expected 0.0, got %f", words[0].Start)
	}
	if words[0].End != 0.2 {
		t.Errorf("word[0].End: expected 0.2, got %f", words[0].End)
	}

	// "hat" from " hat"
	if words[1].Word != "hat" {
		t.Errorf("word[1]: expected 'hat', got %q", words[1].Word)
	}

	// "ein" from " ein"
	if words[2].Word != "ein" {
		t.Errorf("word[2]: expected 'ein', got %q", words[2].Word)
	}

	// "Ende." merged from " En" + "de" + "."
	if words[3].Word != "Ende." {
		t.Errorf("word[3]: expected 'Ende.', got %q", words[3].Word)
	}
	if words[3].Start != 0.8 {
		t.Errorf("word[3].Start: expected 0.8, got %f", words[3].Start)
	}
	if words[3].End != 1.05 {
		t.Errorf("word[3].End: expected 1.05, got %f", words[3].End)
	}
}

func TestBuildWords_Empty(t *testing.T) {
	words := buildWords(nil, nil, nil)
	if len(words) != 0 {
		t.Errorf("expected 0 words for nil input, got %d", len(words))
	}

	words = buildWords([]string{}, []float64{}, []float64{})
	if len(words) != 0 {
		t.Errorf("expected 0 words for empty input, got %d", len(words))
	}
}

func TestBuildWords_SkipsWhitespace(t *testing.T) {
	// Whitespace-only tokens should be skipped entirely
	tokens := []string{" Hello", " ", " world", "\t", "!"}
	starts := []float64{0.0, 0.3, 0.5, 0.9, 1.0}
	ends := []float64{0.2, 0.4, 0.8, 0.95, 1.1}

	words := buildWords(tokens, starts, ends)

	// "!" has no leading space so it attaches to "world" -> "world!"
	if len(words) != 2 {
		t.Fatalf("expected 2 words (whitespace filtered, ! merged), got %d: %+v", len(words), words)
	}
	if words[0].Word != "Hello" {
		t.Errorf("word[0]: expected 'Hello', got %q", words[0].Word)
	}
	if words[1].Word != "world!" {
		t.Errorf("word[1]: expected 'world!', got %q", words[1].Word)
	}
}

func TestBuildSegments_SentenceSplit(t *testing.T) {
	words := []Word{
		{Word: "Hello", Start: 0.0, End: 0.5},
		{Word: "world.", Start: 0.5, End: 1.0},
		{Word: "How", Start: 1.2, End: 1.5},
		{Word: "are", Start: 1.5, End: 1.8},
		{Word: "you?", Start: 1.8, End: 2.2},
	}

	segments := buildSegments(words)

	if len(segments) != 2 {
		t.Fatalf("expected 2 segments, got %d", len(segments))
	}
	if segments[0].Text != "Hello world." {
		t.Errorf("segment[0].Text: expected 'Hello world.', got %q", segments[0].Text)
	}
	if segments[0].Start != 0.0 {
		t.Errorf("segment[0].Start: expected 0.0, got %f", segments[0].Start)
	}
	if segments[0].End != 1.0 {
		t.Errorf("segment[0].End: expected 1.0, got %f", segments[0].End)
	}
	if segments[1].Text != "How are you?" {
		t.Errorf("segment[1].Text: expected 'How are you?', got %q", segments[1].Text)
	}
	if segments[1].ID != 1 {
		t.Errorf("segment[1].ID: expected 1, got %d", segments[1].ID)
	}
}

func TestBuildSegments_LongSegmentSplit(t *testing.T) {
	// Create 20 words with no punctuation — should split at ~15
	var words []Word
	for i := range 20 {
		words = append(words, Word{
			Word:  "word",
			Start: float64(i) * 0.5,
			End:   float64(i)*0.5 + 0.4,
		})
	}

	segments := buildSegments(words)

	if len(segments) < 2 {
		t.Fatalf("expected at least 2 segments for 20 words without punctuation, got %d", len(segments))
	}
	// First segment should have exactly 15 words
	firstWords := strings.Fields(segments[0].Text)
	if len(firstWords) != 15 {
		t.Errorf("first segment: expected 15 words, got %d", len(firstWords))
	}
	// Second segment should have the remaining 5 words
	secondWords := strings.Fields(segments[1].Text)
	if len(secondWords) != 5 {
		t.Errorf("second segment: expected 5 words, got %d", len(secondWords))
	}
}

func TestBuildSegments_Empty(t *testing.T) {
	segments := buildSegments(nil)
	if segments != nil {
		t.Errorf("expected nil for nil input, got %v", segments)
	}

	segments = buildSegments([]Word{})
	if segments != nil {
		t.Errorf("expected nil for empty input, got %v", segments)
	}
}

func TestFormatSRTTime(t *testing.T) {
	tests := []struct {
		seconds  float64
		expected string
	}{
		{0.0, "00:00:00,000"},
		{1.5, "00:00:01,500"},
		{61.123, "00:01:01,123"},
		{3661.999, "01:01:01,999"},
		{7200.0, "02:00:00,000"},
		{0.001, "00:00:00,001"},
	}

	for _, tt := range tests {
		got := formatSRTTime(tt.seconds)
		if got != tt.expected {
			t.Errorf("formatSRTTime(%f): expected %q, got %q", tt.seconds, tt.expected, got)
		}
	}
}

func TestFormatVTTTime(t *testing.T) {
	tests := []struct {
		seconds  float64
		expected string
	}{
		{0.0, "00:00:00.000"},
		{1.5, "00:00:01.500"},
		{61.123, "00:01:01.123"},
		{3661.999, "01:01:01.999"},
		{7200.0, "02:00:00.000"},
		{0.001, "00:00:00.001"},
	}

	for _, tt := range tests {
		got := formatVTTTime(tt.seconds)
		if got != tt.expected {
			t.Errorf("formatVTTTime(%f): expected %q, got %q", tt.seconds, tt.expected, got)
		}
	}
}

func TestFormatSRT_Structure(t *testing.T) {
	segments := []Segment{
		{ID: 0, Start: 0.0, End: 2.5, Text: "Hello world."},
		{ID: 1, Start: 3.0, End: 5.5, Text: "How are you?"},
	}

	srt := formatSRT(segments)

	// Should contain numbered blocks
	if !strings.Contains(srt, "1\n") {
		t.Error("SRT should contain sequence number 1")
	}
	if !strings.Contains(srt, "2\n") {
		t.Error("SRT should contain sequence number 2")
	}
	// Should contain SRT-style timestamps with comma separator
	if !strings.Contains(srt, "00:00:00,000 --> 00:00:02,500") {
		t.Error("SRT should contain timestamp line with comma separator")
	}
	// Should contain text
	if !strings.Contains(srt, "Hello world.") {
		t.Error("SRT should contain segment text")
	}
	// Each block ends with a blank line
	blocks := strings.Split(strings.TrimSpace(srt), "\n\n")
	if len(blocks) != 2 {
		t.Errorf("expected 2 SRT blocks, got %d", len(blocks))
	}
	// Should NOT start with "WEBVTT"
	if strings.HasPrefix(srt, "WEBVTT") {
		t.Error("SRT output should not start with WEBVTT header")
	}
}

func TestFormatVTT_Structure(t *testing.T) {
	segments := []Segment{
		{ID: 0, Start: 0.0, End: 2.5, Text: "Hello world."},
		{ID: 1, Start: 3.0, End: 5.5, Text: "How are you?"},
	}

	vtt := formatVTT(segments)

	// Must start with "WEBVTT\n\n"
	if !strings.HasPrefix(vtt, "WEBVTT\n\n") {
		t.Errorf("VTT output must start with 'WEBVTT\\n\\n', got prefix: %q", vtt[:min(len(vtt), 20)])
	}
	// Should contain VTT-style timestamps with dot separator
	if !strings.Contains(vtt, "00:00:00.000 --> 00:00:02.500") {
		t.Error("VTT should contain timestamp line with dot separator")
	}
	// Should contain text
	if !strings.Contains(vtt, "Hello world.") {
		t.Error("VTT should contain segment text")
	}
	if !strings.Contains(vtt, "How are you?") {
		t.Error("VTT should contain second segment text")
	}
	// Should NOT contain comma-style timestamps (that's SRT)
	if strings.Contains(vtt, ",000") || strings.Contains(vtt, ",500") {
		t.Error("VTT timestamps should use dots, not commas")
	}
}
