package server

import (
	"fmt"
	"math"
	"strings"
)

// Word represents a single word with timing information.
type Word struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// Segment represents a timed text segment (sentence or chunk).
type Segment struct {
	ID    int     `json:"id"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// tokenTimesToStartEnd converts a flat slice of token start-times (from
// sherpa-onnx) into parallel start/end slices. Each token's end time is the
// next token's start time; the last token gets a default 0.1 s duration.
func tokenTimesToStartEnd(timestamps []float32) (starts, ends []float64) {
	n := len(timestamps)
	starts = make([]float64, n)
	ends = make([]float64, n)
	for i, ts := range timestamps {
		starts[i] = float64(ts)
		if i+1 < n {
			ends[i] = float64(timestamps[i+1])
		} else {
			ends[i] = float64(ts) + 0.1
		}
	}
	return starts, ends
}

// buildWords merges BPE sub-word tokens into whole words using the
// SentencePiece convention: tokens starting with a space begin a new word,
// tokens without a leading space continue the previous word, and punctuation
// tokens attach to the previous word. Timing spans from the first token's
// start to the last token's end for each merged word.
func buildWords(tokens []string, starts, ends []float64) []Word {
	var words []Word
	var currentWord string
	var wordStart, wordEnd float64

	for i, token := range tokens {
		if i >= len(starts) || i >= len(ends) {
			break
		}

		clean := strings.TrimSpace(token)
		if clean == "" {
			continue
		}

		// New word starts when token has a leading space (SentencePiece convention).
		isNewWord := len(token) > 0 && token[0] == ' '

		if isNewWord && currentWord != "" {
			// Flush previous word.
			words = append(words, Word{
				Word:  currentWord,
				Start: wordStart,
				End:   wordEnd,
			})
			currentWord = ""
		}

		if currentWord == "" {
			wordStart = starts[i]
		}
		currentWord += clean
		wordEnd = ends[i]
	}

	// Flush last word.
	if currentWord != "" {
		words = append(words, Word{
			Word:  currentWord,
			Start: wordStart,
			End:   wordEnd,
		})
	}

	return words
}

// buildSegments groups words into segments, splitting on sentence-ending
// punctuation (.!?) or every ~15 words if no punctuation is found.
func buildSegments(words []Word) []Segment {
	if len(words) == 0 {
		return nil
	}

	var segments []Segment
	var current []Word
	id := 0

	flush := func() {
		if len(current) == 0 {
			return
		}
		text := wordsToText(current)
		segments = append(segments, Segment{
			ID:    id,
			Start: current[0].Start,
			End:   current[len(current)-1].End,
			Text:  text,
		})
		id++
		current = nil
	}

	for _, w := range words {
		current = append(current, w)
		// Split on sentence-ending punctuation
		if endsWithSentencePunctuation(w.Word) {
			flush()
			continue
		}
		// Split every ~15 words if no punctuation
		if len(current) >= 15 {
			flush()
		}
	}
	flush()

	return segments
}

// endsWithSentencePunctuation checks if a word ends with . ! or ?
func endsWithSentencePunctuation(word string) bool {
	if word == "" {
		return false
	}
	last := word[len(word)-1]
	return last == '.' || last == '!' || last == '?'
}

// wordsToText joins word texts with spaces.
func wordsToText(words []Word) string {
	parts := make([]string, len(words))
	for i, w := range words {
		parts[i] = w.Word
	}
	return strings.Join(parts, " ")
}

// formatSRT produces SubRip subtitle format from segments.
func formatSRT(segments []Segment) string {
	var b strings.Builder
	for _, seg := range segments {
		fmt.Fprintf(&b, "%d\n", seg.ID+1)
		fmt.Fprintf(&b, "%s --> %s\n", formatSRTTime(seg.Start), formatSRTTime(seg.End))
		fmt.Fprintf(&b, "%s\n\n", seg.Text)
	}
	return b.String()
}

// formatVTT produces WebVTT subtitle format from segments.
func formatVTT(segments []Segment) string {
	var b strings.Builder
	b.WriteString("WEBVTT\n\n")
	for _, seg := range segments {
		fmt.Fprintf(&b, "%s --> %s\n", formatVTTTime(seg.Start), formatVTTTime(seg.End))
		fmt.Fprintf(&b, "%s\n\n", seg.Text)
	}
	return b.String()
}

// formatSRTTime formats seconds as HH:MM:SS,mmm (SRT timestamp format).
func formatSRTTime(seconds float64) string {
	totalMS := int(math.Round(seconds * 1000))
	h := totalMS / 3600000
	m := (totalMS % 3600000) / 60000
	s := (totalMS % 60000) / 1000
	ms := totalMS % 1000
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms)
}

// formatVTTTime formats seconds as HH:MM:SS.mmm (WebVTT timestamp format).
func formatVTTTime(seconds float64) string {
	totalMS := int(math.Round(seconds * 1000))
	h := totalMS / 3600000
	m := (totalMS % 3600000) / 60000
	s := (totalMS % 60000) / 1000
	ms := totalMS % 1000
	return fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
}
