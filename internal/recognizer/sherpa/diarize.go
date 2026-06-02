package sherpa

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"
)

// diarMergeGapSec merges consecutive turns of the same speaker separated by a
// gap shorter than this into one turn, so each recognition segment is a whole
// utterance (better ASR context) rather than many sub-second fragments.
const diarMergeGapSec = 1.0

// httpDiarizer calls the external GPU diarization service: it posts 16 kHz mono
// WAV and gets back speaker turns.
type httpDiarizer struct {
	url    string
	client *http.Client
}

func newHTTPDiarizer(url string) *httpDiarizer {
	return &httpDiarizer{url: url, client: &http.Client{Timeout: 30 * time.Minute}}
}

type diarSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker int     `json:"speaker"`
}

type diarResponse struct {
	Segments []diarSegment `json:"segments"`
}

// segments diarizes the audio and returns recognition segments tagged with
// their speaker: same-speaker turns are merged, then each turn is split into
// windows the recognizer can handle.
func (d *httpDiarizer) segments(ctx context.Context, samples []float32, sampleRate int) ([]segment, error) {
	wav := encodeWAV(samples, sampleRate)

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	part, err := mw.CreateFormFile("file", "audio.wav")
	if err != nil {
		return nil, fmt.Errorf("diarize: build request: %w", err)
	}
	if _, err := part.Write(wav); err != nil {
		return nil, fmt.Errorf("diarize: write audio: %w", err)
	}
	if err := mw.Close(); err != nil {
		return nil, fmt.Errorf("diarize: close request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, d.url, &body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("diarize: request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("diarize: upstream %d: %s", resp.StatusCode, string(raw))
	}

	var dr diarResponse
	if err := json.NewDecoder(resp.Body).Decode(&dr); err != nil {
		return nil, fmt.Errorf("diarize: decode response: %w", err)
	}

	var segs []segment
	for _, t := range mergeTurns(dr.Segments, diarMergeGapSec) {
		start := int(t.Start * float64(sampleRate))
		end := int(t.End * float64(sampleRate))
		if start < 0 {
			start = 0
		}
		if end > len(samples) {
			end = len(samples)
		}
		if end <= start {
			continue
		}
		turnSamples := samples[start:end]
		for _, w := range splitWindows(turnSamples, sampleRate, chunkTargetSeconds, chunkSearchSeconds) {
			segs = append(segs, segment{
				start:   start + w.start,
				samples: turnSamples[w.start:w.end],
				speaker: t.Speaker,
			})
		}
	}
	return segs, nil
}

type diarTurn struct {
	Start, End float64
	Speaker    int
}

// mergeTurns collapses consecutive same-speaker segments (gap <= gapSec) into
// one turn. Input is assumed sorted by start time (the service sorts it).
func mergeTurns(segs []diarSegment, gapSec float64) []diarTurn {
	var turns []diarTurn
	for _, s := range segs {
		if n := len(turns); n > 0 && turns[n-1].Speaker == s.Speaker && s.Start-turns[n-1].End <= gapSec {
			if s.End > turns[n-1].End {
				turns[n-1].End = s.End
			}
			continue
		}
		turns = append(turns, diarTurn{Start: s.Start, End: s.End, Speaker: s.Speaker})
	}
	return turns
}

// encodeWAV renders float32 samples as a 16-bit mono PCM WAV.
func encodeWAV(samples []float32, sampleRate int) []byte {
	dataLen := len(samples) * 2
	buf := bytes.NewBuffer(make([]byte, 0, 44+dataLen))
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(36+dataLen))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))            // PCM
	_ = binary.Write(buf, binary.LittleEndian, uint16(1))            // mono
	_ = binary.Write(buf, binary.LittleEndian, uint32(sampleRate))   // sample rate
	_ = binary.Write(buf, binary.LittleEndian, uint32(sampleRate*2)) // byte rate
	_ = binary.Write(buf, binary.LittleEndian, uint16(2))            // block align
	_ = binary.Write(buf, binary.LittleEndian, uint16(16))           // bits/sample
	buf.WriteString("data")
	_ = binary.Write(buf, binary.LittleEndian, uint32(dataLen))
	for _, s := range samples {
		v := int32(s * 32767)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		_ = binary.Write(buf, binary.LittleEndian, int16(v))
	}
	return buf.Bytes()
}
