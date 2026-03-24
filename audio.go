package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os/exec"
)

const targetSampleRate = 16000

// maxDecodedBytes caps ffmpeg output to prevent memory bombs from compressed audio.
// 600s * 16kHz * 2 bytes/sample = 19.2 MB; use 25 MB with headroom.
const maxDecodedBytes = 25 << 20

// audioMagic maps recognized audio format signatures to their names.
// Checked against the first bytes of the input to reject non-audio data.
var audioMagic = []struct {
	offset int
	magic  []byte
	name   string
}{
	{0, []byte("RIFF"), "wav"},
	{0, []byte("ID3"), "mp3"},
	{0, []byte("fLaC"), "flac"},
	{0, []byte("OggS"), "ogg"},
	{0, []byte{0x1A, 0x45, 0xDF, 0xA3}, "webm"},
	{4, []byte("ftyp"), "m4a/mp4"},
}

// isKnownAudioFormat checks the first bytes of data for recognized audio signatures.
func isKnownAudioFormat(data []byte) bool {
	// Check structured magic bytes
	for _, m := range audioMagic {
		end := m.offset + len(m.magic)
		if len(data) >= end && bytes.Equal(data[m.offset:end], m.magic) {
			return true
		}
	}

	if len(data) < 2 {
		return false
	}

	// MP3 sync word: first 11 bits set (0xFFE0 or higher)
	if data[0] == 0xFF && (data[1]&0xE0) == 0xE0 {
		return true
	}

	// AAC ADTS sync word: 0xFFF0 or higher (first 12 bits set)
	if data[0] == 0xFF && (data[1]&0xF0) == 0xF0 {
		return true
	}

	return false
}

// decodeAudio converts any audio format to 16kHz mono float32 PCM via ffmpeg.
func decodeAudio(ctx context.Context, data []byte, filename string) ([]float32, int, error) {
	if !isKnownAudioFormat(data) {
		return nil, 0, fmt.Errorf("unsupported audio format")
	}

	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-i", "pipe:0", // read from stdin
		"-ar", "16000", // resample to 16kHz
		"-ac", "1", // mono
		"-f", "s16le", // raw 16-bit little-endian PCM
		"-acodec", "pcm_s16le",
		"-v", "error", // suppress banner
		"pipe:1", // write to stdout
	)

	cmd.Stdin = bytes.NewReader(data)

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, 0, fmt.Errorf("audio decode failed")
	}

	stderr := &limitedBuffer{max: 64 * 1024} // 64 KB cap for error output
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		slog.Debug("ffmpeg start failed", "error", err)
		return nil, 0, fmt.Errorf("audio decode failed")
	}

	// Read up to maxDecodedBytes + 1 to detect overflow
	limitedReader := io.LimitReader(stdoutPipe, maxDecodedBytes+1)
	raw, err := io.ReadAll(limitedReader)
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		slog.Debug("ffmpeg stdout read failed", "error", err)
		return nil, 0, fmt.Errorf("audio decode failed")
	}

	if len(raw) > maxDecodedBytes {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, 0, fmt.Errorf("decoded audio exceeds maximum size (audio too long or corrupt)")
	}

	if err := cmd.Wait(); err != nil {
		slog.Debug("ffmpeg failed", "error", err, "stderr", stderr.buf.String())
		return nil, 0, fmt.Errorf("audio decode failed")
	}

	if len(raw) == 0 {
		return nil, 0, fmt.Errorf("ffmpeg produced no output")
	}

	samples := pcmToFloat32(raw)
	return samples, targetSampleRate, nil
}

// limitedBuffer is a writer that silently discards data after max bytes.
// Used to cap ffmpeg stderr so malformed audio can't cause unbounded memory growth.
type limitedBuffer struct {
	buf bytes.Buffer
	max int
}

func (w *limitedBuffer) Write(p []byte) (int, error) {
	remaining := w.max - w.buf.Len()
	if remaining <= 0 {
		return len(p), nil // discard, report success to avoid EPIPE
	}
	if len(p) > remaining {
		p = p[:remaining]
	}
	w.buf.Write(p)
	return len(p), nil
}

// pcmToFloat32 converts raw 16-bit little-endian PCM bytes to float32 samples.
func pcmToFloat32(raw []byte) []float32 {
	n := len(raw) / 2
	samples := make([]float32, n)
	for i := range n {
		s := int16(binary.LittleEndian.Uint16(raw[i*2 : (i+1)*2]))
		samples[i] = float32(s) / float32(math.MaxInt16)
	}
	return samples
}
