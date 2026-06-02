package audio

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// TargetSampleRate is the sample rate used for all decoded audio.
const TargetSampleRate = 16000

// bytesPerSample is the size of one decoded sample (16-bit little-endian PCM).
const bytesPerSample = 2

// decodeHeadroomSamples is added on top of the caller's sample budget so that
// audio which is only slightly over the configured duration still decodes and
// is rejected by the caller's precise duration check (with a friendly 413)
// rather than being silently truncated here. One second of headroom.
const decodeHeadroomSamples = TargetSampleRate

// fallbackMaxSamples caps ffmpeg output when the caller passes no budget
// (maxSamples <= 0), guarding against memory bombs from compressed audio.
// 600s * 16kHz = 9.6M samples.
const fallbackMaxSamples = 600 * TargetSampleRate

// decodedByteBudget converts a sample budget into the ffmpeg output byte cap,
// applying headroom and the no-budget fallback.
func decodedByteBudget(maxSamples int) int {
	if maxSamples <= 0 {
		maxSamples = fallbackMaxSamples
	}
	return (maxSamples + decodeHeadroomSamples) * bytesPerSample
}

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

// IsKnownFormat checks the first bytes of data for recognized audio signatures.
func IsKnownFormat(data []byte) bool {
	// Check structured magic bytes
	for _, m := range audioMagic {
		end := m.offset + len(m.magic)
		if len(data) >= end && bytes.Equal(data[m.offset:end], m.magic) {
			return true
		}
	}

	if len(data) < 4 {
		return false
	}

	// MP3 frame header: sync word (11 bits set) + valid MPEG version + layer
	// Byte 0: 0xFF, Byte 1: 0xE0+ (sync), bits 3-4 = version (not 01),
	// bits 1-2 = layer (not 00)
	if data[0] == 0xFF && (data[1]&0xE0) == 0xE0 {
		version := (data[1] >> 3) & 0x03
		layer := (data[1] >> 1) & 0x03
		bitrate := (data[2] >> 4) & 0x0F
		sampleRate := (data[2] >> 2) & 0x03
		// Reject reserved values that indicate non-MP3 data
		if version != 0x01 && layer != 0x00 && bitrate != 0x0F && sampleRate != 0x03 {
			return true
		}
	}

	// AAC ADTS: 12-bit sync (0xFFF) + valid profile (bits 6-7 of byte 2 != 0x03)
	if data[0] == 0xFF && (data[1]&0xF0) == 0xF0 {
		profile := (data[2] >> 6) & 0x03
		if profile != 0x03 { // 0x03 is reserved
			return true
		}
	}

	return false
}

// Decode converts any audio format to 16kHz mono float32 PCM via ffmpeg.
// maxSamples bounds the decoded output so a small compressed file cannot
// expand into an unbounded PCM buffer; pass 0 to use the default cap. The cap
// carries one second of headroom over maxSamples so the caller's own duration
// check is what rejects slightly-too-long audio.
func Decode(ctx context.Context, data []byte, filename string, maxSamples int) ([]float32, int, error) {
	if !IsKnownFormat(data) {
		return nil, 0, fmt.Errorf("unsupported audio format")
	}

	maxDecodedBytes := decodedByteBudget(maxSamples)

	// Stage the upload in a temp file rather than feeding ffmpeg's stdin. The
	// MP4/M4A container keeps its `moov` index atom at the end of the file, so
	// ffmpeg must seek backwards to decode it — impossible on a stdin pipe,
	// which is why piped m4a/mov produced no output. A regular file is
	// seekable and decodes every supported container uniformly.
	tmp, err := os.CreateTemp("", "stt-decode-*"+inputExt(filename))
	if err != nil {
		slog.Debug("temp file create failed", "error", err)
		return nil, 0, fmt.Errorf("audio decode failed")
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		slog.Debug("temp file write failed", "error", err)
		return nil, 0, fmt.Errorf("audio decode failed")
	}
	if err := tmp.Close(); err != nil {
		slog.Debug("temp file close failed", "error", err)
		return nil, 0, fmt.Errorf("audio decode failed")
	}

	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-nostdin",       // never block reading the controlling terminal
		"-i", tmp.Name(), // seekable input file (lets ffmpeg read the moov atom)
		"-ar", "16000", // resample to 16kHz
		"-ac", "1", // mono
		"-f", "s16le", // raw 16-bit little-endian PCM
		"-acodec", "pcm_s16le",
		"-v", "error", // suppress banner
		"pipe:1", // write to stdout
	)

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
	limitedReader := io.LimitReader(stdoutPipe, int64(maxDecodedBytes)+1)
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

	if len(raw) < 2 {
		return nil, 0, fmt.Errorf("ffmpeg produced no output")
	}

	samples := pcmToFloat32(raw)
	return samples, TargetSampleRate, nil
}

// inputExt returns the lower-cased extension (with leading dot) of the
// original upload filename so the temp file keeps a hint ffmpeg's demuxer can
// use. Returns "" when the name has no plausible extension; ffmpeg falls back
// to content probing in that case.
func inputExt(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	if len(ext) < 2 || len(ext) > 8 {
		return ""
	}
	for _, r := range ext[1:] {
		if !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9') {
			return ""
		}
	}
	return ext
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
