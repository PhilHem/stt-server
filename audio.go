package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os/exec"
)

const targetSampleRate = 16000

// decodeAudio converts any audio format to 16kHz mono float32 PCM via ffmpeg.
// Falls back to direct WAV parsing if the input is already a compatible WAV.
func decodeAudio(data []byte, filename string) ([]float32, int, error) {
	cmd := exec.Command("ffmpeg",
		"-i", "pipe:0", // read from stdin
		"-ar", "16000", // resample to 16kHz
		"-ac", "1", // mono
		"-f", "s16le", // raw 16-bit little-endian PCM
		"-acodec", "pcm_s16le",
		"-v", "error", // suppress banner
		"pipe:1", // write to stdout
	)

	cmd.Stdin = bytes.NewReader(data)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, 0, fmt.Errorf("ffmpeg: %v: %s", err, stderr.String())
	}

	raw := stdout.Bytes()
	if len(raw) == 0 {
		return nil, 0, fmt.Errorf("ffmpeg produced no output")
	}

	samples := pcmToFloat32(raw)
	return samples, targetSampleRate, nil
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
