package main

import (
	"bytes"
	"encoding/binary"
	"log"
	"os"

	"github.com/youpy/go-wav"
)

func readWave(filename string) (samples []float32, sampleRate int) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Failed to open %s: %v", filename, err)
	}
	defer file.Close()

	reader := wav.NewReader(file)
	format, err := reader.Format()
	if err != nil {
		log.Fatalf("Failed to read WAV format: %v", err)
	}

	if format.AudioFormat != 1 {
		log.Fatalf("Only PCM format supported. Got: %d", format.AudioFormat)
	}

	if format.NumChannels != 1 {
		log.Fatalf("Only mono WAV supported. Got: %d channels", format.NumChannels)
	}

	if format.BitsPerSample != 16 {
		log.Fatalf("Only 16-bit WAV supported. Got: %d-bit", format.BitsPerSample)
	}

	reader.Duration() // initializes reader.Size

	buf := make([]byte, reader.Size)
	n, err := reader.Read(buf)
	if n != int(reader.Size) {
		log.Fatalf("Failed to read %d bytes, got %d", reader.Size, n)
	}

	numSamples := len(buf) / 2
	samples = make([]float32, numSamples)
	for i := range numSamples {
		var s16 int16
		binary.Read(bytes.NewReader(buf[i*2:(i+1)*2]), binary.LittleEndian, &s16)
		samples[i] = float32(s16) / 32768.0
	}

	sampleRate = int(format.SampleRate)
	return
}
