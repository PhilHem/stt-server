package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Recognizer wraps sherpa-onnx offline recognition with mutex serialization.
type Recognizer struct {
	inner *sherpa.OfflineRecognizer
	mu    sync.Mutex
}

// TranscriptionResult holds the output of a transcription.
type TranscriptionResult struct {
	Text          string
	Language      string
	Duration      float32       // audio duration in seconds
	InferenceTime time.Duration // model inference time
}

func newRecognizer(cfg Config) (*Recognizer, error) {
	config := sherpa.OfflineRecognizerConfig{}
	config.FeatConfig.SampleRate = 16000
	config.FeatConfig.FeatureDim = 80
	config.DecodingMethod = "greedy_search"
	config.ModelConfig.NumThreads = cfg.NumThreads
	config.ModelConfig.Provider = cfg.Provider

	// Auto-detect model type from files present in the directory
	if err := detectModel(&config, cfg.ModelDir); err != nil {
		return nil, err
	}

	r := sherpa.NewOfflineRecognizer(&config)
	if r == nil {
		return nil, fmt.Errorf("sherpa-onnx failed to create recognizer (check model files)")
	}

	return &Recognizer{inner: r}, nil
}

func (r *Recognizer) Transcribe(ctx context.Context, samples []float32, sampleRate int) (*TranscriptionResult, error) {
	type result struct {
		res *TranscriptionResult
	}
	ch := make(chan result, 1)

	go func() {
		r.mu.Lock()
		defer r.mu.Unlock()

		stream := sherpa.NewOfflineStream(r.inner)
		defer sherpa.DeleteOfflineStream(stream)

		stream.AcceptWaveform(sampleRate, samples)

		inferStart := time.Now()
		r.inner.Decode(stream)
		inferElapsed := time.Since(inferStart)

		out := stream.GetResult()
		ch <- result{res: &TranscriptionResult{
			Text:          strings.TrimSpace(out.Text),
			Language:      out.Lang,
			Duration:      float32(len(samples)) / float32(sampleRate),
			InferenceTime: inferElapsed,
		}}
	}()

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("inference timed out: %w", ctx.Err())
	case r := <-ch:
		return r.res, nil
	}
}

func (r *Recognizer) Close() {
	sherpa.DeleteOfflineRecognizer(r.inner)
}

// detectModel sets the appropriate model config fields based on what files
// exist in the model directory. Supports transducer (encoder/decoder/joiner)
// and CTC (model.onnx) layouts.
func detectModel(config *sherpa.OfflineRecognizerConfig, dir string) error {
	tokens := findFile(dir, "tokens.txt")
	if tokens == "" {
		return fmt.Errorf("tokens.txt not found in %s", dir)
	}
	config.ModelConfig.Tokens = tokens

	// Transducer layout: encoder + decoder + joiner
	encoder := findOnnx(dir, "encoder")
	decoder := findOnnx(dir, "decoder")
	joiner := findOnnx(dir, "joiner")

	if encoder != "" && decoder != "" && joiner != "" {
		config.ModelConfig.Transducer.Encoder = encoder
		config.ModelConfig.Transducer.Decoder = decoder
		config.ModelConfig.Transducer.Joiner = joiner
		config.ModelConfig.ModelType = "nemo_transducer"
		return nil
	}

	// CTC layout: single model.onnx
	model := findOnnx(dir, "model")
	if model != "" {
		config.ModelConfig.NemoCTC.Model = model
		config.ModelConfig.ModelType = "nemo_ctc"
		return nil
	}

	return fmt.Errorf("no supported model files found in %s (expected encoder/decoder/joiner or model.onnx)", dir)
}

// findOnnx looks for <prefix>.onnx or <prefix>.int8.onnx in dir.
func findOnnx(dir, prefix string) string {
	// Prefer int8 quantized
	for _, suffix := range []string{".int8.onnx", ".onnx"} {
		p := filepath.Join(dir, prefix+suffix)
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func findFile(dir, name string) string {
	p := filepath.Join(dir, name)
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return ""
}
