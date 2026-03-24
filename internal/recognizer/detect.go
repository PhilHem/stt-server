package recognizer

import (
	"fmt"
	"os"
	"path/filepath"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

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
