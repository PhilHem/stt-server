package sherpa

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// detectModel auto-detects the model type from files present in dir and
// configures the sherpa-onnx recognizer accordingly. Returns the model type string.
//
// Detection order (first match wins):
//  1. NeMo transducer: encoder + decoder + joiner (Parakeet, Zipformer)
//  2. Whisper: encoder + decoder (no joiner), supports language hint
//  3. SenseVoice: model.onnx with sense_voice in path, supports language hint
//  4. Paraformer: model.onnx with paraformer in path
//  5. NeMo CTC: model.onnx (generic fallback)
func detectModel(config *sherpa.OfflineRecognizerConfig, dir, language string) (string, error) {
	tokens := findFile(dir, "tokens.txt")
	if tokens == "" {
		return "", fmt.Errorf("tokens.txt not found in %s", dir)
	}
	config.ModelConfig.Tokens = tokens

	// 1. NeMo transducer: encoder + decoder + joiner (Parakeet, Zipformer, etc.)
	encoder := findOnnx(dir, "encoder")
	decoder := findOnnx(dir, "decoder")
	joiner := findOnnx(dir, "joiner")

	if encoder != "" && decoder != "" && joiner != "" {
		config.ModelConfig.Transducer.Encoder = encoder
		config.ModelConfig.Transducer.Decoder = decoder
		config.ModelConfig.Transducer.Joiner = joiner
		config.ModelConfig.ModelType = "nemo_transducer"
		slog.Info("detected model type", "type", "nemo_transducer", "dir", dir)
		return "nemo_transducer", nil
	}

	// 2. Whisper: encoder + decoder (no joiner)
	if encoder != "" && decoder != "" {
		config.ModelConfig.Whisper.Encoder = encoder
		config.ModelConfig.Whisper.Decoder = decoder
		config.ModelConfig.Whisper.Language = language
		config.ModelConfig.Whisper.Task = "transcribe"
		config.ModelConfig.Whisper.TailPaddings = -1
		config.ModelConfig.ModelType = "whisper"
		slog.Info("detected model type", "type", "whisper", "language", language, "dir", dir)
		return "whisper", nil
	}

	// 3. SenseVoice: single model.onnx, supports language
	model := findOnnx(dir, "model")
	if model != "" {
		// Check for SenseVoice-specific files
		if findFile(dir, "tokens.txt") != "" && hasSenseVoiceMarker(dir) {
			config.ModelConfig.SenseVoice.Model = model
			config.ModelConfig.SenseVoice.Language = language
			config.ModelConfig.SenseVoice.UseInverseTextNormalization = 1
			config.ModelConfig.ModelType = "sense_voice"
			slog.Info("detected model type", "type", "sense_voice", "language", language, "dir", dir)
			return "sense_voice", nil
		}

		// 4. Paraformer
		if hasParaformerMarker(dir) {
			config.ModelConfig.Paraformer.Model = model
			config.ModelConfig.ModelType = "paraformer"
			slog.Info("detected model type", "type", "paraformer", "dir", dir)
			return "paraformer", nil
		}

		// 5. NeMo CTC (generic fallback for single model.onnx)
		config.ModelConfig.NemoCTC.Model = model
		config.ModelConfig.ModelType = "nemo_ctc"
		slog.Info("detected model type", "type", "nemo_ctc", "dir", dir)
		return "nemo_ctc", nil
	}

	return "", fmt.Errorf("no supported model files found in %s", dir)
}

// hasSenseVoiceMarker checks for files that indicate a SenseVoice model.
func hasSenseVoiceMarker(dir string) bool {
	// SenseVoice models typically have "sense_voice" or "sensevoice" in the directory name
	base := filepath.Base(dir)
	return contains(base, "sense_voice") || contains(base, "sensevoice") || contains(base, "SenseVoice")
}

// hasParaformerMarker checks for files that indicate a Paraformer model.
func hasParaformerMarker(dir string) bool {
	base := filepath.Base(dir)
	return contains(base, "paraformer") || contains(base, "Paraformer")
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// findOnnx looks for <prefix>.onnx or <prefix>.int8.onnx in dir.
func findOnnx(dir, prefix string) string {
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
