package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

const modelDir = "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <audio.wav>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nTranscribe a 16kHz mono WAV file using Parakeet V3.\n")
		fmt.Fprintf(os.Stderr, "Download the model first: ./download-model.sh\n")
		os.Exit(1)
	}

	wavPath := os.Args[1]

	config := sherpa.OfflineRecognizerConfig{}
	config.ModelConfig.Transducer.Encoder = modelDir + "/encoder.int8.onnx"
	config.ModelConfig.Transducer.Decoder = modelDir + "/decoder.int8.onnx"
	config.ModelConfig.Transducer.Joiner = modelDir + "/joiner.int8.onnx"
	config.ModelConfig.Tokens = modelDir + "/tokens.txt"
	config.ModelConfig.NumThreads = 4
	config.ModelConfig.Provider = "cpu"
	config.ModelConfig.ModelType = "nemo_transducer"
	config.FeatConfig.SampleRate = 16000
	config.FeatConfig.FeatureDim = 80
	config.DecodingMethod = "greedy_search"

	log.Println("Loading model...")
	recognizer := sherpa.NewOfflineRecognizer(&config)
	if recognizer == nil {
		log.Fatal("Failed to create recognizer. Are model files present?")
	}
	defer sherpa.DeleteOfflineRecognizer(recognizer)
	log.Println("Model loaded.")

	log.Printf("Transcribing %s ...", wavPath)
	samples, sampleRate := readWave(wavPath)

	stream := sherpa.NewOfflineStream(recognizer)
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)
	recognizer.Decode(stream)

	result := stream.GetResult()

	fmt.Println()
	fmt.Println(strings.TrimSpace(result.Text))
	if result.Lang != "" {
		log.Printf("Language: %s", result.Lang)
	}
	log.Printf("Duration: %.1fs", float32(len(samples))/float32(sampleRate))
}
