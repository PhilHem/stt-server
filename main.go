package main

import (
	"flag"
	"log"
	"os"
	"runtime"
)

type Config struct {
	ModelDir   string
	Port       int
	NumThreads int
	Provider   string // "cpu" or "cuda"
}

func main() {
	cfg := Config{}

	flag.StringVar(&cfg.ModelDir, "model", envOr("STT_MODEL", ""), "Path to sherpa-onnx model directory (env: STT_MODEL)")
	flag.IntVar(&cfg.Port, "port", envInt("STT_PORT", 8000), "HTTP listen port (env: STT_PORT)")
	flag.IntVar(&cfg.NumThreads, "num-threads", envInt("STT_NUM_THREADS", runtime.NumCPU()), "Inference threads (env: STT_NUM_THREADS)")
	flag.StringVar(&cfg.Provider, "provider", envOr("STT_PROVIDER", "cpu"), "ONNX Runtime provider: cpu or cuda (env: STT_PROVIDER)")
	flag.Parse()

	if cfg.ModelDir == "" {
		log.Fatal("--model or STT_MODEL is required")
	}

	// Verify model directory exists
	if _, err := os.Stat(cfg.ModelDir); os.IsNotExist(err) {
		log.Fatalf("Model directory not found: %s", cfg.ModelDir)
	}

	recognizer, err := newRecognizer(cfg)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer recognizer.Close()

	log.Printf("Model loaded from %s (%d threads, provider=%s)", cfg.ModelDir, cfg.NumThreads, cfg.Provider)

	srv := newServer(recognizer, cfg.Port)
	log.Printf("Listening on :%d", cfg.Port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
