package main

import (
	"flag"
	"log"
	"runtime"
)

type Config struct {
	ModelDir   string
	Port       int
	NumThreads int
	Provider   string // "cpu" or "cuda"
}

func main() {
	var model, cacheDir string
	cfg := Config{}

	flag.StringVar(&model, "model", envOr("STT_MODEL", ""), "Model name or path (env: STT_MODEL)")
	flag.StringVar(&cacheDir, "cache-dir", envOr("STT_CACHE_DIR", ""), "Model cache directory (env: STT_CACHE_DIR)")
	flag.IntVar(&cfg.Port, "port", envInt("STT_PORT", 8000), "HTTP listen port (env: STT_PORT)")
	flag.IntVar(&cfg.NumThreads, "num-threads", envInt("STT_NUM_THREADS", runtime.NumCPU()), "Inference threads (env: STT_NUM_THREADS)")
	flag.StringVar(&cfg.Provider, "provider", envOr("STT_PROVIDER", "cpu"), "ONNX Runtime provider: cpu or cuda (env: STT_PROVIDER)")
	flag.Parse()

	if model == "" {
		log.Fatal("--model or STT_MODEL is required")
	}

	modelDir, err := resolveModel(model, cacheDir)
	if err != nil {
		log.Fatalf("Failed to resolve model: %v", err)
	}
	cfg.ModelDir = modelDir

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
