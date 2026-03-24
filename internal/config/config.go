package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds all runtime configuration for the STT server.
type Config struct {
	ModelDir         string
	Port             int
	NumThreads       int
	Provider         string // "cpu" or "cuda"
	MaxConcurrent    int
	MaxQueue         int
	MaxFileSizeMB    int
	MaxAudioDuration time.Duration
	RequestTimeout   time.Duration
}

// EnvOr returns the environment variable value for key, or fallback if unset/empty.
func EnvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// EnvInt returns the environment variable value for key parsed as int, or fallback if unset/empty/invalid.
func EnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}
