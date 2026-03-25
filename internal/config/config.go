package config

import (
	"fmt"
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
	PoolSize         int    // number of recognizer instances (default 1, set higher for GPU)
	MaxConcurrent    int
	MaxQueue         int
	MaxFileSizeMB    int
	MaxAudioDuration time.Duration
	RequestTimeout   time.Duration
}

// Validate checks that all configuration values are within acceptable ranges.
func (c Config) Validate() error {
	if c.MaxConcurrent <= 0 {
		return fmt.Errorf("max-concurrent must be > 0, got %d", c.MaxConcurrent)
	}
	if c.MaxQueue < 0 {
		return fmt.Errorf("max-queue must be >= 0, got %d", c.MaxQueue)
	}
	if c.MaxFileSizeMB <= 0 {
		return fmt.Errorf("max-file-size must be > 0, got %d", c.MaxFileSizeMB)
	}
	if c.MaxAudioDuration <= 0 {
		return fmt.Errorf("max-audio-duration must be > 0, got %v", c.MaxAudioDuration)
	}
	if c.RequestTimeout <= 0 {
		return fmt.Errorf("request-timeout must be > 0, got %v", c.RequestTimeout)
	}
	if c.Port <= 0 || c.Port > 65535 {
		return fmt.Errorf("port must be 1-65535, got %d", c.Port)
	}
	return nil
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
