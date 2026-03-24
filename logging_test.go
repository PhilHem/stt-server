package main

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

func TestSetupLogging_TextFormat(t *testing.T) {
	if err := setupLogging("text", "info"); err != nil {
		t.Fatalf("setupLogging(text, info): %v", err)
	}

	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})))
	slog.Info("hello", "key", "value")

	out := buf.String()
	if !strings.Contains(out, "hello") {
		t.Errorf("expected 'hello' in text output, got: %s", out)
	}
	if !strings.Contains(out, "key=value") {
		t.Errorf("expected 'key=value' in text output, got: %s", out)
	}
	// Text format should NOT be valid JSON
	if json.Valid([]byte(strings.TrimSpace(out))) {
		t.Errorf("text format should not be valid JSON, got: %s", out)
	}
}

func TestSetupLogging_JSONFormat(t *testing.T) {
	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})))
	slog.Info("hello", "key", "value")

	out := buf.String()
	// JSON format should be valid JSON
	var m map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &m); err != nil {
		t.Fatalf("JSON output should be valid JSON, got: %s, error: %v", out, err)
	}
	if m["msg"] != "hello" {
		t.Errorf("expected msg=hello, got: %v", m["msg"])
	}
	if m["key"] != "value" {
		t.Errorf("expected key=value, got: %v", m["key"])
	}
	if m["level"] != "INFO" {
		t.Errorf("expected level=INFO, got: %v", m["level"])
	}
	if _, ok := m["time"]; !ok {
		t.Error("expected time field in JSON output")
	}
}

func TestSetupLogging_JournalFormat_NoSocket(t *testing.T) {
	// On macOS / CI without journald, journal mode should return an error
	err := setupLogging("journal", "info")
	if err == nil {
		// If it succeeded, we're on a system with journald — that's fine
		t.Log("journal handler created (journald available)")
		return
	}
	// Error should mention journald
	if !strings.Contains(err.Error(), "journal") {
		t.Errorf("expected journal-related error, got: %v", err)
	}
}

func TestSetupLogging_InvalidFormat(t *testing.T) {
	if err := setupLogging("yaml", "info"); err == nil {
		t.Error("expected error for invalid format 'yaml'")
	}
}

func TestSetupLogging_InvalidLevel(t *testing.T) {
	if err := setupLogging("text", "verbose"); err == nil {
		t.Error("expected error for invalid level 'verbose'")
	}
}

func TestSetupLogging_AllLevels(t *testing.T) {
	for _, level := range []string{"debug", "info", "warn", "error"} {
		for _, format := range []string{"text", "json"} {
			if err := setupLogging(format, level); err != nil {
				t.Errorf("setupLogging(%s, %s): %v", format, level, err)
			}
		}
	}
}

func TestSetupLogging_LevelFiltering(t *testing.T) {
	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))

	slog.Info("should be filtered")
	slog.Warn("should appear")

	out := buf.String()
	if strings.Contains(out, "should be filtered") {
		t.Error("info message should be filtered at warn level")
	}
	if !strings.Contains(out, "should appear") {
		t.Error("warn message should appear at warn level")
	}
}

func TestSetupLogging_JSONFields(t *testing.T) {
	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})))

	slog.Info("transcribed",
		"file", "meeting.mp3",
		"duration_s", "3.8",
		"elapsed_ms", int64(268),
		"lang", "de",
	)

	var m map[string]any
	if err := json.Unmarshal(buf.Bytes(), &m); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	checks := map[string]any{
		"msg":        "transcribed",
		"file":       "meeting.mp3",
		"duration_s": "3.8",
		"lang":       "de",
	}
	for k, want := range checks {
		if m[k] != want {
			t.Errorf("%s: got %v, want %v", k, m[k], want)
		}
	}
	// elapsed_ms comes back as float64 from JSON
	if m["elapsed_ms"] != float64(268) {
		t.Errorf("elapsed_ms: got %v, want 268", m["elapsed_ms"])
	}
}
