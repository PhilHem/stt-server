package main

import (
	"fmt"
	"log/slog"
	"os"
)

// setupLogging configures the global slog logger.
// format is "text" (human-readable, default) or "json" (structured).
// level is "debug", "info", "warn", or "error".
func setupLogging(format, level string) error {
	lvl, err := parseLevel(level)
	if err != nil {
		return err
	}

	opts := &slog.HandlerOptions{Level: lvl}

	var handler slog.Handler
	switch format {
	case "text", "":
		handler = slog.NewTextHandler(os.Stderr, opts)
	case "json":
		handler = slog.NewJSONHandler(os.Stderr, opts)
	default:
		return fmt.Errorf("unknown log format %q (expected text or json)", format)
	}

	slog.SetDefault(slog.New(handler))
	return nil
}

func parseLevel(s string) (slog.Level, error) {
	switch s {
	case "debug":
		return slog.LevelDebug, nil
	case "info", "":
		return slog.LevelInfo, nil
	case "warn":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf("unknown log level %q (expected debug, info, warn, error)", s)
	}
}
