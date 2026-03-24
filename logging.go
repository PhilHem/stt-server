package main

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	slogjournal "github.com/systemd/slog-journal"
)

// setupLogging configures the global slog logger.
//
//	text    — human-readable key=value to stderr (default)
//	json    — structured JSON to stderr (for Loki/Promtail/ELK)
//	journal — native systemd journal protocol with first-class fields
func setupLogging(format, level string) error {
	lvl, err := parseLevel(level)
	if err != nil {
		return err
	}

	var handler slog.Handler
	switch format {
	case "text", "":
		handler = slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl})
	case "json":
		handler = slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: lvl})
	case "journal":
		handler, err = newJournalHandler(lvl)
		if err != nil {
			return fmt.Errorf("journal handler: %w (is systemd-journald running?)", err)
		}
	default:
		return fmt.Errorf("unknown log format %q (expected text, json, or journal)", format)
	}

	slog.SetDefault(slog.New(handler))
	return nil
}

// newJournalHandler creates a slog handler that writes structured fields
// directly to the systemd journal via /run/systemd/journal/socket.
//
// slog keys are uppercased and hyphens replaced with underscores to match
// the journal field format (^[A-Z_][A-Z0-9_]*$).
func newJournalHandler(lvl slog.Level) (slog.Handler, error) {
	return slogjournal.NewHandler(&slogjournal.Options{
		Level: lvl,
		ReplaceAttr: func(_ []string, a slog.Attr) slog.Attr {
			a.Key = strings.ReplaceAll(strings.ToUpper(a.Key), "-", "_")
			return a
		},
	})
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
