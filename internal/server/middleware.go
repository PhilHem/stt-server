package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"

	"github.com/google/uuid"
)

// sanitizeRequestID strips unsafe characters and truncates to 128 chars.
func sanitizeRequestID(id string) string {
	// Allow UUID chars plus common ID formats
	var b strings.Builder
	for _, r := range id {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
		}
	}
	s := b.String()
	if len(s) > 128 {
		s = s[:128]
	}
	if s == "" {
		return uuid.NewString()
	}
	return s
}

// httpError writes a JSON error response and logs it.
func httpError(w http.ResponseWriter, r *http.Request, ctx context.Context, reqID string, code int, traceAttrs func(context.Context) []any, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	logAttrs := append([]any{
		"request_id", reqID,
		"status", code,
		"method", r.Method,
		"path", r.URL.Path,
		"error", msg,
	}, traceAttrs(ctx)...)
	slog.Warn("request error", logAttrs...)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "invalid_request_error",
			"code":    strconv.Itoa(code),
		},
	})
}

// propagatorCarrier adapts http.Header for OTel text map propagation.
type propagatorCarrier http.Header

func (c propagatorCarrier) Get(key string) string { return http.Header(c).Get(key) }
func (c propagatorCarrier) Set(key, value string) { http.Header(c).Set(key, value) }
func (c propagatorCarrier) Keys() []string {
	keys := make([]string, 0, len(c))
	for k := range c {
		keys = append(keys, k)
	}
	return keys
}
