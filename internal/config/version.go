package config

// Set at build time via:
//
//	go build -ldflags "-X github.com/PhilHem/stt-server/internal/config.Version=v0.3.0
//	                    -X github.com/PhilHem/stt-server/internal/config.Commit=abc1234
//	                    -X github.com/PhilHem/stt-server/internal/config.BuildTime=2026-03-25T00:00:00Z"
var (
	Version   = "dev"
	Commit    = "unknown"
	BuildTime = "unknown"
)
