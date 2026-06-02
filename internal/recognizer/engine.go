package recognizer

import (
	"context"
	"time"
)

// TranscriptionResult holds the output of a transcription.
type TranscriptionResult struct {
	Text          string
	Language      string
	Duration      float32       // audio duration in seconds
	InferenceTime time.Duration // model inference time
	Tokens        []string      // per-token text
	Timestamps    []float32     // per-token start time in seconds
	Segments      []SpeakerTurn // speaker-attributed turns; empty unless diarization ran
}

// SpeakerTurn is a contiguous stretch of speech by one speaker. Speaker is a
// zero-based index assigned by diarization (not a real identity).
type SpeakerTurn struct {
	Speaker int
	Start   float32 // seconds
	End     float32 // seconds
	Text    string
}

// Engine is the port for speech recognition backends.
// Implementations must be safe for sequential use (the Pool serializes access).
type Engine interface {
	// Transcribe runs speech recognition on the given audio samples.
	Transcribe(ctx context.Context, samples []float32, sampleRate int) (*TranscriptionResult, error)

	// Close releases all resources held by the engine.
	Close()

	// ModelType returns a human-readable identifier for the loaded model
	// (e.g. "nemo_transducer", "whisper").
	ModelType() string
}

// EngineFactory creates a new Engine instance from the given config.
// Pool calls this N times to populate itself.
type EngineFactory func(cfg Config) (Engine, error)

// Config holds the parameters needed to create a recognition engine.
type Config struct {
	ModelDir   string
	NumThreads int
	Provider   string // "cpu" or "cuda"
	Language   string // ISO-639-1 hint (used by Whisper/SenseVoice, ignored by Parakeet)
	VadModel   string // optional Silero VAD model path; enables VAD segmentation

	// SegmentationModel and EmbeddingModel together enable speaker diarization:
	// audio is split into per-speaker turns before recognition. Both must be set.
	SegmentationModel string // pyannote segmentation ONNX path
	EmbeddingModel    string // speaker-embedding ONNX path
}
