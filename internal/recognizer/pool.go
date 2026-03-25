package recognizer

import (
	"context"
	"fmt"
	"log/slog"
)

// Pool manages a channel-based pool of independent Recognizer instances.
// Each instance has its own sherpa-onnx C object and mutex, enabling
// true parallel inference.
type Pool struct {
	instances chan *Recognizer
	size      int
	modelType string
}

// NewPool creates a pool of N recognizer instances from the same config.
func NewPool(cfg Config, n int) (*Pool, error) {
	if n < 1 {
		n = 1
	}

	pool := &Pool{
		instances: make(chan *Recognizer, n),
		size:      n,
	}

	for i := 0; i < n; i++ {
		rec, err := New(cfg)
		if err != nil {
			// Clean up already-created instances
			pool.Close()
			return nil, fmt.Errorf("create recognizer instance %d/%d: %w", i+1, n, err)
		}
		if i == 0 {
			pool.modelType = rec.ModelType
		}
		pool.instances <- rec
		slog.Info("recognizer instance created", "instance", i+1, "total", n)
	}

	return pool, nil
}

// Transcribe borrows a recognizer from the pool, runs inference, and returns it.
func (p *Pool) Transcribe(ctx context.Context, samples []float32, sampleRate int) (*TranscriptionResult, error) {
	select {
	case rec := <-p.instances:
		defer func() { p.instances <- rec }()
		return rec.Transcribe(ctx, samples, sampleRate)
	case <-ctx.Done():
		return nil, fmt.Errorf("pool: all %d instances busy: %w", p.size, ctx.Err())
	}
}

// Close drains the pool and closes all instances.
func (p *Pool) Close() {
	for {
		select {
		case rec := <-p.instances:
			rec.Close()
		default:
			return
		}
	}
}

// ModelType returns the model type. All instances share the same model type
// since they are created from the same config.
func (p *Pool) ModelType() string {
	return p.modelType
}

// Size returns the number of instances in the pool.
func (p *Pool) Size() int {
	return p.size
}
