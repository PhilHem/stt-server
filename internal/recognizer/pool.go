package recognizer

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// Pool manages a channel-based pool of independent Engine instances.
// Each instance is borrowed for a single transcription and returned.
type Pool struct {
	instances chan Engine
	size      int
	modelType string
}

// NewPool creates a pool of N engine instances using the given factory.
func NewPool(factory EngineFactory, cfg Config, n int) (*Pool, error) {
	if n < 1 {
		n = 1
	}

	pool := &Pool{
		instances: make(chan Engine, n),
		size:      n,
	}

	for i := 0; i < n; i++ {
		eng, err := factory(cfg)
		if err != nil {
			// Clean up already-created instances
			pool.Close()
			return nil, fmt.Errorf("create engine instance %d/%d: %w", i+1, n, err)
		}
		if i == 0 {
			pool.modelType = eng.ModelType()
		}
		pool.instances <- eng
		slog.Info("engine instance created", "instance", i+1, "total", n)
	}

	return pool, nil
}

// Transcribe borrows an engine from the pool, runs inference, and returns it.
func (p *Pool) Transcribe(ctx context.Context, samples []float32, sampleRate int) (*TranscriptionResult, error) {
	select {
	case eng := <-p.instances:
		defer func() { p.instances <- eng }()
		return eng.Transcribe(ctx, samples, sampleRate)
	case <-ctx.Done():
		return nil, fmt.Errorf("pool: all %d instances busy: %w", p.size, ctx.Err())
	}
}

// Close drains the pool and closes all instances.
func (p *Pool) Close() {
	for {
		select {
		case eng := <-p.instances:
			eng.Close()
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

// Ping checks that at least one engine instance is available and responsive.
// Returns nil if healthy, an error otherwise. Uses a short timeout to avoid
// blocking the readiness probe.
func (p *Pool) Ping(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	select {
	case eng := <-p.instances:
		defer func() { p.instances <- eng }()
		// Engine is available — for remote backends this confirms the
		// connection was established. ModelType() is a cached value,
		// so this doesn't make a network call.
		if eng.ModelType() == "" {
			return fmt.Errorf("engine has no model type")
		}
		return nil
	case <-ctx.Done():
		return fmt.Errorf("ping: all %d instances busy", p.size)
	}
}
