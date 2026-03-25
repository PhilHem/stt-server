package remote

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	"github.com/PhilHem/stt-server/internal/recognizer"
	"github.com/PhilHem/stt-server/internal/recognizer/remote/sttpb"
)

// Engine is a gRPC client adapter that delegates inference to a remote server.
// The remote server can run any framework (PopTorch, libtorch, NeMo, etc.).
type Engine struct {
	conn      *grpc.ClientConn
	client    sttpb.InferenceEngineClient
	modelType string
}

// New connects to a remote inference server and returns an Engine.
// The endpoint should be a host:port string (e.g. "localhost:50051").
func New(endpoint string) (recognizer.Engine, error) {
	conn, err := grpc.NewClient(endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(64<<20), // 64 MB
			grpc.MaxCallSendMsgSize(64<<20), // 64 MB
		),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second,
			Timeout:             5 * time.Second,
			PermitWithoutStream: true,
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("connect to inference server %s: %w", endpoint, err)
	}

	client := sttpb.NewInferenceEngineClient(conn)

	// Query model type at startup
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := client.GetModelType(ctx, &sttpb.GetModelTypeRequest{})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("query model type from %s: %w", endpoint, err)
	}

	return &Engine{
		conn:      conn,
		client:    client,
		modelType: resp.ModelType,
	}, nil
}

// Transcribe sends audio samples to the remote inference server.
func (e *Engine) Transcribe(ctx context.Context, samples []float32, sampleRate int) (*recognizer.TranscriptionResult, error) {
	if len(samples) == 0 {
		return &recognizer.TranscriptionResult{Duration: 0}, nil
	}

	// Apply a gRPC-specific deadline (defense-in-depth against long HTTP timeouts)
	grpcCtx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	resp, err := e.client.Transcribe(grpcCtx, &sttpb.TranscribeRequest{
		Samples:    float32ToBytes(samples),
		SampleRate: int32(sampleRate),
	})
	if err != nil {
		return nil, fmt.Errorf("remote transcribe: %w", err)
	}

	return &recognizer.TranscriptionResult{
		Text:          resp.Text,
		Language:      resp.Language,
		Duration:      resp.Duration,
		InferenceTime: time.Duration(resp.InferenceTimeMs) * time.Millisecond,
		Tokens:        resp.Tokens,
		Timestamps:    resp.Timestamps,
	}, nil
}

// Close shuts down the gRPC connection.
func (e *Engine) Close() {
	e.conn.Close()
}

// ModelType returns the model type reported by the remote server.
func (e *Engine) ModelType() string {
	return e.modelType
}

// float32ToBytes packs a float32 slice into little-endian bytes.
func float32ToBytes(f []float32) []byte {
	buf := make([]byte, len(f)*4)
	for i, v := range f {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}
