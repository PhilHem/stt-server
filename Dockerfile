FROM golang:1.24-bookworm AS builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY *.go ./
RUN CGO_ENABLED=1 go build -o /stt-server .

# Copy sherpa-onnx shared libraries from the Go module cache
RUN cp $(go env GOMODCACHE)/github.com/k2-fsa/sherpa-onnx-go-linux@*/lib/x86_64-unknown-linux-gnu/*.so /usr/local/lib/ && \
    ldconfig

# Static ffmpeg binary (all codecs, no deps, ~70 MB)
FROM mwader/static-ffmpeg:7.1 AS ffmpeg

# ---

FROM debian:bookworm-slim

# curl for healthcheck, ca-certificates for TLS
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
COPY --from=builder /stt-server /usr/local/bin/stt-server
COPY --from=builder /usr/local/lib/libsherpa-onnx-c-api.so /usr/local/lib/
COPY --from=builder /usr/local/lib/libsherpa-onnx-cxx-api.so /usr/local/lib/
COPY --from=builder /usr/local/lib/libonnxruntime.so /usr/local/lib/
RUN ldconfig

# Model is mounted at runtime: -v /path/to/model:/model
ENV STT_MODEL=/model
ENV STT_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["stt-server"]
