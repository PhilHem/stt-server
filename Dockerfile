# Stage 1: Build minimal static ffmpeg (audio decode only, no external libs)
FROM debian:bookworm-slim AS ffmpeg-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nasm curl ca-certificates xz-utils && \
    rm -rf /var/lib/apt/lists/*

ARG FFMPEG_VERSION=7.1

RUN curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" | tar xJ

WORKDIR /ffmpeg-${FFMPEG_VERSION}
RUN ./configure \
    --disable-everything \
    --enable-ffmpeg \
    --disable-ffplay --disable-ffprobe \
    --disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages \
    --disable-network \
    --disable-autodetect \
    --enable-small \
    --enable-static --disable-shared \
    --enable-protocol=file,pipe \
    --enable-demuxer=mp3,flac,ogg,mov,wav,aac,matroska,webm \
    --enable-decoder=mp3,mp3float,flac,vorbis,opus,aac,pcm_s16le,pcm_s16be,pcm_f32le,pcm_f32be,pcm_mulaw,pcm_alaw \
    --enable-parser=mp3,flac,vorbis,opus,aac \
    --enable-muxer=pcm_s16le \
    --enable-encoder=pcm_s16le \
    --enable-filter=aresample,aformat,anull \
    --enable-swresample \
    --extra-cflags="-Os" \
    --extra-ldflags="-static" && \
    make -j$(nproc) && \
    strip ffmpeg && cp ffmpeg /ffmpeg

# ---

# Stage 2: Build Go binary
FROM golang:1.26-bookworm AS go-builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY *.go ./
RUN CGO_ENABLED=1 go build -ldflags="-s -w" -o /stt-server .

# Copy sherpa-onnx shared libraries from the Go module cache
RUN cp $(go env GOMODCACHE)/github.com/k2-fsa/sherpa-onnx-go-linux@*/lib/x86_64-unknown-linux-gnu/*.so /usr/local/lib/ && \
    ldconfig

# ---

# Stage 3: Minimal runtime
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-builder /ffmpeg /usr/local/bin/ffmpeg
COPY --from=go-builder /stt-server /usr/local/bin/stt-server
COPY --from=go-builder /usr/local/lib/libsherpa-onnx-c-api.so /usr/local/lib/
COPY --from=go-builder /usr/local/lib/libsherpa-onnx-cxx-api.so /usr/local/lib/
COPY --from=go-builder /usr/local/lib/libonnxruntime.so /usr/local/lib/
RUN ldconfig

# Model: pass a name to auto-download, or mount a local directory.
#   Auto-download:  -e STT_MODEL=sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
#   Local mount:    -e STT_MODEL=/model -v /path/to/model:/model:ro
#
# Cache dir persists downloaded models across container restarts.
ENV STT_MODEL=""
ENV STT_CACHE_DIR=/opt/stt/cache
ENV STT_PORT=8000
VOLUME /opt/stt/cache

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["stt-server"]
