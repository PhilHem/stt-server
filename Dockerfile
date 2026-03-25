# Base images pinned by digest for reproducible builds.
# Update digests periodically: docker manifest inspect <image> | grep digest

# Stage 1: Build minimal static ffmpeg (audio decode only, no external libs)
# Pin updated: 2026-03-25. Run `docker manifest inspect debian:bookworm-slim` to refresh.
FROM debian:bookworm-slim@sha256:8af0e5095f9964007f5ebd11191dfe52dcb51bf3afa2c07f055fc5451b78ba0e AS ffmpeg-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nasm curl ca-certificates xz-utils && \
    rm -rf /var/lib/apt/lists/*

ARG FFMPEG_VERSION=7.1
# SHA-256 from https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz.sha256
ARG FFMPEG_SHA256=40973d44970dbc83ef302b0609f2e74982be2d85916dd2ee7472d30678a7abe6

RUN curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o /tmp/ffmpeg.tar.xz && \
    echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c && \
    tar xJf /tmp/ffmpeg.tar.xz && \
    rm /tmp/ffmpeg.tar.xz

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
# Pin updated: 2026-03-25. Run `docker manifest inspect golang:1.26-bookworm` to refresh.
FROM golang:1.26-bookworm@sha256:77d2fa8be6beead13c85eb83d016c17806a376015a8b6a7ba24bc4c992e654b5 AS go-builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_TIME=unknown

COPY cmd/ cmd/
COPY internal/ internal/
RUN CGO_ENABLED=1 go build -ldflags="-s -w \
    -X github.com/PhilHem/stt-server/internal/config.Version=${VERSION} \
    -X github.com/PhilHem/stt-server/internal/config.Commit=${COMMIT} \
    -X github.com/PhilHem/stt-server/internal/config.BuildTime=${BUILD_TIME}" \
    -o /stt-server ./cmd/stt-server/

# Copy sherpa-onnx shared libraries from the Go module cache
RUN cp $(go env GOMODCACHE)/github.com/k2-fsa/sherpa-onnx-go-linux@*/lib/x86_64-unknown-linux-gnu/*.so /usr/local/lib/ && \
    ldconfig

# ---

# Stage 3: Minimal runtime
# Pin updated: 2026-03-25. Run `docker manifest inspect debian:bookworm-slim` to refresh.
FROM debian:bookworm-slim@sha256:8af0e5095f9964007f5ebd11191dfe52dcb51bf3afa2c07f055fc5451b78ba0e

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-builder /ffmpeg /usr/local/bin/ffmpeg
COPY --from=go-builder /stt-server /usr/local/bin/stt-server
COPY --from=go-builder /usr/local/lib/libsherpa-onnx-c-api.so /usr/local/lib/
COPY --from=go-builder /usr/local/lib/libsherpa-onnx-cxx-api.so /usr/local/lib/
COPY --from=go-builder /usr/local/lib/libonnxruntime.so /usr/local/lib/
RUN ldconfig

RUN groupadd -r stt && useradd -r -g stt -s /sbin/nologin stt
USER stt

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
