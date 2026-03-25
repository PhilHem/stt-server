"""Reference gRPC inference server for stt-server's remote backend.

Implements the InferenceEngine service (proto/stt.proto) with pluggable
backends. stt-server delegates audio decoding (ffmpeg) and sends raw
16kHz float32 samples over gRPC; this server handles mel extraction,
model inference, and CTC decoding.

Backends:
    echo        Testing — returns sample count without inference.
    nemo        NeMo ASR models (Parakeet, Conformer, etc.)
    torch       Raw PyTorch model (encoder-only, needs tokens.txt).
    poptorch    Graphcore IPU via PopTorch (Parakeet CTC, 4× IPU).

Usage:
    # Echo (test gRPC plumbing):
    python server.py --backend echo

    # NeMo (handles everything including mel + decoding):
    python server.py --backend nemo --model nvidia/parakeet-tdt-0.6b-v2

    # PyTorch encoder on CPU/CUDA (you provide the model + tokens):
    python server.py --backend torch --model ./parakeet_ctc.pt --tokens ./tokens.txt

    # Graphcore IPU (requires Poplar SDK + PopTorch):
    python server.py --backend poptorch --tokens ./tokens.txt --ipu-frames 3000
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent import futures
from typing import List

import grpc
import numpy as np

import stt_pb2
import stt_pb2_grpc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class InferenceBackend(ABC):
    """Base class for inference backends."""

    @abstractmethod
    def transcribe(
        self, samples: np.ndarray, sample_rate: int
    ) -> stt_pb2.TranscribeResponse:
        ...

    @abstractmethod
    def model_type(self) -> str:
        ...

    def close(self) -> None:
        """Release resources. Override if needed."""


# ---------------------------------------------------------------------------
# Echo backend
# ---------------------------------------------------------------------------

class EchoBackend(InferenceBackend):
    """Returns a fixed response. Useful for testing the gRPC plumbing."""

    def transcribe(self, samples, sample_rate):
        duration = len(samples) / sample_rate if sample_rate > 0 else 0.0
        return stt_pb2.TranscribeResponse(
            text=f"[echo] {len(samples)} samples at {sample_rate}Hz, {duration:.1f}s",
            language="en",
            duration=duration,
            inference_time_ms=0,
        )

    def model_type(self):
        return "echo"


# ---------------------------------------------------------------------------
# NeMo backend
# ---------------------------------------------------------------------------

class NeMoBackend(InferenceBackend):
    """Full NeMo ASR pipeline (mel + encoder + decoder + text)."""

    def __init__(self, model_name: str, device: str = "cpu"):
        import nemo.collections.asr as nemo_asr

        logger.info("Loading NeMo model %s on %s...", model_name, device)
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self._name = model_name
        logger.info("NeMo model loaded: %s", model_name)

    def transcribe(self, samples, sample_rate):
        import torch

        audio = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([len(samples)], dtype=torch.long)

        start = time.monotonic()
        with torch.no_grad():
            hypotheses = self.model.transcribe_audio(audio, lengths)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        text = ""
        tokens = []
        timestamps = []
        if hypotheses:
            hyp = hypotheses[0]
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            if hasattr(hyp, "tokens") and hyp.tokens:
                tokens = [str(t) for t in hyp.tokens]
            if hasattr(hyp, "timestamps") and hyp.timestamps:
                timestamps = [float(t) for t in hyp.timestamps]

        return stt_pb2.TranscribeResponse(
            text=text,
            language="",
            duration=len(samples) / sample_rate,
            inference_time_ms=elapsed_ms,
            tokens=tokens,
            timestamps=timestamps,
        )

    def model_type(self):
        return "nemo_transducer"


# ---------------------------------------------------------------------------
# PyTorch encoder backend (CPU/CUDA)
# ---------------------------------------------------------------------------

class TorchEncoderBackend(InferenceBackend):
    """Runs a PyTorch encoder model (e.g. Parakeet CTC) with mel + CTC decode.

    The model must accept [1, T, 80] mel input and return [1, T', vocab] logits.
    Pair with a tokens.txt for CTC decoding.
    """

    def __init__(self, model_path: str, tokens_path: str, device: str = "cpu"):
        import torch
        from decode import load_tokenizer, detect_blank_id

        logger.info("Loading PyTorch model %s on %s...", model_path, device)
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self._device = device
        self.tokens = load_tokenizer(tokens_path)
        self.blank_id = detect_blank_id(self.tokens)
        logger.info("Loaded %d tokens from %s (blank_id=%d)", len(self.tokens), tokens_path, self.blank_id)

    def transcribe(self, samples, sample_rate):
        import torch
        from audio import samples_to_mel
        from decode import ctc_greedy_decode

        audio_tensor = torch.tensor(samples, dtype=torch.float32)
        mel = samples_to_mel(audio_tensor, sample_rate)
        mel = mel.to(self._device)

        start = time.monotonic()
        with torch.no_grad():
            logits = self.model(mel)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        text, token_list, timestamps = ctc_greedy_decode(logits, self.tokens, blank_id=self.blank_id)

        return stt_pb2.TranscribeResponse(
            text=text,
            language="",
            duration=len(samples) / sample_rate,
            inference_time_ms=elapsed_ms,
            tokens=token_list,
            timestamps=[float(t) for t in timestamps],
        )

    def model_type(self):
        return "torch_encoder"


# ---------------------------------------------------------------------------
# PopTorch / Graphcore IPU backend
# ---------------------------------------------------------------------------

class PopTorchBackend(InferenceBackend):
    """Parakeet CTC 0.6B on Graphcore IPU via PopTorch.

    Compiles the model to a fixed input shape across 4 IPUs.
    Audio shorter than the compiled length is zero-padded;
    longer audio is truncated (compile multiple shapes for production).
    """

    def __init__(
        self,
        tokens_path: str,
        input_frames: int = 3000,
        model_name: str = "nvidia/parakeet-ctc-0.6b",
    ):
        import poptorch
        import torch
        from decode import load_tokenizer, detect_blank_id

        self.tokens = load_tokenizer(tokens_path)
        self.blank_id = detect_blank_id(self.tokens)
        logger.info("Loaded %d tokens from %s (blank_id=%d)", len(self.tokens), tokens_path, self.blank_id)

        self.input_frames = input_frames

        # Import model definition (must be on PYTHONPATH or in cwd)
        from parakeet_ipu_v7 import ParakeetCTC, load_weights
        from huggingface_hub import hf_hub_download

        logger.info("Building ParakeetCTC model...")
        model = ParakeetCTC()
        params = sum(p.numel() for p in model.parameters())
        logger.info("Parameters: %s (%.1fM)", f"{params:,}", params / 1e6)

        weights_path = hf_hub_download(model_name, "model.safetensors")
        model = load_weights(model, weights_path)
        model.eval()

        # Shard across 4 IPUs (6 layers each)
        for i in range(6, 12):
            poptorch.BeginBlock(model.layers[i], f"layer{i}", ipu_id=1)
        for i in range(12, 18):
            poptorch.BeginBlock(model.layers[i], f"layer{i}", ipu_id=2)
        for i in range(18, 24):
            poptorch.BeginBlock(model.layers[i], f"layer{i}", ipu_id=3)
        poptorch.BeginBlock(model.ctc_head, "ctc_head", ipu_id=3)

        opts = poptorch.Options()
        opts.replicationFactor(1)
        opts.setExecutionStrategy(poptorch.ShardedExecution())
        opts._Popart.set("saveInitializersToFile", "/tmp/parakeet_ipu_init.onnx")
        opts.setAvailableMemoryProportion({
            f"IPU{i}": 0.15 for i in range(4)
        })

        logger.info(
            "Compiling for 4 IPUs with %d input frames (~%.0fs audio)...",
            input_frames, input_frames * 0.01,
        )
        self.ipu_model = poptorch.inferenceModel(model, opts)
        self._lock = threading.Lock()
        dummy = torch.randn(1, input_frames, 80)
        start = time.monotonic()
        _ = self.ipu_model(dummy)
        logger.info("Compilation done in %.1fs", time.monotonic() - start)

    def transcribe(self, samples, sample_rate):
        import torch
        from audio import samples_to_mel, pad_to_length
        from decode import ctc_greedy_decode

        audio_tensor = torch.tensor(samples, dtype=torch.float32)
        mel = samples_to_mel(audio_tensor, sample_rate)

        actual_frames = mel.shape[1]
        if actual_frames > self.input_frames:
            logger.warning(
                "Audio truncated: %d frames → %d frames (%.1fs → %.1fs)",
                actual_frames, self.input_frames,
                actual_frames * 0.01, self.input_frames * 0.01,
            )

        # Pad to compiled static shape
        mel = pad_to_length(mel, self.input_frames)

        start = time.monotonic()
        with self._lock:
            with torch.no_grad():
                logits = self.ipu_model(mel)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        text, token_list, timestamps = ctc_greedy_decode(logits, self.tokens, blank_id=self.blank_id)

        return stt_pb2.TranscribeResponse(
            text=text,
            language="",
            duration=len(samples) / sample_rate,
            inference_time_ms=elapsed_ms,
            tokens=token_list,
            timestamps=[float(t) for t in timestamps],
        )

    def model_type(self):
        return "nemo_ctc_ipu"

    def close(self):
        if hasattr(self, "ipu_model"):
            self.ipu_model.destroy()


# ---------------------------------------------------------------------------
# Parakeet TDT backend (ONNX Runtime, full pipeline)
# ---------------------------------------------------------------------------

class ParakeetTDTBackend(InferenceBackend):
    """Parakeet TDT v3 via ONNX Runtime: encoder + decoder + joiner.

    Runs the full Token-and-Duration Transducer pipeline on CPU.
    Produces proper transcription with punctuation and casing.

    Model dir must contain: encoder.int8.onnx, decoder.int8.onnx,
    joiner.int8.onnx, tokens.txt.
    """

    BLANK_ID = 8192
    VOCAB_SIZE = 8193
    N_MELS = 128  # TDT v3 uses 128 mel bins

    def __init__(self, model_dir: str, num_threads: int = 4):
        import onnxruntime as ort
        from pathlib import Path

        model_dir = Path(model_dir)
        logger.info("Loading Parakeet TDT from %s...", model_dir)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        prov = ["CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(str(model_dir / "encoder.int8.onnx"), opts, providers=prov)
        self.decoder = ort.InferenceSession(str(model_dir / "decoder.int8.onnx"), opts, providers=prov)
        self.joiner = ort.InferenceSession(str(model_dir / "joiner.int8.onnx"), opts, providers=prov)

        # Load tokens
        self.token_map = {}
        with open(model_dir / "tokens.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.token_map[int(parts[1])] = parts[0]
        logger.info("Loaded %d tokens, encoder+decoder+joiner ready", len(self.token_map))

    def transcribe(self, samples, sample_rate):
        # Mel spectrogram (128-dim, per-feature normalized)
        mel = self._compute_mel(samples, sample_rate)

        # Encoder
        length = np.array([mel.shape[2]], dtype=np.int64)
        start = time.monotonic()
        enc_out, enc_lens = self.encoder.run(None, {"audio_signal": mel, "length": length})
        enc_len = int(enc_lens[0])

        # TDT greedy decode
        token_ids = self._tdt_decode(enc_out, enc_len)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Tokens to text
        text = "".join(self.token_map.get(i, "") for i in token_ids)
        text = text.replace("\u2581", " ").strip()

        return stt_pb2.TranscribeResponse(
            text=text,
            language="",
            duration=len(samples) / sample_rate,
            inference_time_ms=elapsed_ms,
        )

    def _tdt_decode(self, encoder_out, enc_len):
        """Greedy TDT search with duration-based frame skipping."""
        context_size = 2
        hyp = [self.BLANK_ID] * context_size
        states = np.zeros((2, 1, 640), dtype=np.float32)
        cell = np.zeros((2, 1, 640), dtype=np.float32)
        target_len = np.array([context_size], dtype=np.int32)

        # Initial decoder pass
        targets = np.array([hyp[-context_size:]], dtype=np.int32)
        dec_out, _, states, cell = self.decoder.run(None, {
            "targets": targets, "target_length": target_len,
            "states.1": states, "onnx::Slice_3": cell,
        })
        dec_output = dec_out[:, :, -1:]

        emitted = []
        t = 0
        while t < enc_len:
            enc_frame = encoder_out[:, :, t:t + 1]
            symbols = 0
            while symbols < 10:
                logits = self.joiner.run(None, {
                    "encoder_outputs": enc_frame,
                    "decoder_outputs": dec_output,
                })[0].reshape(-1)

                best_tok = int(np.argmax(logits[:self.VOCAB_SIZE]))
                best_dur = int(np.argmax(logits[self.VOCAB_SIZE:]))

                if best_tok == self.BLANK_ID:
                    t += max(1, best_dur)
                    break

                emitted.append(best_tok)
                hyp.append(best_tok)
                symbols += 1

                targets = np.array([hyp[-context_size:]], dtype=np.int32)
                dec_out, _, states, cell = self.decoder.run(None, {
                    "targets": targets, "target_length": target_len,
                    "states.1": states, "onnx::Slice_3": cell,
                })
                dec_output = dec_out[:, :, -1:]

                if best_dur > 0:
                    t += best_dur
                    break

        return emitted

    def _compute_mel(self, samples, sample_rate):
        """128-dim log-mel spectrogram, per-feature normalized."""
        N_FFT, WIN_LEN, HOP_LEN = 512, 400, 160
        window = np.hanning(WIN_LEN + 1)[:-1].astype(np.float32)
        pad_len = N_FFT // 2
        padded = np.pad(samples, (pad_len, pad_len), mode="reflect")
        n_frames = 1 + (len(padded) - N_FFT) // HOP_LEN
        frames = np.zeros((n_frames, N_FFT), dtype=np.float32)
        for i in range(n_frames):
            s = i * HOP_LEN
            frames[i, :WIN_LEN] = padded[s:s + WIN_LEN] * window
        spec = np.fft.rfft(frames, n=N_FFT)
        power = np.abs(spec) ** 2
        fb = self._mel_fb(N_FFT, self.N_MELS, sample_rate)
        mel = np.log(np.maximum(power @ fb.T, 1e-5))
        mean = mel.mean(axis=0, keepdims=True)
        std = mel.std(axis=0, keepdims=True)
        mel = (mel - mean) / (std + 1e-5)
        return mel.T[np.newaxis, :, :].astype(np.float32)

    @staticmethod
    def _mel_fb(n_fft, n_mels, sr):
        hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        n_freqs = n_fft // 2 + 1
        pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
        freqs = np.linspace(0, sr / 2, n_freqs)
        fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(n_mels):
            lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
            fb[i] = np.maximum(0, np.minimum(
                (freqs - lo) / (c - lo + 1e-10),
                (hi - freqs) / (hi - c + 1e-10)))
            bw = hi - lo
            if bw > 0: fb[i] /= bw
        return fb

    def model_type(self):
        return "nemo_tdt"


# ---------------------------------------------------------------------------
# gRPC servicer
# ---------------------------------------------------------------------------

class InferenceServicer(stt_pb2_grpc.InferenceEngineServicer):
    def __init__(self, backend: InferenceBackend):
        self.backend = backend

    def Transcribe(self, request, context):
        MAX_SAMPLES = 16000 * 600  # 10 minutes at 16kHz

        if len(request.samples) % 4 != 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "samples must be float32 aligned (multiple of 4 bytes)",
            )
            return
        if len(request.samples) == 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "empty audio"
            )
            return
        if request.sample_rate <= 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "sample_rate must be positive"
            )
            return
        if len(request.samples) // 4 > MAX_SAMPLES:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "audio exceeds maximum of 600 seconds",
            )
            return

        samples = np.frombuffer(request.samples, dtype="<f4")

        if not np.isfinite(samples).all():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "audio contains non-finite values",
            )
            return

        logger.info(
            "Transcribe: %d samples, %dHz, %.1fs",
            len(samples),
            request.sample_rate,
            len(samples) / request.sample_rate if request.sample_rate > 0 else 0,
        )
        try:
            resp = self.backend.transcribe(samples, request.sample_rate)
            logger.info(
                "Result: %dms inference, %d chars",
                resp.inference_time_ms,
                len(resp.text),
            )
            return resp
        except Exception as e:
            logger.exception("Inference failed")
            context.abort(grpc.StatusCode.INTERNAL, "inference failed")
            return

    def GetModelType(self, request, context):
        return stt_pb2.GetModelTypeResponse(
            model_type=self.backend.model_type()
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="gRPC inference server for stt-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend",
        choices=["echo", "nemo", "torch", "poptorch", "parakeet-tdt"],
        default="echo",
    )
    parser.add_argument("--model", default="", help="Model name or path")
    parser.add_argument("--tokens", default="", help="Path to tokens.txt (for torch/poptorch)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, cuda:0, ...")
    parser.add_argument("--port", type=int, default=50051, help="gRPC listen port")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent RPCs")
    parser.add_argument(
        "--ipu-frames", type=int, default=3000,
        help="Static input frames for IPU compilation (3000 ≈ 30s audio)",
    )
    args = parser.parse_args()

    if args.ipu_frames <= 0 or args.ipu_frames > 50000:
        parser.error("--ipu-frames must be between 1 and 50000")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Create backend
    backend: InferenceBackend
    if args.backend == "echo":
        backend = EchoBackend()
    elif args.backend == "nemo":
        if not args.model:
            parser.error("--model required for nemo backend")
        backend = NeMoBackend(args.model, args.device)
    elif args.backend == "torch":
        if not args.model:
            parser.error("--model required for torch backend")
        if not args.tokens:
            parser.error("--tokens required for torch backend")
        backend = TorchEncoderBackend(args.model, args.tokens, args.device)
    elif args.backend == "poptorch":
        if not args.tokens:
            parser.error("--tokens required for poptorch backend")
        backend = PopTorchBackend(
            tokens_path=args.tokens,
            input_frames=args.ipu_frames,
            model_name=args.model or "nvidia/parakeet-ctc-0.6b",
        )
    elif args.backend == "parakeet-tdt":
        if not args.model:
            parser.error("--model (model directory) required for parakeet-tdt backend")
        backend = ParakeetTDTBackend(args.model)
    else:
        parser.error(f"Unknown backend: {args.backend}")

    # Smoke test: verify backend can handle a minimal inference
    if args.backend != "echo":
        logger.info("Running startup smoke test...")
        try:
            import struct
            # 100ms of silence at 16kHz = 1600 samples
            test_samples = np.zeros(1600, dtype=np.float32)
            resp = backend.transcribe(test_samples, 16000)
            logger.info("Smoke test passed: %dms inference", resp.inference_time_ms)
        except Exception as e:
            logger.error("Smoke test FAILED — backend cannot serve: %s", e)
            backend.close()
            sys.exit(1)

    # Start server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    stt_pb2_grpc.add_InferenceEngineServicer_to_server(
        InferenceServicer(backend), server
    )
    addr = f"[::]:{args.port}"
    bound_port = server.add_insecure_port(addr)
    if bound_port == 0:
        logger.error("Failed to bind to %s (port in use?)", addr)
        backend.close()
        sys.exit(1)
    server.start()
    logger.info("Listening on %s (backend=%s)", addr, args.backend)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        backend.close()
        server.stop(grace=5)


if __name__ == "__main__":
    main()
