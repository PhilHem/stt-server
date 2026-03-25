"""Audio preprocessing: raw PCM samples → mel spectrogram.

Uses torch for STFT + mel filterbank to match NeMo's feature extraction.
NeMo Parakeet uses 80-dim log-mel with 25ms windows, 10ms hops, 512-pt FFT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# NeMo default feature extraction parameters (FilterbankFeatures)
SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400   # 25ms at 16kHz
HOP_LENGTH = 160   # 10ms at 16kHz → 100 frames/sec
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0

# Pre-normalized log floor
LOG_FLOOR = 1e-5

_HANN_WINDOW = None
_MEL_FB = None

def _get_hann_window():
    global _HANN_WINDOW
    if _HANN_WINDOW is None:
        _HANN_WINDOW = torch.hann_window(WIN_LENGTH)
    return _HANN_WINDOW

def _get_mel_fb():
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE, F_MIN, F_MAX)
    return _MEL_FB


def samples_to_mel(
    samples: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """Convert raw audio samples to log-mel spectrogram.

    Args:
        samples: 1-D float32 tensor of audio samples.
        sample_rate: Input sample rate (must be 16000).
        n_mels: Number of mel bins (default 80).

    Returns:
        Tensor of shape [1, T, n_mels] — batch of 1, T frames, 80 features.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sample_rate}Hz")

    # Ensure 1-D
    if samples.dim() == 0:
        return torch.zeros(1, 1, n_mels)
    samples = samples.view(-1)
    if samples.numel() < WIN_LENGTH:
        return torch.zeros(1, 1, n_mels)

    # STFT
    window = _get_hann_window()
    stft = torch.stft(
        samples,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    # Power spectrum: [freq_bins, T]
    power = stft.abs().pow(2)

    # Mel filterbank
    mel_fb = _get_mel_fb()
    # [n_mels, T]
    mel_spec = torch.matmul(mel_fb, power)

    # Log
    mel_spec = torch.log(mel_spec.clamp(min=LOG_FLOOR))

    # [T, n_mels]
    mel_spec = mel_spec.T

    # Per-feature normalization (matches NeMo's normalize="per_feature"):
    # subtract mean, divide by std for each mel bin independently.
    mean = mel_spec.mean(dim=0, keepdim=True)
    std = mel_spec.std(dim=0, keepdim=True)
    mel_spec = (mel_spec - mean) / (std + 1e-5)

    # [1, T, n_mels]
    return mel_spec.unsqueeze(0)


def pad_to_length(mel: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Pad or truncate mel spectrogram to a fixed number of frames.

    IPU and some static-graph backends require fixed input shapes.

    Args:
        mel: [1, T, n_mels] tensor.
        target_frames: Desired number of frames.

    Returns:
        [1, target_frames, n_mels] tensor (zero-padded or truncated).
    """
    _, t, n_mels = mel.shape
    if t >= target_frames:
        return mel[:, :target_frames, :]
    # Pad on the time dimension
    pad_amount = target_frames - t
    return F.pad(mel, (0, 0, 0, pad_amount))


@torch.no_grad()
def _mel_filterbank(
    n_fft: int, n_mels: int, sample_rate: int, f_min: float, f_max: float
) -> torch.Tensor:
    """Build a mel filterbank matrix [n_mels, n_fft//2+1]."""
    # Mel scale conversions
    def hz_to_mel(f: float) -> float:
        return 2595.0 * torch.log10(torch.tensor(1.0 + f / 700.0)).item()

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Equally spaced mel points
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin frequencies
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

    # Build triangular filters
    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        low = hz_points[i]
        center = hz_points[i + 1]
        high = hz_points[i + 2]

        # Rising slope
        up = (fft_freqs - low) / (center - low + 1e-10)
        # Falling slope
        down = (high - fft_freqs) / (high - center + 1e-10)

        fb[i] = torch.max(torch.zeros_like(fft_freqs), torch.min(up, down))

    # Slaney normalization: divide each filter by its bandwidth in Hz.
    # This matches librosa.filters.mel(norm='slaney') and NeMo's
    # FilterbankFeatures, ensuring equal energy per mel band.
    for i in range(n_mels):
        bandwidth = hz_points[i + 2] - hz_points[i]
        if bandwidth > 0:
            fb[i] /= bandwidth

    return fb
