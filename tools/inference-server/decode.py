"""CTC decoding: logits → text.

Supports greedy CTC decoding with the SentencePiece token convention
used by NeMo Parakeet models (leading space = new word).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import torch

_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def load_tokenizer(tokens_path: str) -> List[str]:
    """Load token vocabulary from tokens.txt.

    Format: one token per line, token ID = line number.
    Use detect_blank_id() to find the CTC blank position.
    """
    tokens = []
    try:
        with open(tokens_path) as f:
            for line in f:
                raw = line.rstrip('\n')
                parts = raw.split()
                if len(parts) >= 1:
                    tokens.append(_CONTROL_CHARS.sub('', parts[0]))
                else:
                    # Preserve blank lines as empty tokens to maintain ID alignment
                    tokens.append('')
    except (OSError, IOError) as e:
        raise RuntimeError(f"cannot load tokens from {tokens_path}: {e}") from e
    return tokens


def detect_blank_id(tokens: List[str]) -> int:
    """Find the CTC blank token ID in the vocabulary.

    NeMo CTC models place blank at the END of the vocab (e.g., ID 1024).
    sherpa-onnx models place blank at ID 0. This detects either convention.
    """
    for i, t in enumerate(tokens):
        if t in ('<blk>', '<blank>', '[blank]'):
            return i
    # Default: last token (NeMo convention)
    return len(tokens) - 1


def ctc_greedy_decode(
    logits: torch.Tensor,
    tokens: List[str],
    blank_id: int = 0,
) -> Tuple[str, List[str], List[float]]:
    """Greedy CTC decoding with token merging.

    Args:
        logits: [1, T, vocab_size] or [T, vocab_size] tensor of log-probabilities.
        tokens: Vocabulary list (index = token ID).
        blank_id: CTC blank token ID (default 0).

    Returns:
        (text, token_list, timestamps) where:
        - text: Decoded string with SentencePiece spaces resolved.
        - token_list: Per-token strings (before merging).
        - timestamps: Per-token start times in seconds (at 10ms resolution).
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)  # [T, vocab_size]

    # Argmax per frame
    pred_ids = logits.argmax(dim=-1).tolist()  # [T]

    # Collapse repeated tokens and remove blanks
    decoded_ids = []
    decoded_frames = []
    prev = -1
    for frame_idx, tid in enumerate(pred_ids):
        if tid != prev and tid != blank_id:
            decoded_ids.append(tid)
            decoded_frames.append(frame_idx)
        prev = tid

    # Map to token strings
    token_list = []
    for tid in decoded_ids:
        if 0 <= tid < len(tokens):
            token_list.append(tokens[tid])
        else:
            token_list.append("<unk>")

    # Timestamps: frame index × hop_length / sample_rate
    # NeMo uses 10ms hops → 1 frame = 0.01s
    # But after subsampling (stride 8), 1 output frame ≈ 0.08s
    # The exact ratio depends on the model's subsampling factor.
    # Parakeet CTC uses 8× subsampling (3 conv layers with stride 2).
    SUBSAMPLE_FACTOR = 8
    FRAME_DURATION = 0.01 * SUBSAMPLE_FACTOR  # 80ms per output frame
    timestamps = [frame * FRAME_DURATION for frame in decoded_frames]

    # Merge tokens to text (SentencePiece convention: ▁ = space/word boundary)
    text = "".join(token_list)
    # Replace SentencePiece space marker with actual space
    text = text.replace("▁", " ").strip()

    return text, token_list, timestamps
