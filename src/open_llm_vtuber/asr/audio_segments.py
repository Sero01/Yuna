"""Cheap energy-based helpers that make offline ASR faster on 16 kHz float audio.

The frontend's VAD sends ~0.6 s of audio before the speech and its ~1.1 s silence
window after it; decode time grows with length, so that padding is trimmed. Long
utterances are cut at their quietest moments so the pieces can be decoded in parallel.
"""

from typing import List

import numpy as np

FRAME = 320  # 20 ms at 16 kHz


def _frame_db(audio: np.ndarray, frame: int = FRAME) -> np.ndarray:
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0)
    frames = audio[: n * frame].astype(np.float32).reshape(n, frame)
    return 10 * np.log10((frames**2).mean(axis=1) + 1e-12)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    keep_s: float = 0.3,
    floor_db: float = -45.0,
    below_peak_db: float = 35.0,
) -> np.ndarray:
    """Drop leading/trailing silence, keeping `keep_s` of it on each side. Silence is
    below `floor_db`, or further below than that for a quiet mic: anything within
    `below_peak_db` of the loudest 20 ms counts as voice."""
    db = _frame_db(audio)
    if len(db) == 0:
        return audio
    voiced = np.nonzero(db > min(floor_db, float(db.max()) - below_peak_db))[0]
    if len(voiced) == 0:
        return audio
    keep = int(keep_s * sample_rate)
    start = max(0, int(voiced[0]) * FRAME - keep)
    end = min(len(audio), (int(voiced[-1]) + 1) * FRAME + keep)
    return audio[start:end]


def split_at_pauses(
    audio: np.ndarray,
    pieces: int,
    sample_rate: int = 16000,
    min_piece_s: float = 3.0,
) -> List[np.ndarray]:
    """Cut `audio` into up to `pieces` parts of similar length, each cut at the quietest
    20 ms within ±30% of where an even split would fall. Parts are never shorter than
    `min_piece_s`, so short audio comes back whole."""
    total_s = len(audio) / sample_rate
    pieces = max(1, min(pieces, int(total_s // min_piece_s)))
    if pieces == 1:
        return [audio]
    db = _frame_db(audio)
    step = len(db) / pieces
    cuts = [0]
    for i in range(1, pieces):
        lo = max(int((i - 0.3) * step), cuts[-1] + 1)
        hi = min(int((i + 0.3) * step), len(db) - 1)
        if hi <= lo:
            continue
        cuts.append(lo + int(np.argmin(db[lo:hi])))
    bounds = [c * FRAME for c in cuts] + [len(audio)]
    return [audio[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
