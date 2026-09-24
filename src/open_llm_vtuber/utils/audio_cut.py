"""Cutting words out of synthesized speech using the TTS engine's word timings."""

from typing import List, Tuple

from pydub import AudioSegment

MAX_TAIL_S = 0.12  # keep up to this much of the pause after the last kept word
LEAD_S = 0.03
FADE_MS = 20


def cut_after_words(
    audio: AudioSegment, words: List[Tuple[float, float]], n: int
) -> AudioSegment:
    """Keep the first `n` words of `audio` (word timings as (offset s, duration s)),
    plus part of the pause after them, with a short fade so the cut doesn't click."""
    start = max(0.0, words[0][0] - LEAD_S)
    end = words[n - 1][0] + words[n - 1][1]
    next_start = words[n][0] if len(words) > n else audio.duration_seconds
    end += min(MAX_TAIL_S, max(0.0, next_start - end) / 2)
    return audio[int(start * 1000) : int(end * 1000)].fade_out(FADE_MS)
