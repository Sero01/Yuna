"""ASR audio helpers (padding trim, split at pauses) and TTS padding trim.

Run directly:  .venv/Scripts/python.exe tests/test_audio_segments.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pydub import AudioSegment  # noqa: E402
from pydub.generators import Sine  # noqa: E402

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.asr.audio_segments import split_at_pauses, trim_silence  # noqa: E402
from src.open_llm_vtuber.utils.audio_trim import trim_padding  # noqa: E402

SR = 16000


def tone(seconds, level=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds, noise=0.0):
    rng = np.random.default_rng(1)
    return (rng.standard_normal(int(seconds * SR)) * noise).astype(np.float32)


def test_trim_keeps_a_margin_around_the_speech():
    audio = np.concatenate([silence(0.64), tone(2.0), silence(1.12)])
    out = trim_silence(audio, SR, keep_s=0.3)
    assert abs(len(out) / SR - 2.6) < 0.03, len(out) / SR


def test_quiet_speech_is_not_trimmed_away():
    # A quiet mic: speech at about -44 dBFS, noise at about -80 dBFS.
    audio = np.concatenate([silence(0.5, 1e-4), tone(1.0, 0.009), silence(0.5, 1e-4)])
    out = trim_silence(audio, SR, keep_s=0.3)
    assert len(out) / SR >= 1.55, len(out) / SR


def test_all_silence_is_left_alone():
    audio = silence(1.0, 1e-5)
    assert len(trim_silence(audio, SR)) == len(audio)
    assert len(trim_silence(np.zeros(0, dtype=np.float32), SR)) == 0


def test_short_audio_is_one_piece():
    audio = tone(5.0)
    assert len(split_at_pauses(audio, 4, SR)) == 1


def test_long_audio_is_cut_in_its_pauses():
    words, pauses = [], []
    pos = 0
    for _ in range(8):
        words.append(tone(1.6))
        pos += len(words[-1])
        pauses.append((pos, pos + int(0.25 * SR)))
        words.append(silence(0.25))
        pos += len(words[-1])
    audio = np.concatenate(words)
    pieces = split_at_pauses(audio, 4, SR)
    assert len(pieces) == 4, [len(p) / SR for p in pieces]
    assert sum(len(p) for p in pieces) == len(audio), "nothing is lost or repeated"
    cut = 0
    for piece in pieces[:-1]:
        cut += len(piece)
        assert any(a <= cut <= b for a, b in pauses), f"cut at {cut / SR:.2f}s"


def test_pieces_are_never_shorter_than_the_minimum():
    audio = tone(7.0)
    pieces = split_at_pauses(audio, 4, SR, min_piece_s=3.0)
    assert len(pieces) == 2, [len(p) / SR for p in pieces]


def test_tts_padding_is_trimmed_to_a_short_lead_and_tail():
    voice = Sine(440).to_audio_segment(duration=500, volume=-6)
    clip = AudioSegment.silent(140) + voice + AudioSegment.silent(850)
    out = trim_padding(clip)
    assert 500 + 25 + 190 <= len(out) <= 500 + 25 + 210, len(out)


def test_tts_trim_leaves_silent_clips_alone():
    clip = AudioSegment.silent(300)
    assert len(trim_padding(clip)) == 300


if __name__ == "__main__":
    run_module(globals())
