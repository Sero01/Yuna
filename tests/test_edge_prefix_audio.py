"""edge-tts renders an opener inside a carrier sentence and cuts it out by word timing.

Run directly:  .venv/Scripts/python.exe tests/test_edge_prefix_audio.py
"""

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydub import AudioSegment  # noqa: E402
from pydub.generators import Sine  # noqa: E402

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.tts import edge_tts as edge_module  # noqa: E402

TICKS_PER_S = 10_000_000  # edge-tts reports offsets in 100 ns units


def fake_communicate(words, calls):
    audio = io.BytesIO()
    Sine(300).to_audio_segment(duration=2000).export(audio, format="wav")
    data = audio.getvalue()

    class FakeCommunicate:
        def __init__(self, text, voice, **kwargs):
            calls.append({"text": text, "voice": voice, **kwargs})

        def stream_sync(self):
            for offset, duration, text in words:
                yield {
                    "type": "WordBoundary",
                    "offset": int(offset * TICKS_PER_S),
                    "duration": int(duration * TICKS_PER_S),
                    "text": text,
                }
            yield {"type": "audio", "data": data[:1000]}
            yield {"type": "audio", "data": data[1000:]}

    return FakeCommunicate


def test_prefix_audio_is_cut_after_the_prefix_words():
    calls = []
    words = [(0.1, 0.4, "Humph"), (0.8, 0.1, "I"), (0.9, 0.4, "suppose")]
    original = edge_module.edge_tts.Communicate
    edge_module.edge_tts.Communicate = fake_communicate(words, calls)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            engine = edge_module.TTSEngine(voice="en-GB-MaisieNeural")
            engine.new_audio_dir = tmp
            path = engine.generate_prefix_audio("Humph,", "I suppose.", "opener_x")
            assert path and path.endswith(".wav") and os.path.exists(path), path
            clip = AudioSegment.from_file(path)
            # 0.07 s .. 0.5 s + min(0.12, half the 0.3 s pause) -> 550 ms
            assert abs(len(clip) - 550) <= 10, len(clip)
    finally:
        edge_module.edge_tts.Communicate = original
    assert calls == [
        {
            "text": "Humph, I suppose.",
            "voice": "en-GB-MaisieNeural",
            "boundary": "WordBoundary",
        }
    ], calls


def test_prefix_audio_without_word_timings_fails_softly():
    original = edge_module.edge_tts.Communicate
    edge_module.edge_tts.Communicate = fake_communicate([], [])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            engine = edge_module.TTSEngine(voice="en-GB-MaisieNeural")
            engine.new_audio_dir = tmp
            assert engine.generate_prefix_audio("Humph,", "I suppose.", "x") is None
    finally:
        edge_module.edge_tts.Communicate = original


if __name__ == "__main__":
    run_module(globals())
