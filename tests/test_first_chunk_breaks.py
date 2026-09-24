"""The fast first chunk (faster_first_response) ends at the first clause break, so the
first TTS request doesn't wait for a whole sentence.

Run directly:  .venv/Scripts/python.exe tests/test_first_chunk_breaks.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.agent.output_types import DisplayText  # noqa: E402
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager  # noqa: E402
from src.open_llm_vtuber.utils.sentence_divider import SentenceDivider  # noqa: E402


async def divide(tokens):
    """All chunks, plus how many tokens had arrived when the first chunk came out."""
    consumed = 0

    async def stream():
        nonlocal consumed
        for token in tokens:
            consumed += 1
            yield token

    divider = SentenceDivider(faster_first_response=True, valid_tags=["think"])
    chunks, consumed_at_first = [], None
    async for sentence in divider.process_stream(stream()):
        if consumed_at_first is None:
            consumed_at_first = consumed
        chunks.append(sentence.text)
    return chunks, consumed_at_first


async def test_first_chunk_ends_at_a_comma():
    chunks, consumed = await divide(["Mostly", ",", " yeah", "."])
    assert chunks[0] == "Mostly,", chunks
    assert consumed == 2


async def test_first_chunk_ends_at_an_em_dash():
    chunks, consumed = await divide(["Sure", " —", " give", " me", " a", " sec", "."])
    assert chunks[0] == "Sure —", chunks
    assert consumed == 2


async def test_first_chunk_ends_at_a_colon_followed_by_a_space():
    chunks, consumed = await divide(["Here's the thing", ":", " it", " works", "."])
    assert chunks[0] == "Here's the thing:", chunks
    assert consumed == 3


async def test_a_colon_inside_a_time_is_not_a_break():
    chunks, _ = await divide(["It's", " 3", ":", "30", " now", "."])
    assert chunks[0] == "It's 3:30 now.", chunks


async def test_a_comma_inside_a_number_is_not_a_break():
    chunks, _ = await divide(["It", " costs", " 1", ",", "000", " dollars", "."])
    assert chunks[0] == "It costs 1,000 dollars.", chunks


async def test_a_comma_after_a_number_is_a_break_when_a_space_follows():
    chunks, _ = await divide(["In", " 2024", ",", " we", " won", "."])
    assert chunks[0] == "In 2024,", chunks


async def test_first_chunk_ends_at_the_earliest_break():
    chunks, _ = await divide(["Well; honestly, no."])
    assert chunks[0] == "Well;", chunks


class RecordingTTS:
    def __init__(self):
        self.texts = []

    async def async_generate_audio(self, text, file_name_no_ext=None):
        self.texts.append(text)
        return None

    def remove_file(self, filepath, verbose=True):
        pass


async def test_a_chunk_of_only_a_dash_is_not_synthesized():
    # "[smirk] —" reaches TTS as " —": brackets are filtered, the dash is kept.
    sent = []

    async def send(message):
        sent.append(json.loads(message))

    tts = RecordingTTS()
    manager = TTSTaskManager()
    await manager.speak(" —", DisplayText(text="[smirk] —"), None, None, tts, send)
    await asyncio.gather(*manager.task_list)
    await asyncio.sleep(0.05)
    manager.clear()
    assert tts.texts == [], tts.texts
    assert sent and sent[0]["audio"] is None, sent


if __name__ == "__main__":
    run_module(globals())
