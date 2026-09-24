"""Building an audio payload (decode + base64 + RMS) must not stall the event loop.

Run directly:  .venv/Scripts/python.exe tests/test_tts_manager_offloop.py
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.agent.output_types import Actions, DisplayText  # noqa: E402
from src.open_llm_vtuber.conversations import tts_manager as tts_manager_module  # noqa: E402
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager  # noqa: E402

PAYLOAD_BUILD_S = 0.3


class FakeTTS:
    async def async_generate_audio(self, text, file_name_no_ext=None):
        return "fake.wav"

    def remove_file(self, path):
        pass


def slow_payload(audio_path=None, display_text=None, actions=None, **_):
    time.sleep(PAYLOAD_BUILD_S)  # stands in for pydub/ffmpeg work
    return {"type": "audio", "audio": "x", "display_text": None, "actions": None}


async def test_payload_build_does_not_block_other_coroutines():
    original = tts_manager_module.prepare_audio_payload
    tts_manager_module.prepare_audio_payload = slow_payload
    try:
        sent = []

        async def send(message):
            sent.append(json.loads(message))

        ticks = []

        async def heartbeat():
            while True:
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.02)

        beat = asyncio.create_task(heartbeat())
        manager = TTSTaskManager()
        await manager.speak(
            "Hello there.",
            DisplayText(text="Hello there."),
            Actions(),
            None,
            FakeTTS(),
            send,
        )
        await asyncio.gather(*manager.task_list)
        await asyncio.sleep(0.05)
        beat.cancel()
        manager.clear()
    finally:
        tts_manager_module.prepare_audio_payload = original

    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert gaps and max(gaps) < PAYLOAD_BUILD_S / 2, (
        f"event loop stalled for {max(gaps):.2f}s while a payload was built"
    )
    assert len(sent) == 1 and sent[0]["audio"] == "x", sent


if __name__ == "__main__":
    run_module(globals())
