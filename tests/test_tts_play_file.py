"""Already-rendered audio (e.g. pre-made acks) is sent in order and awaited like TTS.

Run directly:  .venv/Scripts/python.exe tests/test_tts_play_file.py
"""

import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydub.generators import Sine  # noqa: E402

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.agent.output_types import Actions, AudioOutput, DisplayText  # noqa: E402
from src.open_llm_vtuber.conversations.conversation_utils import process_agent_output  # noqa: E402
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager  # noqa: E402


def tone(path: str) -> str:
    Sine(440).to_audio_segment(duration=200).export(path, format="wav")
    return path


async def test_play_file_reads_before_returning():
    with tempfile.TemporaryDirectory() as tmp:
        path = tone(os.path.join(tmp, "ack.wav"))
        sent = []

        async def send(message):
            sent.append(json.loads(message))

        manager = TTSTaskManager()
        await manager.play_file(path, DisplayText(text="On it."), Actions(), send)
        os.remove(path)  # the caller may delete the file as soon as play_file returns
        assert manager.task_list, (
            "finalize_conversation_turn waits only when task_list is non-empty"
        )
        await asyncio.gather(*manager.task_list)
        await asyncio.sleep(0.05)
        manager.clear()
    assert len(sent) == 1 and sent[0]["type"] == "audio", sent
    assert sent[0]["display_text"]["text"] == "On it." and sent[0]["audio"]


async def test_audio_output_goes_through_the_tts_manager():
    with tempfile.TemporaryDirectory() as tmp:
        path = tone(os.path.join(tmp, "ack.wav"))
        sent = []

        async def send(message):
            sent.append(json.loads(message))

        manager = TTSTaskManager()
        output = AudioOutput(
            audio_path=path,
            display_text=DisplayText(text="Fine, on it."),
            transcript="Fine, on it.",
            actions=Actions(),
        )
        text = await process_agent_output(
            output=output,
            character_config=SimpleNamespace(character_name="Yuna", avatar="yuna.png"),
            live2d_model=None,
            tts_engine=None,
            websocket_send=send,
            tts_manager=manager,
        )
        assert text == "Fine, on it."
        assert manager.task_list
        await asyncio.gather(*manager.task_list)
        await asyncio.sleep(0.05)
        manager.clear()
    assert sent and sent[0]["display_text"]["name"] == "Yuna", sent


async def test_unreadable_file_sends_a_silent_payload():
    sent = []

    async def send(message):
        sent.append(json.loads(message))

    manager = TTSTaskManager()
    await manager.play_file("Z:/missing.wav", DisplayText(text="On it."), None, send)
    await asyncio.gather(*manager.task_list)
    await asyncio.sleep(0.05)
    manager.clear()
    assert (
        sent
        and sent[0]["audio"] is None
        and sent[0]["display_text"]["text"] == "On it."
    )


if __name__ == "__main__":
    run_module(globals())
