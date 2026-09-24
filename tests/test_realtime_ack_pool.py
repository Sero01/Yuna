"""Acknowledgement pool and the unknown-expression-tag filter.

Run directly:  .venv/Scripts/python.exe tests/test_realtime_ack_pool.py
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from realtime_fakes import FakeLive2D, wait_until  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.ack_pool import (  # noqa: E402
    Ack,
    AckPool,
    clean_ack,
)
from src.open_llm_vtuber.agent.agents.realtime.tags import (  # noqa: E402
    strip_unknown_tags,
    strip_unknown_tags_from,
)
from src.open_llm_vtuber.utils.sentence_divider import (  # noqa: E402
    SentenceWithTags,
    TagInfo,
    TagState,
)


def test_clean_ack():
    assert clean_ack('  "Ugh, fine. [soft] On it."  ') == "Ugh, fine. On it."
    assert clean_ack("Fine.\nSecond line") == "Fine."
    assert clean_ack("") == "" and clean_ack(None) == ""
    assert len(clean_ack("word " * 100)) <= 150


async def test_fill_generates_up_to_size_and_take_refills():
    texts = iter(["One.", "Two.", "Three."])

    async def generate():
        return next(texts)

    pool = AckPool(generate, size=2)
    pool.fill()
    await wait_until(lambda: len(pool.ready) == 2)
    assert [a.text for a in pool.ready] == ["One.", "Two."]
    assert pool.take().text == "One."
    await wait_until(lambda: len(pool.ready) == 2)
    assert [a.text for a in pool.ready] == ["Two.", "Three."]


async def test_synthesizer_prerenders_audio_and_clear_deletes_it():
    with tempfile.TemporaryDirectory() as tmp:

        async def generate():
            return "On it."

        async def synthesize(text):
            path = os.path.join(tmp, f"ack{len(os.listdir(tmp))}.wav")
            with open(path, "wb") as f:
                f.write(b"RIFF")
            return path

        pool = AckPool(generate, synthesize, size=2)
        pool.fill()
        await wait_until(lambda: len(pool.ready) == 2)
        paths = [a.audio_path for a in pool.ready]
        assert all(p and os.path.exists(p) for p in paths)
        pool.clear()
        assert pool.ready == [] and not any(os.path.exists(p) for p in paths)


async def test_failures_leave_the_pool_empty_without_raising():
    async def generate():
        raise RuntimeError("talker down")

    pool = AckPool(generate)
    pool.fill()
    await asyncio.sleep(0.05)
    assert pool.ready == [] and pool.take() is None

    async def generate_ok():
        return "On it."

    async def synthesize(text):
        raise RuntimeError("tts down")

    pool = AckPool(generate_ok, synthesize, size=1)
    pool.fill()
    await wait_until(lambda: len(pool.ready) == 1)
    assert pool.ready[0] == Ack("On it.", None), "text ack kept when TTS fails"


def test_strip_unknown_tags_from():
    known = {"joy", "sadness"}
    assert (
        strip_unknown_tags_from("Fine. [soft] I'll help. [joy]", known)
        == "Fine. I'll help. [joy]"
    )
    assert strip_unknown_tags_from("[JOY] yay", known) == "[JOY] yay"
    assert strip_unknown_tags_from("[sigh]", known).strip() == ""


async def test_strip_unknown_tags_decorator():
    @strip_unknown_tags(FakeLive2D())
    async def stream():
        yield SentenceWithTags(text="Hmph. [soft] ", tags=[])
        yield SentenceWithTags(text="[sigh]", tags=[])
        yield SentenceWithTags(
            text="thinking [x]", tags=[TagInfo("think", TagState.START)]
        )
        yield {"type": "tool_call_status"}
        yield SentenceWithTags(text="Yay [joy] ", tags=[])

    items = [item async for item in stream()]
    texts = [i.text for i in items if isinstance(i, SentenceWithTags)]
    assert texts == ["Hmph. ", "thinking [x]", "Yay [joy] "], texts
    assert {"type": "tool_call_status"} in items


if __name__ == "__main__":
    run_module(globals())
