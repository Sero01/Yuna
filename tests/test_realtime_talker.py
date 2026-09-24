"""Talker: SSE parsing, provider pinning, hedging, errors.

Run directly:  .venv/Scripts/python.exe tests/test_realtime_talker.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from _harness import run_module  # noqa: E402
from realtime_fakes import ChunkStream, sse_chunks  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.loop_client import LoopBoundClient  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.talker import Talker, TalkerError  # noqa: E402

MESSAGES = [
    {"role": "system", "content": "You are Yuna."},
    {"role": "user", "content": "hi"},
]


def is_backup(body):
    provider = body.get("provider") or {}
    return "ignore" in provider or provider.get("sort") == "throughput"


class FakeProviders:
    """Serves primary and backup requests; each spec is (status, chunks, first_delay)."""

    def __init__(self, primary, backup=(200, sse_chunks("backup"), 0.0)):
        self.primary = primary
        self.backup = backup
        self.bodies = []
        self.streams = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        status, chunks, first_delay = self.backup if is_backup(body) else self.primary
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "overloaded"}})
        stream = ChunkStream(chunks, first_delay=first_delay)
        self.streams.append(stream)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        )


def make_talker(fake, provider="", hedge=0.9):
    get = LoopBoundClient(transport=httpx.MockTransport(fake.handler)).get
    return Talker(
        get,
        "https://openrouter.ai/api/v1",
        "or-key",
        "deepseek/deepseek-v4.1-flash",
        provider=provider,
        hedge_after_s=hedge,
    )


async def collect(talker, **kw):
    return [piece async for piece in talker.stream(MESSAGES, **kw)]


async def test_streams_pieces_with_reasoning_off():
    fake = FakeProviders((200, sse_chunks("Hmph", ", hi."), 0.0))
    pieces = await collect(make_talker(fake))
    assert pieces == ["Hmph", ", hi."], pieces
    body = fake.bodies[0]
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["reasoning"] == {"enabled": False}
    assert body["stream"] is True and body["max_tokens"] == 200
    assert body["provider"] == {"sort": "latency"}
    assert body["messages"] == MESSAGES
    assert len(fake.bodies) == 1


async def test_max_tokens_override():
    fake = FakeProviders((200, sse_chunks("ok"), 0.0))
    await collect(make_talker(fake), max_tokens=40)
    assert fake.bodies[0]["max_tokens"] == 40


async def test_slow_primary_is_hedged_and_loses():
    fake = FakeProviders((200, sse_chunks("slow"), 0.5), (200, sse_chunks("fast"), 0.0))
    started = asyncio.get_running_loop().time()
    pieces = await collect(make_talker(fake, provider="makora", hedge=0.05))
    assert pieces == ["fast"], pieces
    assert asyncio.get_running_loop().time() - started < 0.4
    assert fake.bodies[0]["provider"] == {"order": ["makora"], "allow_fallbacks": True}
    assert fake.bodies[1]["provider"] == {"ignore": ["makora"], "sort": "latency"}
    await asyncio.sleep(0.05)
    assert not fake.streams[0].finished, "the losing primary stream is abandoned"


async def test_fast_primary_does_not_hedge():
    fake = FakeProviders((200, sse_chunks("hi"), 0.0))
    pieces = await collect(make_talker(fake, hedge=0.2))
    await asyncio.sleep(0.3)
    assert pieces == ["hi"] and len(fake.bodies) == 1, fake.bodies


async def test_failed_primary_switches_to_backup_immediately():
    fake = FakeProviders((500, [], 0.0), (200, sse_chunks("backup"), 0.0))
    started = asyncio.get_running_loop().time()
    pieces = await collect(make_talker(fake, hedge=5.0))
    assert pieces == ["backup"], pieces
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_both_failing_raises():
    fake = FakeProviders((500, [], 0.0), (502, [], 0.0))
    try:
        await collect(make_talker(fake, hedge=0.05))
    except TalkerError:
        return
    raise AssertionError("expected TalkerError")


async def test_without_hedging_errors_surface():
    fake = FakeProviders((500, [], 0.0))
    try:
        await collect(make_talker(fake, hedge=0))
    except TalkerError:
        assert len(fake.bodies) == 1
        return
    raise AssertionError("expected TalkerError")


async def test_error_chunk_raises():
    chunks = [b'data: {"error": {"message": "rate limited"}}\n\n']
    fake = FakeProviders((200, chunks, 0.0))
    try:
        await collect(make_talker(fake, hedge=0))
    except TalkerError as e:
        assert "rate limited" in str(e)
        return
    raise AssertionError("expected TalkerError")


async def test_ignores_comments_and_empty_deltas():
    chunks = [b": OPENROUTER PROCESSING\n\n", b'data: {"choices": [{"delta": {}}]}\n\n']
    chunks += sse_chunks("ok")
    fake = FakeProviders((200, chunks, 0.0))
    assert await collect(make_talker(fake)) == ["ok"]


async def test_complete_joins_the_stream():
    fake = FakeProviders((200, sse_chunks("Fine, ", "on it."), 0.0))
    assert await make_talker(fake).complete(MESSAGES) == "Fine, on it."
    assert fake.bodies[0]["max_tokens"] == 60


async def test_complete_never_hedges():
    # Acks are generated in the background; nobody waits on them, so a backup
    # request would only add cost.
    fake = FakeProviders((200, sse_chunks("on it."), 0.2))
    assert await make_talker(fake, hedge=0.05).complete(MESSAGES) == "on it."
    assert len(fake.bodies) == 1, fake.bodies


if __name__ == "__main__":
    run_module(globals())
