"""Test doubles shared by the real-time agent tests (no network, no sockets)."""

import asyncio
import json
from typing import Callable, Dict, List, Optional

import httpx


def sse_chunks(*pieces: str, done: bool = True) -> List[bytes]:
    """OpenAI-style streaming chunks, one content piece each."""
    chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n".encode()
        for p in pieces
    ]
    if done:
        chunks.append(b"data: [DONE]\n\n")
    return chunks


class ChunkStream(httpx.AsyncByteStream):
    """Response body yielding chunks after optional delays; records how it ended."""

    def __init__(self, chunks, first_delay: float = 0.0, delay: float = 0.0):
        self.chunks = list(chunks)
        self.first_delay = first_delay
        self.delay = delay
        self.finished = False
        self.closed = False

    async def __aiter__(self):
        await asyncio.sleep(self.first_delay)
        for chunk in self.chunks:
            yield chunk
            if self.delay:
                await asyncio.sleep(self.delay)
        self.finished = True

    async def aclose(self):
        self.closed = True


class QueueStream(httpx.AsyncByteStream):
    """SSE body fed from a queue of event dicts; `None` ends the stream."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def __aiter__(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield f"data: {json.dumps(event)}\n\n".encode()

    async def aclose(self):
        pass


class FakeHermes:
    """Enough of hermes-agent's Runs API for the task manager."""

    def __init__(
        self,
        start_status: int = 202,
        steer_accepted: bool = True,
        run_status: Optional[dict] = None,
    ):
        self.start_status = start_status
        self.steer_accepted = steer_accepted
        self.run_status = run_status or {"status": "running"}
        self.approval_pending = True
        self.unreachable = False
        self.requests: List[
            tuple
        ] = []  # (method, path, json body or None, auth header)
        self.streams: Dict[str, asyncio.Queue] = {}
        self._runs = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append(
            (request.method, path, body, request.headers.get("authorization"))
        )
        if self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if request.method == "POST" and path == "/v1/runs":
            if self.start_status >= 300:
                return httpx.Response(
                    self.start_status, json={"error": {"message": "nope"}}
                )
            self._runs += 1
            run_id = f"run_{self._runs}"
            self.streams[run_id] = asyncio.Queue()
            return httpx.Response(202, json={"run_id": run_id, "status": "started"})
        parts = path.strip("/").split("/")  # v1, runs, <id>, [action]
        run_id = parts[2] if len(parts) > 2 else ""
        action = parts[3] if len(parts) > 3 else ""
        if action == "events":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=QueueStream(self.streams[run_id]),
            )
        if action == "stop":
            return httpx.Response(200, json={"run_id": run_id, "status": "stopping"})
        if action == "steer":
            code = 200 if self.steer_accepted else 409
            return httpx.Response(code, json={"accepted": self.steer_accepted})
        if action == "approval":
            if not self.approval_pending:
                return httpx.Response(
                    409, json={"error": {"code": "approval_not_pending"}}
                )
            self.approval_pending = False
            return httpx.Response(200, json={"choice": body["choice"], "resolved": 1})
        if request.method == "GET" and not action:
            return httpx.Response(200, json={"run_id": run_id, **self.run_status})
        return httpx.Response(404, json={"error": "not found"})

    def emit(self, run_id: str, **event) -> None:
        self.streams[run_id].put_nowait(event)

    def end_stream(self, run_id: str) -> None:
        self.streams[run_id].put_nowait(None)

    def calls(self, method: str, suffix: str) -> List[tuple]:
        return [r for r in self.requests if r[0] == method and r[1].endswith(suffix)]


class FakeLive2D:
    emo_map = {"joy": 3, "sadness": 1}

    def extract_emotion(self, text: str) -> list:
        lower = text.lower()
        return [v for k, v in self.emo_map.items() if f"[{k}]" in lower]


def template_tts_preprocessor_config():
    """The real TTS preprocessor settings from the default config template."""
    import os

    from src.open_llm_vtuber.config_manager.utils import read_yaml, validate_config

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = validate_config(
        read_yaml(os.path.join(root, "config_templates", "conf.default.yaml"))
    )
    return config.character_config.tts_preprocessor_config


async def wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)
