"""Regression test: concurrent writes to one WebSocket must not corrupt the drain state.

Background: websockets.legacy asserts in _drain_helper that no other coroutine is
already awaiting a drain. Two tasks writing the same socket (the TTS sender task and
the conversation coroutine) tripped that assertion once the transport backpressured,
killing the turn with an empty-message "Error in conversation chain".

Run directly:  uv run python tests/test_ws_concurrent_send.py
"""

import asyncio
import base64
import os
import sys

import uvicorn
from fastapi import FastAPI, WebSocket

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.open_llm_vtuber.utils.ws_send import SerializedWebSocket  # noqa: E402

PAYLOAD = "x" * 300_000  # comparable to a base64-encoded WAV payload
WRITES_PER_TASK = 40


def _build_app(errors: list, use_lock: bool) -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws")
    async def endpoint(websocket: WebSocket):
        await websocket.accept()
        # Mirrors routes.py: the socket is wrapped once, at accept time.
        if use_lock:
            websocket = SerializedWebSocket(websocket)
        send = websocket.send_text

        async def writer(tag: str):
            try:
                for i in range(WRITES_PER_TASK):
                    await send(f'{{"t":"{tag}{i}","d":"{PAYLOAD}"}}')
            except Exception as exc:  # noqa: BLE001 - we are classifying failures
                errors.append(type(exc).__name__)

        # Mirrors production: TTSTaskManager._sender_task and the conversation
        # coroutine both write this socket.
        await asyncio.gather(writer("A"), writer("B"), return_exceptions=True)

    return app


async def _deaf_client(port: int) -> None:
    """Complete the handshake, then stop reading so the server transport pauses."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    key = base64.b64encode(os.urandom(16)).decode()
    writer.write(
        f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
    )
    await writer.drain()
    await reader.readuntil(b"\r\n\r\n")
    await asyncio.sleep(8)
    writer.close()


async def _run(use_lock: bool, port: int) -> list:
    errors: list = []
    config = uvicorn.Config(
        _build_app(errors, use_lock), host="127.0.0.1", port=port, log_level="critical"
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1.5)
    try:
        await _deaf_client(port)
    except Exception:  # noqa: BLE001 - client teardown is not under test
        pass
    server.should_exit = True
    try:
        await asyncio.wait_for(serve_task, timeout=5)
    except asyncio.TimeoutError:
        pass
    return errors


def test_locked_send_survives_concurrent_writers():
    errors = asyncio.run(_run(use_lock=True, port=8812))
    assert "AssertionError" not in errors, (
        f"locked_send must serialize writes, but drain state was corrupted: {errors}"
    )


def test_raw_send_text_reproduces_the_bug():
    """Guards the premise: without the lock the assertion still fires."""
    errors = asyncio.run(_run(use_lock=False, port=8813))
    assert "AssertionError" in errors, (
        f"expected the unlocked pattern to trip the drain assertion, got: {errors}"
    )


if __name__ == "__main__":
    print("\nunlocked (expect the bug to reproduce):")
    raw = asyncio.run(_run(use_lock=False, port=8813))
    print(f"  errors={raw or 'none'}  -> {'REPRODUCED' if 'AssertionError' in raw else 'NOT reproduced'}")

    print("\nlocked (expect no assertion):")
    safe = asyncio.run(_run(use_lock=True, port=8812))
    print(f"  errors={safe or 'none'}  -> {'STILL BROKEN' if 'AssertionError' in safe else 'OK'}")

    ok = "AssertionError" in raw and "AssertionError" not in safe
    print(f"\n{'PASS' if ok else 'FAIL'}: lock eliminates the assertion\n")
    sys.exit(0 if ok else 1)


# --- passthrough: the wrapper must not break normal connection behaviour ---


async def _run_passthrough(port: int) -> dict:
    """Round-trip through the wrapper: receive, send_text, send_json."""
    result: dict = {}
    app = FastAPI()

    @app.websocket("/ws")
    async def endpoint(websocket: WebSocket):
        await websocket.accept()
        wrapped = SerializedWebSocket(websocket)
        # __getattr__ passthrough must expose receive_text and client_state.
        result["has_client_state"] = wrapped.client_state is not None
        incoming = await wrapped.receive_text()
        result["received"] = incoming
        await wrapped.send_text("text-ok")
        await wrapped.send_json({"kind": "json-ok"})

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1.5)

    import websockets

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as client:
            await client.send("hello-from-client")
            result["text_reply"] = await asyncio.wait_for(client.recv(), timeout=5)
            result["json_reply"] = await asyncio.wait_for(client.recv(), timeout=5)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"

    server.should_exit = True
    try:
        await asyncio.wait_for(serve_task, timeout=5)
    except asyncio.TimeoutError:
        pass
    return result


def test_wrapper_passes_through_receive_and_send():
    r = asyncio.run(_run_passthrough(8814))
    assert r.get("error") is None, r["error"]
    assert r.get("received") == "hello-from-client", r
    assert r.get("text_reply") == "text-ok", r
    assert '"json-ok"' in str(r.get("json_reply")), r
    assert r.get("has_client_state") is True, r
