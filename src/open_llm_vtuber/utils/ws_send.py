"""Serialized WebSocket sending.

A conversation turn has several concurrent writers on one socket: the TTS sender
task draining its payload queue, the conversation coroutine emitting control and
tool-status messages, heartbeat replies, and group broadcasts. ``websockets.legacy``
does not support that - once the transport backpressures (large base64 audio
payloads do this readily), the second writer trips an assertion in ``_drain_helper``
and the turn dies with an unhelpful empty-message "Error in conversation chain".

Wrapping the socket once, at accept time, serializes every write without touching
the ~30 call sites that send on it.
"""

import asyncio
from typing import Any


class SerializedWebSocket:
    """Proxy around a Starlette ``WebSocket`` that serializes all sends.

    Send methods share one lock, so interleaved writes from different tasks are
    queued rather than corrupting the transport's drain state. Every other
    attribute (``receive_text``, ``client_state``, ``accept``, ...) passes straight
    through to the wrapped socket.
    """

    def __init__(self, websocket: Any) -> None:
        self._websocket = websocket
        self._send_lock = asyncio.Lock()

    async def send_text(self, data: str) -> None:
        async with self._send_lock:
            await self._websocket.send_text(data)

    async def send_json(self, data: Any, mode: str = "text") -> None:
        async with self._send_lock:
            await self._websocket.send_json(data, mode=mode)

    async def send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            await self._websocket.send_bytes(data)

    async def send(self, message: Any) -> None:
        async with self._send_lock:
            await self._websocket.send(message)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this proxy does not define itself.
        return getattr(self._websocket, name)


def locked_send(websocket: Any):
    """Return a serialized send callable for ``websocket``.

    Convenience for code that needs a bare ``Callable[[str], Awaitable[None]]``.
    When ``websocket`` is already a :class:`SerializedWebSocket`, its lock is reused.
    """
    if not isinstance(websocket, SerializedWebSocket):
        websocket = SerializedWebSocket(websocket)
    return websocket.send_text
