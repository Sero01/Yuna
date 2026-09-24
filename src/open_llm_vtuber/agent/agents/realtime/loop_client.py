"""httpx client bound to the running event loop, plus connection warm-up pings.

The agent is built during server start-up (one event loop) and used by connection
handlers (another), so clients can't be created up front.
"""

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx
from loguru import logger


class LoopBoundClient:
    """Hands out one AsyncClient per running event loop."""

    def __init__(self, **client_kwargs: Any):
        self._kwargs = client_kwargs
        self._client: Optional[httpx.AsyncClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def get(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop or self._client.is_closed:
            self._client = httpx.AsyncClient(**self._kwargs)
            self._loop = loop
        return self._client


def warm_up_pings(
    http: LoopBoundClient, url: str, headers: Dict[str, str], connections: int = 2
) -> Callable[[], Awaitable[None]]:
    """Coroutine function that opens or refreshes `connections` pooled connections.

    Jev and the talker run concurrently, so a turn needs two warm connections to avoid
    paying a TLS handshake on the critical path.
    """

    async def warm() -> None:
        client = http.get()

        async def ping() -> None:
            try:
                await client.get(url, headers=headers, timeout=10.0)
            except httpx.HTTPError as e:
                logger.debug(f"Warm-up ping failed: {e}")

        await asyncio.gather(*(ping() for _ in range(connections)))

    return warm
