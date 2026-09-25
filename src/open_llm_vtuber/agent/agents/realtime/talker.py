"""Streaming talker (OpenRouter chat completions) with provider racing or hedging.

With a race provider set, every reply goes to the pinned provider and the race provider
at once and the first to stream a token is spoken. Otherwise a backup provider is only
asked once the primary has been silent for `hedge_after_s`.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx
from loguru import logger

_END = object()


class TalkerError(Exception):
    """The talker could not produce (the rest of) a reply."""


class _Attempt:
    """One streaming request. Buffers its pieces; `settled` is set on the first piece,
    the end of the stream, or an error. `failed` is set only for errors before any piece."""

    def __init__(self, talker: "Talker", body: Dict[str, Any], label: str):
        self.label = label
        self.queue: asyncio.Queue = asyncio.Queue()
        self.settled = asyncio.Event()
        self.failed: Optional[BaseException] = None
        self.task = asyncio.create_task(self._run(talker, body))

    async def _run(self, talker: "Talker", body: Dict[str, Any]) -> None:
        got_piece = False
        try:
            async for piece in talker._sse_pieces(body):
                got_piece = True
                self.queue.put_nowait(piece)
                self.settled.set()
            self.queue.put_nowait(_END)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not got_piece:
                self.failed = e
            self.queue.put_nowait(TalkerError(f"{self.label}: {type(e).__name__}: {e}"))
        finally:
            self.settled.set()

    async def pieces(self) -> AsyncIterator[str]:
        while True:
            item = await self.queue.get()
            if item is _END:
                return
            if isinstance(item, TalkerError):
                raise item
            yield item

    def cancel(self) -> None:
        self.task.cancel()


class Talker:
    def __init__(
        self,
        get_client: Callable[[], httpx.AsyncClient],
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "",
        temperature: float = 0.8,
        max_tokens: int = 200,
        hedge_after_s: float = 1.5,
        race_provider: str = "",
    ):
        self._get_client = get_client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.hedge_after_s = hedge_after_s
        self.race_provider = race_provider

    def body(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        backup: bool = False,
        race: bool = False,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "reasoning": {"enabled": False},
        }
        if race:
            # No fallbacks: a slow stand-in provider can't win the race, it only costs.
            body["provider"] = {"order": [self.race_provider], "allow_fallbacks": False}
        elif backup:
            # Prompt caching is per provider, so the backup deliberately goes elsewhere.
            body["provider"] = (
                {"ignore": [self.provider], "sort": "latency"}
                if self.provider
                else {"sort": "throughput"}
            )
        else:
            body["provider"] = (
                {"order": [self.provider], "allow_fallbacks": True}
                if self.provider
                else {"sort": "latency"}
            )
        return body

    async def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        hedge: bool = True,
    ) -> AsyncIterator[str]:
        started = time.perf_counter()
        attempts = [_Attempt(self, self.body(messages, max_tokens), "primary")]
        racing = hedge and bool(self.race_provider)
        if racing:
            attempts.append(
                _Attempt(self, self.body(messages, max_tokens, race=True), "race")
            )
        try:
            if racing:
                winner = await self._first_success(attempts)
            elif hedge:
                winner = await self._choose(attempts, messages, max_tokens)
            else:
                winner = attempts[0]
            for attempt in attempts:
                if attempt is not winner:
                    attempt.cancel()
            first = True
            async for piece in winner.pieces():
                if first:
                    first = False
                    logger.info(
                        f"Talker: first token after {time.perf_counter() - started:.2f}s "
                        f"({winner.label})"
                    )
                yield piece
        finally:
            for attempt in attempts:
                attempt.cancel()

    async def complete(
        self, messages: List[Dict[str, str]], max_tokens: int = 60
    ) -> str:
        # Only used off the critical path (acks), where a backup request is pure cost.
        pieces = self.stream(messages, max_tokens, hedge=False)
        return "".join([piece async for piece in pieces])

    async def _choose(
        self,
        attempts: List[_Attempt],
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
    ) -> _Attempt:
        primary = attempts[0]
        if self.hedge_after_s <= 0:
            return primary  # errors surface from pieces()
        try:
            await asyncio.wait_for(primary.settled.wait(), self.hedge_after_s)
        except asyncio.TimeoutError:
            logger.info(
                f"Talker: no token after {self.hedge_after_s}s, hedging with a backup provider"
            )
        if primary.settled.is_set() and primary.failed is None:
            return primary
        backup = _Attempt(self, self.body(messages, max_tokens, backup=True), "backup")
        attempts.append(backup)
        winner = await self._first_success([primary, backup])
        logger.info(f"Talker: hedge won by {winner.label}")
        return winner

    async def _first_success(self, attempts: List[_Attempt]) -> _Attempt:
        """The first attempt to stream a piece (or end cleanly); TalkerError if all fail."""
        pending = [a for a in attempts if not (a.settled.is_set() and a.failed)]
        while pending:
            waiters = {asyncio.ensure_future(a.settled.wait()): a for a in pending}
            done, not_done = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED
            )
            for waiter in not_done:
                waiter.cancel()
            for waiter in done:
                attempt = waiters[waiter]
                if attempt.failed is None:
                    return attempt
                pending.remove(attempt)
        failures = " ".join(f"{a.label}={a.failed!r}" for a in attempts)
        raise TalkerError(f"talker failed: {failures}")

    async def _sse_pieces(self, body: Dict[str, Any]) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "Open-LLM-VTuber",
        }
        async with self._get_client().stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=httpx.Timeout(30.0, connect=10.0),
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread())[:300]
                raise TalkerError(f"HTTP {response.status_code}: {detail!r}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise TalkerError(str(chunk["error"])[:300])
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
