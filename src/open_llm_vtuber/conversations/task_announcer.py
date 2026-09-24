"""Starts a turn to announce finished background tasks once the conversation is quiet."""

import asyncio
import inspect
import time
from typing import Awaitable, Callable, Optional, Union

from loguru import logger


class TaskAnnouncer:
    """One per client. `notify()` when results are waiting; the announcer waits until no
    turn is running and none has run for `quiet_gap_s`, then calls `start_turn()` once."""

    def __init__(
        self,
        is_busy: Callable[[], bool],
        start_turn: Callable[[], Union[None, Awaitable[None]]],
        quiet_gap_s: float = 1.0,
        poll_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._is_busy = is_busy
        self._start_turn = start_turn
        self.quiet_gap_s = quiet_gap_s
        self.poll_s = poll_s
        self._clock = clock
        self._waiter: Optional[asyncio.Task] = None

    def notify(self) -> None:
        if self._waiter is not None and not self._waiter.done():
            return
        self._waiter = asyncio.get_running_loop().create_task(
            self._announce_when_quiet()
        )

    def close(self) -> None:
        if self._waiter is not None:
            self._waiter.cancel()
            self._waiter = None

    async def _announce_when_quiet(self) -> None:
        quiet_since: Optional[float] = None
        while True:
            if self._is_busy():
                quiet_since = None
            else:
                now = self._clock()
                if quiet_since is None:
                    quiet_since = now
                if now - quiet_since >= self.quiet_gap_s:
                    break
            await asyncio.sleep(self.poll_s)
        try:
            result = self._start_turn()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Could not start the task-result turn")
