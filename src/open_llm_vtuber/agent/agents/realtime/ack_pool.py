"""'On it' acknowledgements for new tasks, prepared ahead of time (text and audio)."""

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from loguru import logger

_TAGS = re.compile(r"\[[^\]]*\]")

GenerateText = Callable[[], Awaitable[str]]
Synthesize = Callable[[str], Awaitable[Optional[str]]]


def clean_ack(text: Optional[str]) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = _TAGS.sub("", text.splitlines()[0])
    text = re.sub(r"\s+", " ", text).strip().strip("\"'").strip()
    return text[:150].strip()


@dataclass
class Ack:
    text: str
    audio_path: Optional[str] = None


def discard_audio(ack: Ack) -> None:
    if ack.audio_path:
        try:
            os.remove(ack.audio_path)
        except OSError:
            pass


class AckPool:
    """Keeps up to `size` acknowledgements ready; refills in the background."""

    def __init__(
        self,
        generate_text: GenerateText,
        synthesize: Optional[Synthesize] = None,
        size: int = 2,
    ):
        self._generate_text = generate_text
        self._synthesize = synthesize
        self.size = size
        self.ready: List[Ack] = []
        self._task: Optional[asyncio.Task] = None

    def take(self) -> Optional[Ack]:
        ack = self.ready.pop(0) if self.ready else None
        self.fill()
        return ack

    def fill(self) -> None:
        """Start topping the pool up in the running loop (no-op if already doing so)."""
        loop = asyncio.get_running_loop()
        if (
            self._task is not None
            and not self._task.done()
            and self._task.get_loop() is loop
        ):
            return
        self._task = loop.create_task(self._refill())

    def set_synthesizer(self, synthesize: Optional[Synthesize]) -> None:
        self._synthesize = synthesize
        self.clear()  # drop acks rendered with the previous voice

    def clear(self) -> None:
        if self._task is not None:
            try:
                self._task.cancel()
            except RuntimeError:  # its event loop is already closed
                pass
            self._task = None
        for ack in self.ready:
            discard_audio(ack)
        self.ready = []

    async def _refill(self) -> None:
        while len(self.ready) < self.size:
            try:
                text = clean_ack(await self._generate_text())
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Could not generate a task acknowledgement: {e}")
                return
            if not text:
                return
            audio_path = None
            if self._synthesize is not None:
                try:
                    audio_path = await self._synthesize(text)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"Could not pre-render acknowledgement audio: {e}")
            self.ready.append(Ack(text, audio_path))
