"""Openers: play the reply's first interjection ("Hmph,", "Huh?") from a pre-rendered clip.

The talker is asked to start most replies with one of a fixed set of interjections. When
it does, that word is sent as a cached clip the moment it streams in, while the rest of
the sentence is still being synthesized. Unlike a blind filler, the talker picked it for
this reply, so it fits and is never doubled.

Clips are cut out of a carrier sentence ("Hmph, I suppose that makes sense.") when the
TTS engine reports word timings, so they keep a mid-sentence pitch instead of the
falling pitch of a one-word sentence.
"""

import asyncio
import hashlib
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Collection, Dict, List, Optional, Set, Tuple

from loguru import logger
from pydub import AudioSegment

from ....utils.audio_cut import cut_after_words  # noqa: F401  (re-exported)
from ....utils.stream_audio import preload_static_clip
from ....utils.tts_preprocessor import fix_pronunciation

DEFAULT_CACHE_DIR = os.path.join("cache", "openers")
DEFAULT_CARRIER = "I suppose that makes sense."

_MAX_WORD = 12

_LEADING_TAGS = re.compile(r"\s*((?:\[[^\[\]]{1,30}\]\s*)*)")
_OPEN_TAG = re.compile(r"\[[^\[\]]{0,30}")
_WORD_PUNCT = re.compile(r"([A-Za-z][A-Za-z'\-]*)([?!.,…]+)")
_PARTIAL_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

Render = Callable[[str], Awaitable[Optional[str]]]


class _Undecided:
    def __repr__(self) -> str:
        return "UNDECIDED"


UNDECIDED = _Undecided()


@dataclass
class OpenerMatch:
    key: str  # normalized phrase, e.g. "hmph," or "huh?"
    text: str  # as the talker wrote it, e.g. "Hmph!"
    tags: str  # expression tags before it, e.g. "[joy] "
    rest: str  # the text after it


def opener_key(phrase: str) -> str:
    """'Hmph!' -> 'hmph,'; question marks keep their own clip ('Huh?' -> 'huh?')."""
    m = _WORD_PUNCT.match(phrase.strip())
    word, punct = (m.group(1), m.group(2)) if m else (phrase.strip(" ,.!?"), "")
    return word.lower() + ("?" if "?" in punct else ",")


def opener_instruction(phrases: List[str]) -> str:
    return (
        "How you start speaking: begin most replies with a short interjection that fits "
        "the moment, followed by a comma or question mark, chosen from: "
        + " ".join(phrases)
        + ". An expression keyword may come before it but never replaces it: "
        '"[anger] Hmph, you ..." rather than "[anger] You ...". '
        "Vary them, and skip it when it would sound forced."
    )


def opener_reminder(phrases: List[str]) -> str:
    """Repeated after the user's message: from the system prompt alone the talker skipped
    the opener on ~40% of replies, mostly ones it began with an expression keyword."""
    return (
        "Start your reply with one of these interjections, after any expression "
        "keyword: " + " ".join(phrases) + " (whichever fits best, but not the one your "
        "last reply opened with; leave it out only if it would sound wrong)."
    )


def match_opener(buffer: str, keys: Collection[str], final: bool = False):
    """Does the reply start with a known opener? Returns an OpenerMatch, None (no), or
    UNDECIDED (need more text; never returned when `final`)."""
    lead = _LEADING_TAGS.match(buffer)
    tags, after = lead.group(1), buffer[lead.end() :]
    if not final:
        if _OPEN_TAG.fullmatch(after):
            return UNDECIDED  # an expression tag still streaming in
        if len(after) <= _MAX_WORD and (not after or _PARTIAL_WORD.fullmatch(after)):
            return UNDECIDED  # the first word may not be finished
    m = _WORD_PUNCT.match(after)
    if not m:
        return None
    if m.end() == len(after) and not final:
        return UNDECIDED  # more punctuation may follow ("Hmph" "." "..")
    key = opener_key(m.group(0))
    if key not in keys:
        return None
    return OpenerMatch(key=key, text=m.group(0), tags=tags, rest=after[m.end() :])


class OpenerLibrary:
    """Rendered opener clips for one voice, cached on disk by (voice, phrase)."""

    def __init__(
        self,
        phrases: List[str],
        render: Optional[Render],
        cache_dir: str = DEFAULT_CACHE_DIR,
        cache_key: str = "",
        carrier: str = DEFAULT_CARRIER,
    ):
        self.phrases = list(phrases)
        self.render = render
        self.cache_dir = cache_dir
        self.cache_key = cache_key
        self.carrier = carrier
        self._clips: Dict[str, str] = {}
        self._built = False
        self._build_task: Optional[asyncio.Task] = None

    def keys(self) -> Set[str]:
        return set(self._clips)

    def clip(self, key: str) -> Optional[str]:
        return self._clips.get(key)

    def use(self, render: Optional[Render], cache_key: str) -> None:
        """Render with a (new) voice; the build runs in the background."""
        if self._build_task is not None:
            try:
                self._build_task.cancel()
            except RuntimeError:  # its event loop is already closed
                pass
            self._build_task = None
        self.render = render
        self.cache_key = cache_key
        self._clips = {}
        self._built = False
        self.ensure_built()

    def ensure_built(self) -> None:
        """Start (or restart) the build in the running loop unless it's done or running.
        The server initializes in one event loop and serves in another, so a build
        started at init may have been cancelled with its loop."""
        if self.render is None or self._built:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet; the first turn starts it
        task = self._build_task
        if task is not None and not task.done() and task.get_loop() is loop:
            return
        self._build_task = loop.create_task(self.build())

    async def build(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        for phrase in self.phrases:
            key = opener_key(phrase)
            path = self._path(key)
            if not os.path.exists(path):
                try:
                    rendered = await self.render(phrase) if self.render else None
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"Opener '{phrase}' could not be rendered: {e}")
                    continue
                if not rendered or not os.path.exists(rendered):
                    continue
                partial = path + ".partial"
                shutil.move(rendered, partial)
                os.replace(partial, path)
            try:
                # Trimmed: the voices start ~120 ms into their clips.
                await asyncio.to_thread(preload_static_clip, path, True)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # it still plays, just decoded on each use
                logger.warning(f"Opener '{phrase}' could not be preloaded: {e}")
            self._clips[key] = path
        self._built = True
        logger.info(f"Opener clips ready: {len(self._clips)}/{len(self.phrases)}")

    def _path(self, key: str) -> str:
        digest = hashlib.sha1(f"{self.cache_key}\0{key}".encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, digest[:16] + ".wav")


def engine_renderer(engine, carrier: str = DEFAULT_CARRIER) -> Tuple[Render, str]:
    """A render function for `engine`, and a cache key for its voice."""
    cache_key = f"{type(engine).__module__}:{getattr(engine, 'voice', '')}:{carrier}"
    prefix_audio = getattr(engine, "generate_prefix_audio", None)

    async def render(phrase: str) -> Optional[str]:
        name = f"opener_{uuid.uuid4().hex[:8]}"
        spoken = fix_pronunciation(phrase)
        if prefix_audio is not None:
            path = await asyncio.to_thread(
                prefix_audio, spoken, fix_pronunciation(carrier), name
            )
            if path:
                return path
        # No word timings: say the opener on its own (falling pitch, but it works).
        path = await engine.async_generate_audio(spoken, file_name_no_ext=name)
        if not path:
            return None
        return await asyncio.to_thread(_trimmed_wav, path)

    return render, cache_key


def _trimmed_wav(path: str) -> str:
    from ....conversations.filler import trim_silence

    target = os.path.splitext(path)[0] + "_trim.wav"
    trim_silence(AudioSegment.from_file(path)).export(target, format="wav")
    if os.path.abspath(target) != os.path.abspath(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return target
