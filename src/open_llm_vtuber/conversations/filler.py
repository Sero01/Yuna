"""Filler voices: pre-rendered backchannel clips played while a reply is generated.

Time to first token swings from ~1s to ~25s, so after the user stops speaking Yuna
plays a short neutral clip ("hmm") right away, and one "still thinking" clip if the
reply is slow. Clips are rendered once with the character's TTS engine, cached on
disk, and kept in memory as ready-to-send audio payloads, so sending one costs only
a ``json.dumps``.

Frontend constraints (the prebuilt frontend can't change):
- ``conversation-chain-start`` clears the frontend audio queue, so fillers are sent
  after the start signals, never before.
- Payloads carry ``display_text: None`` so they play (with lip-sync) without adding
  to the AI chat bubble or subtitle.

Fillers bypass TTSTaskManager, chat history and agent memory.
"""

import asyncio
import hashlib
import json
import os
import random
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from ..config_manager.filler import FillerConfig
from ..config_manager.tts import TTSConfig
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend

TIER0 = "tier0"
ESCALATION = "escalation"
TOOL = "tool"

DEFAULT_CACHE_DIR = os.path.join("cache", "fillers")

# edge-tts pads clips with ~0.5-1s of silence; leading silence is pure added latency.
SILENCE_THRESHOLD_DBFS = -45.0
KEEP_SILENCE_MS = 30


def trim_silence(audio: AudioSegment) -> AudioSegment:
    """Strip leading/trailing silence, keeping a few ms so the clip isn't clipped."""
    start = detect_leading_silence(audio, silence_threshold=SILENCE_THRESHOLD_DBFS)
    end = detect_leading_silence(
        audio.reverse(), silence_threshold=SILENCE_THRESHOLD_DBFS
    )
    if start + end >= len(audio):
        return audio  # all silence; leave it for the payload builder to reject
    return audio[
        max(0, start - KEEP_SILENCE_MS) : len(audio) - max(0, end - KEEP_SILENCE_MS)
    ]


@dataclass
class FillerClip:
    text: str
    payload: Dict[str, Any]
    duration_ms: int


class FillerLibrary:
    """Renders, caches and picks filler clips for one TTS voice."""

    def __init__(
        self,
        config: FillerConfig,
        tts_engine: TTSInterface,
        cache_key: str,
        cache_dir: str = DEFAULT_CACHE_DIR,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config
        self.tts_engine = tts_engine
        self.cache_key = cache_key
        self.cache_dir = cache_dir
        self.rng = rng or random.Random()
        self.clips: Dict[str, List[FillerClip]] = {TIER0: [], ESCALATION: [], TOOL: []}
        self.ready = False
        self._recent: Dict[str, Deque[str]] = {
            tier: deque(maxlen=config.no_repeat_window) for tier in self.clips
        }

    async def build(self) -> None:
        """Render missing clips and build their payloads. Never raises."""
        self.ready = False
        self.clips = {tier: [] for tier in self.clips}
        os.makedirs(self.cache_dir, exist_ok=True)
        tiers = (
            (TIER0, self.config.tier0_phrases),
            (ESCALATION, self.config.escalation_phrases),
            (TOOL, self.config.tool_phrases),
        )
        for tier, phrases in tiers:
            for text in phrases:
                clip = await self._load_clip(text)
                if not clip:
                    continue
                if tier == TIER0 and clip.duration_ms > self.config.max_tier0_ms:
                    logger.warning(
                        f"Filler '{text}' is {clip.duration_ms}ms, over max_tier0_ms "
                        f"({self.config.max_tier0_ms}ms); dropped."
                    )
                    continue
                self.clips[tier].append(clip)
        self.ready = True
        logger.info(
            "Filler clips ready: "
            + ", ".join(f"{tier}={len(clips)}" for tier, clips in self.clips.items())
        )

    async def _load_clip(self, text: str) -> Optional[FillerClip]:
        try:
            path = self._cached_path(text) or await self._render(text)
            if not path:
                logger.warning(f"Filler '{text}' could not be rendered; skipped.")
                return None
            # pydub decode + base64 + RMS is CPU work; keep it off the event loop.
            payload = await asyncio.to_thread(prepare_audio_payload, path)
            duration_ms = len(payload["volumes"]) * payload["slice_length"]
            return FillerClip(text=text, payload=payload, duration_ms=duration_ms)
        except asyncio.CancelledError:
            raise  # a cancelled build must not finish as a partial "ready" library
        except Exception as e:
            logger.warning(f"Filler '{text}' failed to load: {e}")
            return None

    def _stem(self, text: str) -> str:
        digest = hashlib.sha1(f"{self.cache_key}\0{text}".encode("utf-8"))
        return digest.hexdigest()[:16]

    def _cached_path(self, text: str) -> Optional[str]:
        path = os.path.join(self.cache_dir, self._stem(text) + ".wav")
        return path if os.path.exists(path) else None

    async def _render(self, text: str) -> Optional[str]:
        rendered = await self.tts_engine.async_generate_audio(
            text=text, file_name_no_ext=f"filler_{uuid.uuid4().hex[:8]}"
        )
        if not rendered or not os.path.exists(rendered):
            return None
        target = os.path.join(self.cache_dir, self._stem(text) + ".wav")
        try:
            await asyncio.to_thread(self._trim_to, rendered, target)
        finally:
            # Best-effort: if we were cancelled mid-trim the worker thread may still
            # hold the file open (Windows), and an OSError here would mask the cancel.
            try:
                os.remove(rendered)
            except OSError:
                pass
        return target

    @staticmethod
    def _trim_to(source: str, target: str) -> None:
        # Write to a temp name first so a crash can't leave a half-written cache hit.
        partial = target + ".partial"
        trim_silence(AudioSegment.from_file(source)).export(partial, format="wav")
        os.replace(partial, target)

    def pick(self, tier: str) -> Optional[FillerClip]:
        """Pick a random clip from ``tier``, avoiding recently played ones."""
        clips = self.clips.get(tier) or []
        if not clips:
            return None
        recent = self._recent[tier]
        candidates = [c for c in clips if c.text not in recent] or clips
        clip = self.rng.choice(candidates)
        recent.append(clip.text)
        return clip


class FillerController:
    """Plays fillers for one conversation turn.

    ``start()`` sends the instant clip (probability-gated) and arms one escalation.
    A running tool call plays the tool line instead of the generic escalation.
    ``stop()`` must be called as soon as the real reply starts, and on turn exit.
    """

    def __init__(
        self,
        library: Optional[FillerLibrary],
        config: Optional[FillerConfig],
        websocket_send: Optional[WebSocketSend],
        rng: Optional[random.Random] = None,
    ) -> None:
        self.library = library
        self.config = config
        self.websocket_send = websocket_send
        self.rng = rng or random.Random()
        self._stopped = library is None
        self._escalated = False
        self._timer: Optional[asyncio.Task] = None

    @classmethod
    def disabled(cls) -> "FillerController":
        return cls(library=None, config=None, websocket_send=None)

    async def start(self) -> None:
        if self._stopped or not self.library.ready:
            return
        if self.rng.random() < self.config.probability:
            await self._play(TIER0)
        self._timer = asyncio.create_task(self._escalate_after_delay())

    async def on_tool_status(self, event: Dict[str, Any]) -> None:
        if self._stopped or self._escalated or event.get("status") != "running":
            return
        if not self.library.ready:
            return
        self._escalated = True
        self._cancel_timer()
        await self._play(TOOL)

    def stop(self) -> None:
        self._stopped = True
        self._cancel_timer()

    def _cancel_timer(self) -> None:
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    async def _escalate_after_delay(self) -> None:
        await asyncio.sleep(self.config.escalation_delay)
        if self._stopped or self._escalated:
            return
        self._escalated = True
        # Detach before sending so stop() can't cancel us mid-write.
        self._timer = None
        await self._play(ESCALATION)

    async def _play(self, tier: str) -> None:
        clip = self.library.pick(tier)
        if not clip:
            return
        try:
            await self.websocket_send(json.dumps(clip.payload))
            logger.debug(f"Filler ({tier}): '{clip.text}' ({clip.duration_ms}ms)")
        except Exception as e:
            # A filler is best-effort; never let it break the turn.
            logger.warning(f"Could not send filler: {e}")


# One library per (filler config, TTS voice), shared by all client contexts.
_libraries: Dict[Tuple[str, str], FillerLibrary] = {}
_build_tasks: Dict[Tuple[str, str], asyncio.Task] = {}


def _tts_cache_key(tts_config: TTSConfig) -> str:
    engine_config = getattr(tts_config, tts_config.tts_model.lower(), None)
    engine_json = engine_config.model_dump_json() if engine_config else ""
    return f"{tts_config.tts_model}:{engine_json}"


def get_filler_library(
    filler_config: FillerConfig,
    tts_config: TTSConfig,
    tts_engine: TTSInterface,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[FillerLibrary]:
    """Return the shared library for this voice, starting its build if needed.

    Must be called from a running event loop. Returns None when fillers are disabled.
    The returned library may not be ready yet; turns before then get no fillers.
    """
    if not filler_config.enabled or tts_engine is None:
        return None
    tts_key = _tts_cache_key(tts_config)
    key = (filler_config.model_dump_json(), tts_key)
    library = _libraries.get(key)
    if library is None:
        library = FillerLibrary(
            filler_config, tts_engine, cache_key=tts_key, cache_dir=cache_dir
        )
        _libraries[key] = library
    task = _build_tasks.get(key)
    # run_server.py initializes in one asyncio.run() loop and serves in another, so
    # a build started at init gets cancelled when that loop closes: restart it here.
    stale = task is None or (
        not library.ready
        and (task.done() or task.get_loop() is not asyncio.get_running_loop())
    )
    if stale:
        _build_tasks[key] = asyncio.create_task(library.build())
    return library


def create_filler_controller(
    context: Any,
    websocket_send: WebSocketSend,
    is_voice_turn: bool,
) -> FillerController:
    """Build the controller for a turn; disabled unless this is a voice turn."""
    try:
        character_config = context.character_config
        filler_config = character_config.filler_config
        if not is_voice_turn or not filler_config.enabled:
            return FillerController.disabled()
        library = get_filler_library(
            filler_config, character_config.tts_config, context.tts_engine
        )
        if library is None:
            return FillerController.disabled()
        return FillerController(library, filler_config, websocket_send)
    except Exception as e:
        logger.warning(f"Fillers disabled for this turn: {e}")
        return FillerController.disabled()
