"""Tests for filler voice playback (conversations/filler.py).

Uses a fake TTS engine that writes sine-tone WAVs and a fake websocket send that
records payloads, so nothing touches the network or a real socket.

Run directly:  uv run python tests/test_filler.py
"""

import asyncio
import json
import os
import random
import sys
import tempfile
from types import SimpleNamespace

from pydub import AudioSegment
from pydub.generators import Sine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.open_llm_vtuber.config_manager.filler import FillerConfig  # noqa: E402
from src.open_llm_vtuber.conversations.filler import (  # noqa: E402
    FillerController,
    FillerLibrary,
    get_filler_library,
)

DELAY = 0.05  # escalation delay used in tests


class FakeTTS:
    """Writes a tone whose length is looked up per phrase; 'FAIL' phrases fail."""

    def __init__(self, out_dir: str, lengths_ms: dict | None = None, pad_ms: int = 0):
        self.out_dir = out_dir
        self.pad_ms = pad_ms
        os.makedirs(out_dir, exist_ok=True)
        self.lengths_ms = lengths_ms or {}
        self.calls = []

    async def async_generate_audio(self, text, file_name_no_ext=None):
        self.calls.append(text)
        if "FAIL" in text:
            return None
        path = os.path.join(self.out_dir, f"{file_name_no_ext}.wav")
        # Distinct pitch per phrase so payloads can be mapped back to their text.
        freq = 300 + sum(map(ord, text)) % 700
        tone = Sine(freq).to_audio_segment(duration=self.lengths_ms.get(text, 300))
        # edge-tts pads its output with silence; pad_ms simulates that.
        pad = AudioSegment.silent(duration=self.pad_ms)
        (pad + tone + pad).export(path, format="wav")
        return path


class FakeSend:
    def __init__(self, lib):
        self.lib = lib
        self.messages = []

    async def __call__(self, text: str):
        self.messages.append(json.loads(text))

    def texts(self):
        """Map each sent payload back to the phrase it was rendered from."""
        by_audio = {
            clip.payload["audio"]: clip.text
            for clips in self.lib.clips.values()
            for clip in clips
        }
        return [by_audio[m["audio"]] for m in self.messages]


def make_config(**overrides) -> FillerConfig:
    base = dict(
        enabled=True,
        probability=1.0,
        no_repeat_window=1,
        escalation_delay=DELAY,
        max_tier0_ms=900,
        tier0_phrases=["Hmm.", "Mm."],
        escalation_phrases=["Still thinking."],
        tool_phrases=["Let me look that up."],
    )
    base.update(overrides)
    return FillerConfig(**base)


async def make_library(tmp, config=None, lengths_ms=None, seed=0, pad_ms=0):
    config = config or make_config()
    tts = FakeTTS(tmp, lengths_ms, pad_ms)
    lib = FillerLibrary(
        config=config,
        tts_engine=tts,
        cache_key="test-voice",
        cache_dir=os.path.join(tmp, "fillers"),
        rng=random.Random(seed),
    )
    await lib.build()
    return lib, tts


def controller(lib, config=None, seed=0):
    send = FakeSend(lib)
    ctrl = FillerController(
        library=lib,
        config=config or make_config(),
        websocket_send=send,
        rng=random.Random(seed),
    )
    return ctrl, send


async def test_tier0_payload_shape(tmp):
    lib, _ = await make_library(tmp)
    ctrl, send = controller(lib)
    await ctrl.start()
    ctrl.stop()
    assert len(send.messages) == 1, send.messages
    msg = send.messages[0]
    assert msg["type"] == "audio"
    assert msg["display_text"] is None  # no chat bubble / subtitle
    assert msg["audio"] and msg["volumes"]
    assert send.texts()[0] in ("Hmm.", "Mm.")


async def test_probability_zero_skips_tier0(tmp):
    cfg = make_config(probability=0.0)
    lib, _ = await make_library(tmp, cfg)
    ctrl, send = controller(lib, cfg)
    await ctrl.start()
    ctrl.stop()
    assert send.messages == []


async def test_escalation_fires_after_delay(tmp):
    lib, _ = await make_library(tmp)
    ctrl, send = controller(lib)
    await ctrl.start()
    await asyncio.sleep(DELAY * 3)
    ctrl.stop()
    assert send.texts()[1:] == ["Still thinking."], send.texts()


async def test_escalation_even_when_tier0_gated(tmp):
    cfg = make_config(probability=0.0)
    lib, _ = await make_library(tmp, cfg)
    ctrl, send = controller(lib, cfg)
    await ctrl.start()
    await asyncio.sleep(DELAY * 3)
    ctrl.stop()
    assert send.texts() == ["Still thinking."], send.texts()


async def test_stop_cancels_escalation(tmp):
    lib, _ = await make_library(tmp)
    ctrl, send = controller(lib)
    await ctrl.start()
    ctrl.stop()
    await asyncio.sleep(DELAY * 3)
    assert len(send.messages) == 1, send.texts()
    # Calls after stop are no-ops.
    await ctrl.on_tool_status({"type": "tool_call_status", "status": "running"})
    assert len(send.messages) == 1, send.texts()


async def test_tool_line_replaces_escalation_and_plays_once(tmp):
    lib, _ = await make_library(tmp)
    ctrl, send = controller(lib)
    await ctrl.start()
    await ctrl.on_tool_status({"type": "tool_call_status", "status": "running"})
    await ctrl.on_tool_status({"type": "tool_call_status", "status": "completed"})
    await ctrl.on_tool_status({"type": "tool_call_status", "status": "running"})
    await asyncio.sleep(DELAY * 3)
    ctrl.stop()
    assert send.texts()[1:] == ["Let me look that up."], send.texts()


async def test_tool_line_ignored_after_escalation(tmp):
    lib, _ = await make_library(tmp)
    ctrl, send = controller(lib)
    await ctrl.start()
    await asyncio.sleep(DELAY * 3)
    await ctrl.on_tool_status({"type": "tool_call_status", "status": "running"})
    ctrl.stop()
    assert send.texts()[1:] == ["Still thinking."], send.texts()


async def test_no_repeat_window(tmp):
    lib, _ = await make_library(tmp)
    picks = [lib.pick("tier0").text for _ in range(20)]
    for a, b in zip(picks, picks[1:]):
        assert a != b, picks


async def test_long_tier0_clips_dropped(tmp):
    lib, _ = await make_library(tmp, lengths_ms={"Hmm.": 2000, "Mm.": 300})
    picks = {lib.pick("tier0").text for _ in range(10)}
    assert picks == {"Mm."}, picks
    # Escalation clips have no length cap.
    long_lib, _ = await make_library(
        os.path.join(tmp, "b"), lengths_ms={"Still thinking.": 2000}
    )
    assert long_lib.pick("escalation").text == "Still thinking."


async def test_silence_trimmed_before_length_check(tmp):
    lib, _ = await make_library(tmp, pad_ms=800)
    clips = lib.clips["tier0"]
    assert {c.text for c in clips} == {"Hmm.", "Mm."}, clips
    for clip in clips:
        # 300ms tone plus a little breathing room, not 1900ms of padding.
        assert 300 <= clip.duration_ms <= 450, clip.duration_ms


async def test_failed_render_is_skipped(tmp):
    cfg = make_config(tier0_phrases=["FAIL one", "Mm."], tool_phrases=["FAIL two"])
    lib, _ = await make_library(tmp, cfg)
    assert lib.ready
    assert {lib.pick("tier0").text for _ in range(5)} == {"Mm."}
    assert lib.pick("tool") is None


async def test_cache_reused(tmp):
    _, tts1 = await make_library(tmp)
    assert len(tts1.calls) == 4
    _, tts2 = await make_library(tmp)
    assert tts2.calls == [], tts2.calls


async def test_not_ready_library_sends_nothing(tmp):
    cfg = make_config()
    lib = FillerLibrary(
        config=cfg,
        tts_engine=FakeTTS(tmp),
        cache_key="x",
        cache_dir=os.path.join(tmp, "fillers"),
    )
    ctrl, send = controller(lib, cfg)
    await ctrl.start()
    await asyncio.sleep(DELAY * 3)
    ctrl.stop()
    assert send.messages == []


async def test_disabled_controller_is_noop(tmp):
    ctrl = FillerController.disabled()
    await ctrl.start()
    await ctrl.on_tool_status({"type": "tool_call_status", "status": "running"})
    ctrl.stop()


async def test_send_failure_does_not_raise(tmp):
    lib, _ = await make_library(tmp)

    async def broken_send(_):
        raise RuntimeError("socket closed")

    ctrl = FillerController(
        library=lib, config=make_config(), websocket_send=broken_send
    )
    await ctrl.start()
    await asyncio.sleep(DELAY * 3)
    ctrl.stop()


async def test_library_survives_init_loop_exit(tmp):
    """run_server.py initializes in one asyncio.run() loop, then uvicorn serves in
    another. A build started (and cancelled) in the init loop must restart."""
    cfg = make_config()
    tts_config = SimpleNamespace(tts_model="fake_" + os.path.basename(tmp), fake=None)
    cache_dir = os.path.join(tmp, "fillers")

    def in_fresh_loop(wait_for_build: bool):
        async def run():
            lib = get_filler_library(cfg, tts_config, FakeTTS(tmp), cache_dir)
            if wait_for_build:
                for _ in range(200):
                    if lib.ready:
                        break
                    await asyncio.sleep(0.01)
            return lib

        return asyncio.run(run())

    first = await asyncio.to_thread(in_fresh_loop, False)  # init loop exits early
    assert not first.ready, "init-loop build should not have finished"
    second = await asyncio.to_thread(in_fresh_loop, True)  # "uvicorn" loop
    assert second is first, "registry should reuse the library"
    assert second.ready, "build should restart in the new loop"
    assert len(second.clips["tier0"]) == 2, second.clips


async def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                await test(tmp)
                print(f"PASS {test.__name__}")
            except Exception as e:
                failed += 1
                print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
