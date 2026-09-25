"""SenseVoice keep-warm thread: decodes only while idle, stops with its engine.

Run directly:  .venv/Scripts/python.exe tests/test_asr_keep_warm.py
"""

import gc
import os
import sys
import threading
import time
import weakref

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.asr import sherpa_onnx_asr  # noqa: E402


class FakeEngine:
    def __init__(self, idle_since):
        self._last_decode = idle_since
        self.decodes = 0

    def _decode(self, audio):
        self.decodes += 1
        return ""


def run_warmer(engine):
    thread = threading.Thread(
        target=sherpa_onnx_asr._keep_warm, args=(weakref.ref(engine),), daemon=True
    )
    thread.start()
    return thread


def test_idle_engine_is_kept_warm_and_busy_one_is_not():
    saved = sherpa_onnx_asr.KEEP_WARM_S
    sherpa_onnx_asr.KEEP_WARM_S = 0.05
    try:
        idle = FakeEngine(idle_since=time.monotonic() - 60)
        busy = FakeEngine(idle_since=time.monotonic() + 3600)  # "just decoded"
        threads = [run_warmer(idle), run_warmer(busy)]
        time.sleep(0.3)
        assert idle.decodes >= 1, idle.decodes
        assert busy.decodes == 0, busy.decodes
        del idle, busy
        gc.collect()
        for thread in threads:
            thread.join(timeout=1.0)
            assert not thread.is_alive(), "the warmer stops once its engine is gone"
    finally:
        sherpa_onnx_asr.KEEP_WARM_S = saved


if __name__ == "__main__":
    run_module(globals())
