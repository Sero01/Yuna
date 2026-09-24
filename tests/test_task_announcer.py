"""Task announcer: waits for a quiet moment, then starts one task-result turn.

Run directly:  .venv/Scripts/python.exe tests/test_task_announcer.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.conversations.task_announcer import TaskAnnouncer  # noqa: E402


async def test_waits_for_the_conversation_to_be_quiet():
    busy = {"value": True}
    started = []
    announcer = TaskAnnouncer(
        is_busy=lambda: busy["value"],
        start_turn=lambda: started.append(1),
        quiet_gap_s=0.1,
        poll_s=0.01,
    )
    announcer.notify()
    await asyncio.sleep(0.15)
    assert started == [], "never interrupts a running turn"
    busy["value"] = False
    await asyncio.sleep(0.04)
    assert started == [], "waits for the quiet gap"
    await asyncio.sleep(0.15)
    assert started == [1]


async def test_several_notifications_start_one_turn():
    started = []
    announcer = TaskAnnouncer(
        lambda: False, lambda: started.append(1), quiet_gap_s=0.02, poll_s=0.01
    )
    announcer.notify()
    announcer.notify()
    announcer.notify()
    await asyncio.sleep(0.1)
    assert started == [1]
    announcer.notify()
    await asyncio.sleep(0.1)
    assert started == [1, 1], "a later notification starts another turn"


async def test_async_start_turn_is_awaited():
    started = []

    async def start():
        await asyncio.sleep(0)
        started.append(1)

    announcer = TaskAnnouncer(lambda: False, start, quiet_gap_s=0, poll_s=0.01)
    announcer.notify()
    await asyncio.sleep(0.05)
    assert started == [1]


async def test_start_turn_errors_are_contained():
    def boom():
        raise RuntimeError("socket gone")

    announcer = TaskAnnouncer(lambda: False, boom, quiet_gap_s=0, poll_s=0.01)
    announcer.notify()
    await asyncio.sleep(0.05)
    announcer.notify()  # still usable afterwards
    await asyncio.sleep(0.05)


async def test_close_cancels_a_pending_announcement():
    busy = {"value": True}
    started = []
    announcer = TaskAnnouncer(
        lambda: busy["value"], lambda: started.append(1), quiet_gap_s=0, poll_s=0.01
    )
    announcer.notify()
    announcer.close()
    busy["value"] = False
    await asyncio.sleep(0.05)
    assert started == []


if __name__ == "__main__":
    run_module(globals())
