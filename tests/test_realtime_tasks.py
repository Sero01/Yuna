"""Hermes runs: start, events, completion, failure, cancel, steer, timeout, polling.

Run directly:  .venv/Scripts/python.exe tests/test_realtime_tasks.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from _harness import run_module  # noqa: E402
from realtime_fakes import FakeHermes, wait_until  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.loop_client import LoopBoundClient  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.prompts import (  # noqa: E402
    MEMORY_REVIEW_INSTRUCTIONS,
    TASK_REMEMBER_HINT,
    WORKER_INSTRUCTIONS,
    WORKER_SUMMARY,
)
from src.open_llm_vtuber.agent.agents.realtime.tasks import (  # noqa: E402
    TaskManager,
    TaskRecord,
    TaskStartError,
)


def make_manager(hermes, **kw):
    get = LoopBoundClient(transport=httpx.MockTransport(hermes.handler)).get
    return TaskManager(get, "http://localhost:8642/", "hermes-key", **kw)


HISTORY = [
    {"role": "user", "content": "hey"},
    {"role": "assistant", "content": "Hmph."},
]


async def test_start_posts_a_run():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    rec = await tasks.start("weather in delhi", HISTORY)
    method, path, body, auth = hermes.requests[0]
    assert (method, path) == ("POST", "/v1/runs")
    assert body == {
        "input": "weather in delhi",
        "conversation_history": HISTORY,
        "instructions": WORKER_INSTRUCTIONS,
    }
    assert auth == "Bearer hermes-key"
    assert (rec.id, rec.run_id, rec.status) == ("t1", "run_1", "running")
    assert [t.id for t in tasks.active()] == ["t1"]
    await wait_until(lambda: hermes.calls("GET", "/events"))
    tasks.close()


async def test_start_asks_for_the_configured_reasoning_effort():
    hermes = FakeHermes()
    tasks = make_manager(hermes, reasoning_effort="high")
    await tasks.start("weather in delhi", [])
    assert hermes.requests[0][2]["model_options"] == {"reasoning_effort": "high"}
    tasks.close()


async def test_start_points_hermes_at_the_saved_transcripts():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    tasks.set_transcripts(
        "C:/yuna/chat_history/mao_pro_001", "2026-09-24_01-39-48_abc.json"
    )
    await tasks.start("which restaurant did you pick last time", [])
    instructions = hermes.requests[0][2]["instructions"]
    assert instructions.startswith(WORKER_INSTRUCTIONS)
    assert "C:/yuna/chat_history/mao_pro_001" in instructions
    assert "2026-09-24_01-39-48_abc.json" in instructions
    tasks.close()


async def test_start_gives_hermes_the_conversation_summary():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    tasks.set_transcripts(
        "C:/yuna/chat_history/mao_pro_001", "2026-09-24_01-39-48_abc.json"
    )
    await tasks.start(
        "book that place", HISTORY, summary="They picked a ramen place in Indiranagar."
    )
    instructions = hermes.requests[0][2]["instructions"]
    expected = WORKER_SUMMARY.format(
        summary="They picked a ramen place in Indiranagar."
    )
    assert instructions.startswith(f"{WORKER_INSTRUCTIONS}\n\n{expected}\n\n")
    assert "C:/yuna/chat_history/mao_pro_001" in instructions
    tasks.close()


async def test_start_can_ask_hermes_to_consider_remembering():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    await tasks.start(
        "my birthday is november 14th, remind me a week before", [], remember=True
    )
    instructions = hermes.requests[0][2]["instructions"]
    assert instructions.startswith(WORKER_INSTRUCTIONS)
    assert instructions.endswith(TASK_REMEMBER_HINT)
    tasks.close()


async def test_memory_review_runs_silently():
    hermes = FakeHermes()
    tasks = make_manager(hermes, reasoning_effort="high")
    announced = []
    tasks.listener = announced.append
    job = tasks.remember("Sam: hey\nYuna: What now?\nSam: my sister's name is ayesha")
    await wait_until(lambda: hermes.calls("GET", "/events"))
    method, path, body, _ = hermes.requests[0]
    assert (method, path) == ("POST", "/v1/runs")
    assert body["instructions"] == MEMORY_REVIEW_INSTRUCTIONS
    assert body["input"].endswith("Sam: my sister's name is ayesha")
    assert body["model_options"] == {"reasoning_effort": "high"}
    assert "conversation_history" not in body
    assert tasks.tasks == {} and tasks.active() == []
    hermes.emit("run_1", event="run.completed", output="Saved: sister is Ayesha.")
    assert await job == "Saved: sister is Ayesha."
    assert announced == [] and tasks.untold_results() == []
    tasks.close()


async def test_memory_review_denies_approval_requests():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    job = tasks.remember("Sam: i hate horror movies")
    await wait_until(lambda: hermes.calls("GET", "/events"))
    hermes.emit("run_1", event="approval.request", command="rm -rf x", request_id="r1")
    await wait_until(lambda: hermes.calls("POST", "/approval"))
    assert hermes.calls("POST", "/approval")[0][2] == {
        "choice": "deny",
        "request_id": "r1",
    }
    hermes.emit("run_1", event="run.completed", output="nothing")
    assert await job == "nothing"
    tasks.close()


async def test_memory_review_failures_are_only_logged():
    hermes = FakeHermes(start_status=500)
    tasks = make_manager(hermes)
    assert await tasks.remember("Sam: i'm vegetarian") is None
    hermes = FakeHermes()
    hermes.unreachable = True
    tasks = make_manager(hermes)
    assert await tasks.remember("Sam: i'm vegetarian") is None


async def test_completed_run_reports_result_once():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    finished = []
    tasks.listener = finished.append
    rec = await tasks.start("weather in delhi", [])
    hermes.emit(
        "run_1", event="tool.started", tool="web_search", preview="delhi weather"
    )
    await wait_until(lambda: rec.last_tool == "web_search")
    assert "last step: web_search" in rec.line(rec.started_at + 12)
    hermes.emit("run_1", event="run.completed", output=" 34 degrees and humid. ")
    await wait_until(lambda: rec.status == "done")
    assert rec.result == "34 degrees and humid."
    assert finished == [rec]
    assert tasks.untold_results() == [rec]
    assert "-> 34 degrees and humid." in rec.line(rec.finished_at + 120)
    assert "[done 2m ago]" in rec.line(rec.finished_at + 120)


async def test_failed_run_is_reported():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    finished = []
    tasks.listener = finished.append
    rec = await tasks.start("send email", [])
    hermes.emit("run_1", event="run.failed", error="provider auth failed")
    await wait_until(lambda: rec.status == "failed")
    assert rec.error == "provider auth failed" and finished == [rec]
    assert "[failed" in rec.line(
        rec.finished_at
    ) and "provider auth failed" in rec.line(rec.finished_at)


async def test_cancel_stops_the_run_without_announcing():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    finished = []
    tasks.listener = finished.append
    rec = await tasks.start("find flights", [])
    assert await tasks.cancel("t1") is True
    assert rec.status == "cancelled"
    assert hermes.calls("POST", "/v1/runs/run_1/stop")
    assert await tasks.cancel("t1") is False
    assert await tasks.cancel("nope") is False
    assert finished == [] and tasks.untold_results() == []


async def test_steer():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    await tasks.start("find flights", [])
    assert await tasks.steer("t1", "only under five thousand") is True
    assert hermes.calls("POST", "/steer")[0][2] == {"input": "only under five thousand"}
    hermes.steer_accepted = False
    assert await tasks.steer("t1", "and hotels") is False
    assert await tasks.steer("t9", "x") is False


async def test_timeout_stops_and_fails_the_run():
    hermes = FakeHermes()
    tasks = make_manager(hermes, timeout_s=0.1)
    finished = []
    tasks.listener = finished.append
    rec = await tasks.start("slow thing", [])
    await wait_until(lambda: rec.status == "failed")
    assert rec.error == "timed out" and finished == [rec]
    await wait_until(lambda: hermes.calls("POST", "/stop"))


async def test_max_active_tasks():
    hermes = FakeHermes()
    tasks = make_manager(hermes, max_active=1)
    await tasks.start("one", [])
    try:
        await tasks.start("two", [])
    except TaskStartError as e:
        assert "already running" in str(e)
        return
    raise AssertionError("expected TaskStartError")


async def test_start_errors_are_task_start_errors():
    for hermes in (FakeHermes(start_status=401), FakeHermes()):
        if hermes.start_status == 202:
            hermes.unreachable = True
        tasks = make_manager(hermes)
        try:
            await tasks.start("x", [])
        except TaskStartError as e:
            assert "Hermes" in str(e)
            continue
        raise AssertionError("expected TaskStartError")


async def test_stream_end_without_result_falls_back_to_polling():
    hermes = FakeHermes(run_status={"status": "completed", "output": "done via poll"})
    tasks = make_manager(hermes, poll_interval_s=0.01)
    rec = await tasks.start("x", [])
    hermes.end_stream("run_1")
    await wait_until(lambda: rec.status == "done")
    assert rec.result == "done via poll"
    assert hermes.calls("GET", "/v1/runs/run_1")


async def test_recent_lines_untold_and_mark_told():
    now = [1000.0]
    hermes = FakeHermes()
    tasks = make_manager(hermes, clock=lambda: now[0], recent_window_s=600)
    first = await tasks.start("old one", [])
    hermes.emit("run_1", event="run.completed", output="old result")
    await wait_until(lambda: first.status == "done")
    now[0] += 700  # the first task is now outside the recent window
    second = await tasks.start("new one", [])
    assert [tid for tid, _ in tasks.recent_lines()] == ["t2"]
    assert tasks.untold_results() == [first]
    tasks.mark_told("t1")
    assert tasks.untold_results() == []
    tasks.close()
    assert second.status == "running"


async def test_mark_told_ignores_running_tasks():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    rec = await tasks.start("x", [])
    tasks.mark_told("t1")
    assert rec.told is False, "a running task's result must still be announced later"
    tasks.close()


APPROVAL_EVENT = {
    "event": "approval.request",
    "command": "hermes gateway restart",
    "description": "restart a system service",
    "choices": ["once", "session", "always", "deny"],
    "request_id": "req_1",
}


async def test_approval_request_is_held_and_announced():
    now = [1000.0]
    hermes = FakeHermes()
    tasks = make_manager(hermes, clock=lambda: now[0])
    notified = []
    tasks.listener = notified.append
    rec = await tasks.start("restart the gateway", [])
    hermes.emit("run_1", **APPROVAL_EVENT)
    await wait_until(lambda: rec.approval is not None)
    assert notified == [rec], "an approval request is announced right away"
    assert rec.status == "running"
    assert tasks.untold_approvals() == [rec]
    now[0] += 20
    line = rec.line(now[0])
    assert "waiting 20s for the user to allow it once or deny it" in line, line
    assert "restart a system service" in line and "hermes gateway restart" in line
    tasks.mark_approval_told("t1")
    assert tasks.untold_approvals() == []
    tasks.close()


async def test_respond_sends_the_choice_for_that_request():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    rec = await tasks.start("restart the gateway", [])
    hermes.emit("run_1", **APPROVAL_EVENT)
    await wait_until(lambda: rec.approval is not None)
    assert tasks.awaiting_approval() == [rec]
    assert await tasks.respond("t1", "once") is True
    method, path, body, _ = hermes.calls("POST", "/approval")[0]
    assert path == "/v1/runs/run_1/approval"
    assert body == {"choice": "once", "request_id": "req_1"}
    assert rec.approval is None and tasks.awaiting_approval() == []
    tasks.close()


async def test_voice_can_only_allow_once_or_deny():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    rec = await tasks.start("restart the gateway", [])
    hermes.emit("run_1", **APPROVAL_EVENT)
    await wait_until(lambda: rec.approval is not None)
    for choice in ("always", "session"):
        assert await tasks.respond("t1", choice) is False
    assert not hermes.calls("POST", "/approval")
    assert rec.approval is not None
    tasks.close()


async def test_respond_after_hermes_gave_up_waiting_fails_and_clears():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    rec = await tasks.start("restart the gateway", [])
    hermes.emit("run_1", **APPROVAL_EVENT)
    await wait_until(lambda: rec.approval is not None)
    hermes.approval_pending = False
    assert await tasks.respond("t1", "deny") is False
    assert rec.approval is None
    assert await tasks.respond("t1", "deny") is False, "nothing is waiting any more"
    assert len(hermes.calls("POST", "/approval")) == 1
    tasks.close()


async def test_approval_clears_when_the_run_moves_on():
    hermes = FakeHermes()
    tasks = make_manager(hermes)
    rec = await tasks.start("restart the gateway", [])
    hermes.emit("run_1", **APPROVAL_EVENT)
    await wait_until(lambda: rec.approval is not None)
    hermes.emit("run_1", event="tool.completed", tool="web_search")
    hermes.emit("run_1", event="tool.started", tool="read_file")
    await wait_until(lambda: rec.last_tool == "read_file")
    assert rec.approval is None, "a new step means Hermes stopped waiting"
    hermes.emit("run_1", **APPROVAL_EVENT)
    await wait_until(lambda: rec.approval is not None)
    hermes.emit("run_1", event="tool.completed", tool="web_search")
    hermes.emit("run_1", event="run.completed", output="done")
    await wait_until(lambda: rec.status == "done")
    assert rec.approval is None


async def test_polled_status_reports_a_waiting_approval():
    hermes = FakeHermes(
        run_status={
            "status": "waiting_for_approval",
            "approval": dict(APPROVAL_EVENT),
        }
    )
    tasks = make_manager(hermes, poll_interval_s=0.01)
    notified = []
    tasks.listener = notified.append
    rec = await tasks.start("restart the gateway", [])
    hermes.end_stream("run_1")
    await wait_until(lambda: rec.approval is not None)
    assert rec.approval.command == "hermes gateway restart"
    await asyncio.sleep(0.05)  # several more polls see the same request
    assert notified == [rec], "one request is announced once"
    hermes.run_status = {"status": "running"}
    await wait_until(lambda: rec.approval is None)
    tasks.close()


def test_line_formats():
    rec = TaskRecord(id="t3", request="weather", run_id="r", started_at=100.0)
    assert rec.line(112) == "t3 [running 12s] weather"
    rec.status, rec.finished_at = "cancelled", 130.0
    assert rec.line(130 + 61) == "t3 [cancelled 1m ago] weather"


if __name__ == "__main__":
    run_module(globals())
