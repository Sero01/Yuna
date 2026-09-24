"""Background tasks run by hermes-agent through its Runs API."""

import asyncio
import itertools
import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from .prompts import WORKER_INSTRUCTIONS

RUNNING, DONE, FAILED, CANCELLED = "running", "done", "failed", "cancelled"

# Misheard speech must never grant a lasting permission, so voice can only answer
# "once" or "deny"; "session" and "always" are left to Hermes' own clients.
VOICE_APPROVAL_CHOICES = ("once", "deny")


class TaskStartError(Exception):
    """Hermes could not start the task. The message is safe to pass to the talker."""


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m"


@dataclass
class Approval:
    """A command Hermes won't run until the user allows it."""

    command: str
    description: str
    request_id: str
    requested_at: float
    told: bool = False

    @classmethod
    def from_event(cls, event: dict, now: float) -> "Approval":
        return cls(
            command=str(event.get("command") or ""),
            description=str(event.get("description") or ""),
            request_id=str(event.get("request_id") or ""),
            requested_at=now,
        )


@dataclass
class TaskRecord:
    id: str
    request: str
    run_id: str
    started_at: float
    status: str = RUNNING
    finished_at: Optional[float] = None
    result: str = ""
    error: str = ""
    last_tool: str = ""
    told: bool = False
    approval: Optional[Approval] = None

    def line(self, now: float) -> str:
        """One line for the talker's task-state message and Jev's active_tasks."""
        if self.status == RUNNING and self.approval is not None:
            a = self.approval
            return (
                f"{self.id} [running {_ago(now - self.started_at)}, paused: waiting "
                f"{_ago(now - a.requested_at)} for the user to allow it once or deny it] "
                f"{self.request} -> needs permission to {a.description}: `{a.command}`"
            )
        if self.status == RUNNING:
            step = f", last step: {self.last_tool}" if self.last_tool else ""
            return f"{self.id} [running {_ago(now - self.started_at)}{step}] {self.request}"
        ago = _ago(now - (self.finished_at or now))
        if self.status == DONE:
            return f"{self.id} [done {ago} ago] {self.request} -> {self.result[:300]}"
        if self.status == FAILED:
            return f"{self.id} [failed {ago} ago: {self.error[:150]}] {self.request}"
        return f"{self.id} [cancelled {ago} ago] {self.request}"


class TaskManager:
    """Starts and follows hermes-agent runs; calls `listener` when one finishes."""

    def __init__(
        self,
        get_client: Callable[[], httpx.AsyncClient],
        base_url: str,
        api_key: str,
        timeout_s: float = 600.0,
        max_active: int = 3,
        recent_window_s: float = 600.0,
        poll_interval_s: float = 2.0,
        clock: Callable[[], float] = time.time,
    ):
        self._get_client = get_client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_active = max_active
        self.recent_window_s = recent_window_s
        self.poll_interval_s = poll_interval_s
        self._clock = clock
        self.tasks: Dict[str, TaskRecord] = {}
        self.listener: Optional[Callable[[TaskRecord], None]] = None
        self._watchers: Dict[str, asyncio.Task] = {}
        self._ids = itertools.count(1)

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    # ---- queries ----

    def get(self, task_id: Optional[str]) -> Optional[TaskRecord]:
        return self.tasks.get(task_id) if task_id else None

    def active(self) -> List[TaskRecord]:
        return [t for t in self.tasks.values() if t.status == RUNNING]

    def recent(self, limit: int = 5) -> List[TaskRecord]:
        now = self._clock()
        keep = [
            t
            for t in self.tasks.values()
            if t.status == RUNNING
            or now - (t.finished_at or now) <= self.recent_window_s
        ]
        return keep[-limit:]

    def recent_lines(self, limit: int = 5) -> List[Tuple[str, str]]:
        now = self._clock()
        return [(t.id, t.line(now)) for t in self.recent(limit)]

    def untold_results(self) -> List[TaskRecord]:
        return [
            t for t in self.tasks.values() if t.status in (DONE, FAILED) and not t.told
        ]

    def mark_told(self, task_id: str) -> None:
        rec = self.tasks.get(task_id)
        if rec is not None and rec.status in (DONE, FAILED):
            rec.told = True

    def awaiting_approval(self) -> List[TaskRecord]:
        return [t for t in self.active() if t.approval is not None]

    def untold_approvals(self) -> List[TaskRecord]:
        return [t for t in self.awaiting_approval() if not t.approval.told]

    def mark_approval_told(self, task_id: str) -> None:
        rec = self.tasks.get(task_id)
        if rec is not None and rec.approval is not None:
            rec.approval.told = True

    # ---- actions ----

    async def start(self, request: str, history: List[Dict[str, str]]) -> TaskRecord:
        if len(self.active()) >= self.max_active:
            raise TaskStartError(f"already running {self.max_active} tasks")
        body = {
            "input": request,
            "conversation_history": history,
            "instructions": WORKER_INSTRUCTIONS,
        }
        try:
            response = await self._get_client().post(
                f"{self.base_url}/v1/runs",
                json=body,
                headers=self._headers,
                timeout=10.0,
            )
        except httpx.HTTPError as e:
            raise TaskStartError("Hermes isn't reachable") from e
        if response.status_code >= 300:
            raise TaskStartError(
                f"Hermes refused the task (HTTP {response.status_code})"
            )
        try:
            run_id = response.json().get("run_id")
        except ValueError:
            run_id = None
        if not run_id:
            raise TaskStartError("Hermes didn't return a run id")
        rec = TaskRecord(
            id=f"t{next(self._ids)}",
            request=request,
            run_id=run_id,
            started_at=self._clock(),
        )
        self.tasks[rec.id] = rec
        self._watchers[rec.id] = asyncio.create_task(self._watch(rec))
        logger.info(f"Task {rec.id} started as Hermes run {run_id}: {request}")
        return rec

    async def cancel(self, task_id: str) -> bool:
        rec = self.tasks.get(task_id)
        if rec is None or rec.status != RUNNING:
            return False
        self._finish(rec, CANCELLED)
        watcher = self._watchers.pop(task_id, None)
        if watcher is not None:
            watcher.cancel()
        await self._request_stop(rec)
        return True

    async def steer(self, task_id: str, text: str) -> bool:
        rec = self.tasks.get(task_id)
        if rec is None or rec.status != RUNNING:
            return False
        try:
            response = await self._get_client().post(
                f"{self.base_url}/v1/runs/{rec.run_id}/steer",
                json={"input": text},
                headers=self._headers,
                timeout=10.0,
            )
            return response.status_code == 200 and bool(response.json().get("accepted"))
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(f"Could not steer task {task_id}: {e}")
            return False

    async def respond(self, task_id: str, choice: str) -> bool:
        """Answer the task's pending approval; False if nothing was waiting for it."""
        rec = self.tasks.get(task_id)
        if choice not in VOICE_APPROVAL_CHOICES or rec is None or rec.approval is None:
            return False
        approval, rec.approval = rec.approval, None
        body = {"choice": choice}
        if approval.request_id:
            body["request_id"] = approval.request_id
        try:
            response = await self._get_client().post(
                f"{self.base_url}/v1/runs/{rec.run_id}/approval",
                json=body,
                headers=self._headers,
                timeout=10.0,
            )
        except httpx.HTTPError as e:
            logger.warning(f"Could not answer the approval for task {task_id}: {e}")
            return False
        logger.info(
            f"Task {task_id}: approval answered '{choice}' "
            f"(HTTP {response.status_code}) for `{approval.command}`"
        )
        return response.status_code == 200

    def close(self) -> None:
        for watcher in self._watchers.values():
            try:
                watcher.cancel()
            except RuntimeError:  # its event loop is already closed
                pass
        self._watchers.clear()

    # ---- following a run ----

    async def _watch(self, rec: TaskRecord) -> None:
        try:
            await asyncio.wait_for(self._track(rec), self.timeout_s)
        except asyncio.TimeoutError:
            await self._request_stop(rec)
            self._finish(rec, FAILED, error="timed out")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"Task {rec.id}: watcher failed")
            self._finish(
                rec, FAILED, error=f"lost track of the task ({type(e).__name__})"
            )
        finally:
            if self._watchers.get(rec.id) is asyncio.current_task():
                del self._watchers[rec.id]

    async def _track(self, rec: TaskRecord) -> None:
        try:
            await self._follow_events(rec)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                f"Task {rec.id}: event stream failed ({type(e).__name__}: {e})"
            )
        failures = 0
        while rec.status == RUNNING:
            try:
                response = await self._get_client().get(
                    f"{self.base_url}/v1/runs/{rec.run_id}",
                    headers=self._headers,
                    timeout=10.0,
                )
                response.raise_for_status()
                self._apply_status(rec, response.json())
                failures = 0
            except (httpx.HTTPError, ValueError) as e:
                failures += 1
                logger.warning(f"Task {rec.id}: status poll failed ({e})")
                if failures >= 3:
                    self._finish(rec, FAILED, error="lost contact with Hermes")
                    return
            if rec.status == RUNNING:
                await asyncio.sleep(self.poll_interval_s)

    async def _follow_events(self, rec: TaskRecord) -> None:
        async with self._get_client().stream(
            "GET",
            f"{self.base_url}/v1/runs/{rec.run_id}/events",
            headers=self._headers,
            timeout=httpx.Timeout(10.0, read=None),
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"events endpoint returned HTTP {response.status_code}"
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                self._apply_event(rec, event)
                if rec.status != RUNNING:
                    return

    def _apply_event(self, rec: TaskRecord, event: dict) -> None:
        name = event.get("event")
        if name == "approval.request":
            self._hold_approval(rec, event)
            return
        # Hermes blocks the whole step while it waits, so any sign of new progress
        # (except a parallel tool finishing) means it is no longer waiting.
        if name != "tool.completed":
            rec.approval = None
        if name == "tool.started":
            rec.last_tool = event.get("tool") or rec.last_tool
        elif name == "run.completed":
            self._finish(rec, DONE, result=event.get("output") or "")
        elif name == "run.failed":
            self._finish(rec, FAILED, error=event.get("error") or "the task failed")
        elif name == "run.cancelled":
            self._finish(rec, CANCELLED)

    def _apply_status(self, rec: TaskRecord, status: dict) -> None:
        state = status.get("status")
        if state == "completed":
            self._finish(rec, DONE, result=status.get("output") or "")
        elif state == "failed":
            self._finish(rec, FAILED, error=status.get("error") or "the task failed")
        elif state == "cancelled":
            self._finish(rec, CANCELLED)
        elif state == "waiting_for_approval":
            self._hold_approval(rec, status.get("approval") or {})
        else:
            rec.approval = None

    def _hold_approval(self, rec: TaskRecord, event: dict) -> None:
        approval = Approval.from_event(event, self._clock())
        held = rec.approval
        if held is not None and (held.request_id, held.command) == (
            approval.request_id,
            approval.command,
        ):
            return  # already known (the status poll repeats it)
        rec.approval = approval
        logger.info(
            f"Task {rec.id} needs approval to {approval.description}: `{approval.command}`"
        )
        self._notify(rec)

    def _notify(self, rec: TaskRecord) -> None:
        if self.listener is None:
            return
        try:
            self.listener(rec)
        except Exception:
            logger.exception("Task listener failed")

    def _finish(
        self, rec: TaskRecord, status: str, result: str = "", error: str = ""
    ) -> None:
        if rec.status != RUNNING:
            return
        rec.status = status
        rec.approval = None
        rec.result = (result or "").strip()
        rec.error = (error or "").strip()
        rec.finished_at = self._clock()
        logger.info(f"Task {rec.id} {status}: {rec.result or rec.error}"[:300])
        if status in (DONE, FAILED):
            self._notify(rec)

    async def _request_stop(self, rec: TaskRecord) -> None:
        try:
            await self._get_client().post(
                f"{self.base_url}/v1/runs/{rec.run_id}/stop",
                headers=self._headers,
                timeout=10.0,
            )
        except httpx.HTTPError as e:
            logger.warning(f"Could not stop Hermes run {rec.run_id}: {e}")
