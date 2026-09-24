"""Per-turn routing: Jev (OpenRouter decisions API) with a JSON-classifier fallback."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from .prompts import (
    APPROVAL_Q,
    FALLBACK_ROUTER_PROMPT,
    FOLLOWUP_ACTION_Q,
    FOLLOWUP_TASK_INSTRUCTIONS,
    OFFER_Q,
    ROUTE_Q,
)

FOLLOWUP_ACTIONS = ("status", "change", "cancel")
# Allowing a command needs a confident "yes"; refusing is safe on any "no".
APPROVE_MIN_P = 0.8


@dataclass
class RouterState:
    latest_user_message: str
    recent_conversation: str = ""
    active_tasks: List[Tuple[str, str]] = field(default_factory=list)  # (task id, line)
    pending_offer: Optional[str] = None
    pending_approval: Optional[str] = None  # what Hermes is waiting to be allowed to do


@dataclass
class Route:
    kind: str  # chat | new_task | followup | unsure | accept_offer | approval
    p_task: float = 0.0
    # followup: status | change | cancel; approval: once | deny
    action: Optional[str] = None
    task_id: Optional[str] = None  # followup only
    source: str = "jev"  # jev | fallback | default


def _task_lines(state: RouterState) -> str:
    return "\n".join(line for _, line in state.active_tasks) or "none"


class FallbackRouter:
    """Asks a fast chat model for a JSON verdict when Jev is unavailable."""

    def __init__(
        self,
        get_client: Callable[[], httpx.AsyncClient],
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "",
        timeout_s: float = 1.0,
    ):
        self._get_client = get_client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.timeout_s = timeout_s

    async def decide(self, state: RouterState) -> Route:
        prompt = FALLBACK_ROUTER_PROMPT.format(
            active_tasks=_task_lines(state),
            recent_conversation=state.recent_conversation or "none",
            latest_user_message=state.latest_user_message,
        )
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 40,
        }
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": True}
        response = await self._get_client().post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON in fallback router reply: {content[:100]!r}")
        verdict = json.loads(match.group(0))
        route = verdict.get("route")
        if route == "followup" and state.active_tasks:
            action = verdict.get("action")
            return Route(
                "followup",
                action=action if action in FOLLOWUP_ACTIONS else "status",
                task_id=state.active_tasks[-1][0],
                source="fallback",
            )
        if route == "new_task":
            return Route("new_task", p_task=1.0, source="fallback")
        return Route("chat", source="fallback")


class JevRouter:
    """Routes each turn with one Jev call; never raises."""

    def __init__(
        self,
        get_client: Callable[[], httpx.AsyncClient],
        url: str,
        api_key: str,
        model: str = "typesafe/jev-1.13",
        timeout_s: float = 1.5,
        unsure_low: float = 0.35,
        unsure_high: float = 0.65,
        fallback: Optional[FallbackRouter] = None,
        fallback_after_s: float = 0.6,
    ):
        self._get_client = get_client
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.unsure_low = unsure_low
        self.unsure_high = unsure_high
        self.fallback = fallback
        self.fallback_after_s = fallback_after_s

    async def decide(self, state: RouterState) -> Route:
        """Ask Jev; if it hasn't answered after `fallback_after_s` (or failed), race
        the fallback router against it. The first usable answer wins, Jev on a tie.
        While an approval is pending only Jev is asked, since only Jev can answer it."""
        jev = asyncio.create_task(self._jev_route(state))
        hedge: Optional[asyncio.Task] = None
        try:
            pending = {jev}
            if self.fallback is not None and not state.pending_approval:
                await asyncio.wait({jev}, timeout=self.fallback_after_s)
                if not jev.done() or jev.result() is None:
                    if not jev.done():
                        logger.info(
                            f"Jev: no answer after {self.fallback_after_s}s, "
                            "hedging with the fallback router"
                        )
                    hedge = asyncio.create_task(self._fallback_route(state))
                    pending.add(hedge)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in sorted(done, key=lambda t: t is not jev):
                    if task.result() is not None:
                        return task.result()
            return Route("chat", source="default")
        finally:
            for task in (jev, hedge):
                if task is not None:
                    task.cancel()

    async def _jev_route(self, state: RouterState) -> Optional[Route]:
        try:
            answers = await asyncio.wait_for(self._ask(state), self.timeout_s)
            return self._interpret(answers, state)
        except Exception as e:
            logger.warning(f"Jev routing failed ({type(e).__name__}: {e})")
            return None

    async def _fallback_route(self, state: RouterState) -> Optional[Route]:
        try:
            return await asyncio.wait_for(
                self.fallback.decide(state), self.fallback.timeout_s
            )
        except Exception as e:
            logger.warning(f"Fallback router failed ({type(e).__name__}: {e})")
            return None

    def questions(self, state: RouterState) -> Dict[str, Any]:
        questions: Dict[str, Any] = {"route": ROUTE_Q}
        if state.active_tasks:
            questions["followup_action"] = FOLLOWUP_ACTION_Q
            if len(state.active_tasks) > 1:
                questions["followup_task"] = {
                    "type": "choice",
                    "instructions": FOLLOWUP_TASK_INSTRUCTIONS,
                    "criteria": {tid: line for tid, line in state.active_tasks},
                }
        if state.pending_offer:
            questions["accepts_offer"] = OFFER_Q
        if state.pending_approval:
            questions["approval"] = APPROVAL_Q
        return questions

    def jev_state(self, state: RouterState) -> Dict[str, str]:
        jev_state = {
            "recent_conversation": state.recent_conversation or "none",
            "active_tasks": _task_lines(state),
            "latest_user_message": state.latest_user_message,
        }
        if state.pending_offer:
            jev_state["pending_offer"] = state.pending_offer
        if state.pending_approval:
            jev_state["pending_approval"] = state.pending_approval
        return jev_state

    async def _ask(self, state: RouterState) -> Dict[str, Any]:
        body = {
            "model": self.model,
            "state": self.jev_state(state),
            "questions": self.questions(state),
        }
        response = await self._get_client().post(
            self.url, json=body, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        answers = response.json().get("answers") or {}
        if not isinstance(answers.get("route"), dict):
            raise ValueError("Jev returned no route answer")
        return answers

    def _interpret(self, answers: Dict[str, Any], state: RouterState) -> Route:
        route = answers["route"]
        choice = route.get("choice")
        probabilities = route.get("probabilities") or {}
        p_task = float(
            probabilities.get("new_task", 1.0 if choice == "new_task" else 0.0)
        )
        if state.pending_approval:
            answer = answers.get("approval") or {}
            verdict = answer.get("choice")
            p = float((answer.get("probabilities") or {}).get(verdict, 0.0))
            if verdict == "deny" or (verdict == "approve" and p >= APPROVE_MIN_P):
                action = "once" if verdict == "approve" else "deny"
                return Route("approval", p_task=p_task, action=action)
        if state.pending_offer:
            accepts = (answers.get("accepts_offer") or {}).get("noul")
            if accepts is not None and float(accepts) >= 0.5:
                return Route("accept_offer", p_task=p_task)
        if choice == "followup" and state.active_tasks:
            action = (answers.get("followup_action") or {}).get("choice")
            ids = [tid for tid, _ in state.active_tasks]
            task_id = (
                ids[0]
                if len(ids) == 1
                else (answers.get("followup_task") or {}).get("choice")
            )
            if task_id not in ids:
                task_id = ids[-1]
            return Route(
                "followup",
                p_task=p_task,
                action=action if action in FOLLOWUP_ACTIONS else "status",
                task_id=task_id,
            )
        if self.unsure_low <= p_task <= self.unsure_high:
            return Route("unsure", p_task=p_task)
        if choice == "new_task":
            return Route("new_task", p_task=p_task)
        return Route("chat", p_task=p_task)
