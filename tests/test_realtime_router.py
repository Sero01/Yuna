"""Routing: Jev answers -> Route, unsure band, follow-ups, offers, fallbacks.

Run directly:  .venv/Scripts/python.exe tests/test_realtime_router.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.loop_client import LoopBoundClient  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.router import (  # noqa: E402
    FallbackRouter,
    JevRouter,
    RouterState,
)

JEV_URL = "https://openrouter.ai/api/alpha/decisions"


class FakeOpenRouter:
    def __init__(
        self,
        jev_answers=None,
        jev_status=200,
        jev_delay=0.0,
        fallback_reply='{"route": "chat", "action": null}',
        fallback_status=200,
        fallback_delay=0.0,
    ):
        self.jev_answers = jev_answers or {}
        self.jev_status = jev_status
        self.jev_delay = jev_delay
        self.fallback_reply = fallback_reply
        self.fallback_status = fallback_status
        self.fallback_delay = fallback_delay
        self.jev_bodies = []
        self.fallback_bodies = []
        self.auth = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.auth.append(request.headers.get("authorization"))
        if request.url.path.endswith("/decisions"):
            self.jev_bodies.append(body)
            await asyncio.sleep(self.jev_delay)
            if self.jev_status != 200:
                return httpx.Response(
                    self.jev_status, json={"error": {"message": "down"}}
                )
            return httpx.Response(200, json={"answers": self.jev_answers})
        self.fallback_bodies.append(body)
        await asyncio.sleep(self.fallback_delay)
        if self.fallback_status != 200:
            return httpx.Response(
                self.fallback_status, json={"error": {"message": "down"}}
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": self.fallback_reply}}]}
        )


def route_answer(choice, **probs):
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": probs,
        "confidence": 0.9,
    }


def make_router(fake, with_fallback=True, timeout=0.5, fallback_after=None):
    get = LoopBoundClient(transport=httpx.MockTransport(fake.handler)).get
    fallback = (
        FallbackRouter(
            get,
            "https://openrouter.ai/api/v1",
            "or-key",
            "meta-llama/llama-3.3-70b-instruct",
            "groq",
            timeout_s=0.5,
        )
        if with_fallback
        else None
    )
    return JevRouter(
        get,
        JEV_URL,
        "or-key",
        timeout_s=timeout,
        fallback=fallback,
        fallback_after_s=timeout if fallback_after is None else fallback_after,
    )


def state(text, tasks=(), offer=None, approval=None):
    return RouterState(
        latest_user_message=text,
        recent_conversation="Sam: hey\nYuna: Hmph, what now?",
        active_tasks=list(tasks),
        pending_offer=offer,
        pending_approval=approval,
    )


APPROVAL = "restart a system service: `hermes gateway restart`"
WAITING = [("t1", "t1 [running 40s, paused: waiting ...] restart the gateway")]


def approval_answer(choice, p):
    return {"type": "choice", "choice": choice, "probabilities": {choice: p}}


async def test_clear_yes_to_a_pending_approval_allows_it_once():
    fake = FakeOpenRouter(
        {
            "route": route_answer("followup", chat=0.3, new_task=0.0, followup=0.7),
            "followup_action": {"choice": "status"},
            "approval": approval_answer("approve", 0.93),
        }
    )
    route = await make_router(fake).decide(
        state("yeah go ahead", WAITING, approval=APPROVAL)
    )
    assert (route.kind, route.action) == ("approval", "once"), route
    body = fake.jev_bodies[0]
    assert "approval" in body["questions"]
    assert body["state"]["pending_approval"] == APPROVAL


async def test_no_to_a_pending_approval_denies_it():
    fake = FakeOpenRouter(
        {
            "route": route_answer("chat", chat=0.9, new_task=0.1),
            "approval": approval_answer("deny", 0.6),
        }
    )
    route = await make_router(fake).decide(
        state("no, don't", WAITING, approval=APPROVAL)
    )
    assert (route.kind, route.action) == ("approval", "deny"), route


async def test_a_doubtful_yes_does_not_approve():
    fake = FakeOpenRouter(
        {
            "route": route_answer("chat", chat=0.9, new_task=0.1),
            "approval": approval_answer("approve", 0.6),
        }
    )
    route = await make_router(fake).decide(
        state("hmm maybe", WAITING, approval=APPROVAL)
    )
    assert route.kind == "chat", route


async def test_other_talk_during_an_approval_routes_normally():
    fake = FakeOpenRouter(
        {
            "route": route_answer("chat", chat=0.9, new_task=0.1),
            "approval": approval_answer("neither", 0.9),
        }
    )
    route = await make_router(fake).decide(
        state("how many options are there", WAITING, approval=APPROVAL)
    )
    assert route.kind == "chat", route


async def test_no_approval_question_without_a_pending_approval():
    fake = FakeOpenRouter({"route": route_answer("chat", chat=0.9, new_task=0.1)})
    await make_router(fake).decide(state("yes", WAITING))
    assert "approval" not in fake.jev_bodies[0]["questions"]
    assert "pending_approval" not in fake.jev_bodies[0]["state"]


async def test_pending_approval_waits_for_jev_instead_of_hedging():
    fake = FakeOpenRouter(
        {
            "route": route_answer("chat", chat=0.9, new_task=0.1),
            "approval": approval_answer("approve", 0.95),
        },
        jev_delay=0.3,
        fallback_reply='{"route": "chat"}',
    )
    route = await make_router(fake, timeout=1.5, fallback_after=0.1).decide(
        state("yes do it", WAITING, approval=APPROVAL)
    )
    assert (route.kind, route.source) == ("approval", "jev"), route
    assert fake.fallback_bodies == [], "the fallback router can't answer approvals"


async def test_chat_route_and_request_shape():
    fake = FakeOpenRouter(
        {"route": route_answer("chat", chat=0.97, new_task=0.03, followup=0)}
    )
    route = await make_router(fake).decide(state("how many calories in a banana"))
    assert route.kind == "chat" and route.source == "jev", route
    body = fake.jev_bodies[0]
    assert body["model"] == "typesafe/jev-1.13"
    assert set(body["questions"]) == {"route"}, body["questions"]
    assert body["state"]["latest_user_message"] == "how many calories in a banana"
    assert body["state"]["active_tasks"] == "none"
    assert "pending_offer" not in body["state"]
    assert fake.auth[0] == "Bearer or-key"


async def test_new_task_route():
    fake = FakeOpenRouter(
        {"route": route_answer("new_task", chat=0.05, new_task=0.95, followup=0)}
    )
    route = await make_router(fake).decide(state("what's the weather in delhi"))
    assert route.kind == "new_task" and route.p_task == 0.95, route


async def test_middling_task_probability_is_unsure():
    for choice, p in (("new_task", 0.55), ("chat", 0.4)):
        fake = FakeOpenRouter(
            {"route": route_answer(choice, chat=1 - p, new_task=p, followup=0)}
        )
        route = await make_router(fake).decide(
            state("is the new spiderman movie any good")
        )
        assert route.kind == "unsure", (choice, p, route)


async def test_followup_with_one_task_targets_it():
    fake = FakeOpenRouter(
        {
            "route": route_answer("followup", chat=0.1, new_task=0.0, followup=0.9),
            "followup_action": {"type": "choice", "choice": "cancel"},
        }
    )
    tasks = [("t1", "t1 [running 12s] find flights to goa")]
    route = await make_router(fake).decide(state("forget the flights", tasks))
    assert (route.kind, route.action, route.task_id) == ("followup", "cancel", "t1"), (
        route
    )
    assert set(fake.jev_bodies[0]["questions"]) == {"route", "followup_action"}
    assert "find flights to goa" in fake.jev_bodies[0]["state"]["active_tasks"]


async def test_followup_with_several_tasks_uses_jevs_pick_or_the_latest():
    tasks = [("t1", "t1 [running 30s] find flights"), ("t2", "t2 [running 5s] weather")]
    fake = FakeOpenRouter(
        {
            "route": route_answer("followup", followup=0.9, chat=0.1, new_task=0),
            "followup_action": {"choice": "status"},
            "followup_task": {"choice": "t1"},
        }
    )
    route = await make_router(fake).decide(state("how are the flights going", tasks))
    assert route.task_id == "t1" and route.action == "status", route
    q = fake.jev_bodies[0]["questions"]["followup_task"]
    assert set(q["criteria"]) == {"t1", "t2"}

    fake.jev_answers["followup_task"] = {"choice": "t9"}
    route = await make_router(fake).decide(state("how is it going", tasks))
    assert route.task_id == "t2", route


async def test_unknown_followup_action_defaults_to_status():
    fake = FakeOpenRouter(
        {
            "route": route_answer("followup", followup=0.9, chat=0.1, new_task=0),
            "followup_action": {"choice": "dance"},
        }
    )
    route = await make_router(fake).decide(state("and?", [("t1", "t1 [running 3s] x")]))
    assert route.action == "status", route


async def test_accepted_offer():
    fake = FakeOpenRouter(
        {
            "route": route_answer("chat", chat=0.7, new_task=0.3, followup=0),
            "accepts_offer": {"type": "noul", "noul": 0.86},
        }
    )
    route = await make_router(fake).decide(
        state("yeah go ahead", offer="is spiderman good")
    )
    assert route.kind == "accept_offer", route
    body = fake.jev_bodies[0]
    assert "accepts_offer" in body["questions"]
    assert body["state"]["pending_offer"] == "is spiderman good"


async def test_declined_offer_routes_normally():
    fake = FakeOpenRouter(
        {
            "route": route_answer("chat", chat=0.95, new_task=0.05, followup=0),
            "accepts_offer": {"type": "noul", "noul": 0.1},
        }
    )
    route = await make_router(fake).decide(
        state("nah never mind", offer="is spiderman good")
    )
    assert route.kind == "chat", route


async def test_jev_error_uses_fallback():
    fake = FakeOpenRouter(
        jev_status=500, fallback_reply='Sure: {"route": "new_task", "action": null}'
    )
    route = await make_router(fake).decide(state("remind me to call mom at 7"))
    assert (route.kind, route.source) == ("new_task", "fallback"), route
    fb = fake.fallback_bodies[0]
    assert fb["model"] == "meta-llama/llama-3.3-70b-instruct"
    assert fb["provider"] == {"order": ["groq"], "allow_fallbacks": True}
    assert "remind me to call mom at 7" in fb["messages"][0]["content"]


async def test_jev_timeout_uses_fallback():
    fake = FakeOpenRouter(jev_delay=1.0, fallback_reply='{"route": "chat"}')
    started = asyncio.get_running_loop().time()
    route = await make_router(fake, timeout=0.1).decide(state("hello"))
    assert route.source == "fallback", route
    assert asyncio.get_running_loop().time() - started < 0.8


def elapsed_since(started):
    return asyncio.get_running_loop().time() - started


async def test_slow_jev_is_hedged_with_the_fallback_before_its_timeout():
    fake = FakeOpenRouter(jev_delay=1.0, fallback_reply='{"route": "new_task"}')
    started = asyncio.get_running_loop().time()
    route = await make_router(fake, timeout=1.5, fallback_after=0.1).decide(
        state("what's the weather in delhi")
    )
    assert (route.kind, route.source) == ("new_task", "fallback"), route
    assert elapsed_since(started) < 0.5, "should not wait for Jev's timeout"


async def test_fast_jev_never_calls_the_fallback():
    fake = FakeOpenRouter({"route": route_answer("chat", chat=0.9, new_task=0.1)})
    route = await make_router(fake, timeout=1.5, fallback_after=0.3).decide(
        state("hello")
    )
    await asyncio.sleep(0.4)
    assert route.source == "jev", route
    assert fake.fallback_bodies == [], "no hedge request when Jev answers in time"


async def test_jev_still_wins_if_it_answers_while_the_hedge_is_in_flight():
    fake = FakeOpenRouter(
        {"route": route_answer("new_task", chat=0.05, new_task=0.95)},
        jev_delay=0.2,
        fallback_reply='{"route": "chat"}',
        fallback_delay=0.4,
    )
    route = await make_router(fake, timeout=1.5, fallback_after=0.1).decide(
        state("remind me to call mom")
    )
    assert (route.kind, route.source) == ("new_task", "jev"), route
    assert len(fake.fallback_bodies) == 1


async def test_jev_error_starts_the_fallback_without_waiting_for_the_hedge_delay():
    fake = FakeOpenRouter(jev_status=500, fallback_reply='{"route": "chat"}')
    started = asyncio.get_running_loop().time()
    route = await make_router(fake, timeout=1.5, fallback_after=1.0).decide(
        state("hello")
    )
    assert route.source == "fallback", route
    assert elapsed_since(started) < 0.5


async def test_fallback_followup_targets_latest_task():
    fake = FakeOpenRouter(
        jev_status=503, fallback_reply='{"route": "followup", "action": "cancel"}'
    )
    tasks = [("t1", "t1 [running] a"), ("t2", "t2 [running] b")]
    route = await make_router(fake).decide(state("stop it", tasks))
    assert (route.kind, route.action, route.task_id) == ("followup", "cancel", "t2"), (
        route
    )


async def test_everything_failing_means_chat():
    fake = FakeOpenRouter(jev_status=500, fallback_status=500)
    route = await make_router(fake).decide(state("hello"))
    assert (route.kind, route.source) == ("chat", "default"), route
    route = await make_router(
        FakeOpenRouter(jev_status=500), with_fallback=False
    ).decide(state("hi"))
    assert (route.kind, route.source) == ("chat", "default"), route


async def test_malformed_jev_answer_uses_fallback():
    fake = FakeOpenRouter({"something": "else"}, fallback_reply='{"route": "chat"}')
    route = await make_router(fake).decide(state("hello"))
    assert route.source == "fallback", route


def test_client_is_recreated_per_loop():
    http = LoopBoundClient()

    async def grab():
        client = http.get()
        assert http.get() is client, "same loop reuses the client"
        return client

    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second, "a new event loop gets a new client"


if __name__ == "__main__":
    run_module(globals())
