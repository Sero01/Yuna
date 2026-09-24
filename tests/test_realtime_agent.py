"""RealtimeAgent turn flows with a fake router/talker and a fake Hermes.

Run directly:  .venv/Scripts/python.exe tests/test_realtime_agent.py
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from _harness import run_module  # noqa: E402
from realtime_fakes import (  # noqa: E402
    FakeHermes,
    FakeLive2D,
    template_tts_preprocessor_config,
    wait_until,
)
from src.open_llm_vtuber.agent.agents.realtime.ack_pool import Ack  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.context import ContextBuilder  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.loop_client import LoopBoundClient  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.prompts import (  # noqa: E402
    ACK_INSTRUCTION,
    APPROVAL_EXPIRED_INSTRUCTION,
    APPROVAL_REQUEST_INSTRUCTION,
    APPROVED_INSTRUCTION,
    CANCEL_INSTRUCTION,
    DENIED_INSTRUCTION,
    CHANGE_INSTRUCTION,
    FALLBACK_LINE,
    TASK_STATE_HEADER,
    UNSURE_INSTRUCTION,
)
from src.open_llm_vtuber.agent.agents.realtime.realtime_agent import RealtimeAgent  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.router import Route  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.talker import TalkerError  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.tasks import TaskManager  # noqa: E402
from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource  # noqa: E402
from src.open_llm_vtuber.agent.output_types import AudioOutput, SentenceOutput  # noqa: E402


class FakeRouter:
    def __init__(self, *routes, delay=0.0, talker=None):
        self.routes = list(routes)
        self.delay = delay
        self.talker = talker
        self.states = []
        self.talker_calls_at_decision = []

    async def decide(self, state):
        self.states.append(state)
        await asyncio.sleep(self.delay)
        if self.talker is not None:
            self.talker_calls_at_decision.append(len(self.talker.calls))
        return self.routes.pop(0) if self.routes else Route("chat")


class FakeTalker:
    def __init__(
        self,
        reply="Hmph, hello there, dummy.",
        delay=0.0,
        fail=False,
        ack="Fine, on it.",
    ):
        self.reply = reply
        self.delay = delay
        self.fail = fail
        self.ack = ack
        self.calls = []
        self.completions = []
        self.cancelled = 0

    async def stream(self, messages, max_tokens=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if self.fail:
            raise TalkerError("both providers down")
        finished = False
        try:
            for word in self.reply.split(" "):
                await asyncio.sleep(self.delay)
                yield word + " "
            finished = True
        finally:
            if not finished:
                self.cancelled += 1

    async def complete(self, messages, max_tokens=60):
        self.completions.append(messages)
        return self.ack


TTS_PREPROCESSOR = template_tts_preprocessor_config()


def trailing(call):
    last = call["messages"][-1]
    return last["content"] if last["role"] == "system" else None


def make_agent(hermes=None, router=None, talker=None, **kw):
    hermes = hermes or FakeHermes()
    get = LoopBoundClient(transport=httpx.MockTransport(hermes.handler)).get
    tasks = TaskManager(get, "http://hermes", "hermes-key", poll_interval_s=0.01)
    agent = RealtimeAgent(
        router=router or FakeRouter(),
        talker=talker or FakeTalker(),
        tasks=tasks,
        context=ContextBuilder("You are Yuna."),
        live2d_model=FakeLive2D(),
        tts_preprocessor_config=TTS_PREPROCESSOR,
        **kw,
    )
    return agent, hermes


def user_turn(text, **metadata):
    return BatchInput(
        texts=[TextData(source=TextSource.INPUT, content=text, from_name="Sam")],
        metadata=metadata or None,
    )


async def collect(agent, batch):
    outputs = [out async for out in agent.chat(batch)]
    await asyncio.sleep(0.01)  # let cancelled background streams finish unwinding
    return outputs


def spoken(outputs):
    return " ".join(
        o.display_text.text if isinstance(o, SentenceOutput) else o.transcript
        for o in outputs
    )


async def test_chat_turn_releases_the_speculative_reply():
    talker = FakeTalker(reply="Hmph, hello there, dummy.")
    router = FakeRouter(Route("chat", p_task=0.02), delay=0.05, talker=talker)
    agent, hermes = make_agent(router=router, talker=talker)
    outputs = await collect(agent, user_turn("hi yuna"))
    assert "hello there" in spoken(outputs), spoken(outputs)
    assert all(isinstance(o, SentenceOutput) for o in outputs)
    assert len(talker.calls) == 1, "chat turns speak the speculative reply"
    assert router.talker_calls_at_decision == [1], (
        "talker starts before routing finishes"
    )
    assert talker.calls[0]["messages"][-1] == {"role": "user", "content": "hi yuna"}
    assert agent.memory.messages[-2:] == [
        {"role": "user", "content": "hi yuna"},
        {"role": "assistant", "content": "Hmph, hello there, dummy."},
    ]
    assert router.states[0].latest_user_message == "hi yuna"
    assert not hermes.calls("POST", "/v1/runs")


async def test_router_sees_recent_conversation_with_names():
    router = FakeRouter(Route("chat"), Route("chat"))
    agent, _ = make_agent(router=router, talker=FakeTalker(reply="Hmph."))
    await collect(agent, user_turn("hey"))
    await collect(agent, user_turn("what's up"))
    assert router.states[1].recent_conversation == "Sam: hey\nYuna: Hmph."


async def test_new_task_discards_speculative_reply_and_plays_prerendered_ack():
    talker = FakeTalker(reply="It is thirty four degrees and sunny.", delay=0.05)
    agent, hermes = make_agent(
        router=FakeRouter(Route("new_task", p_task=0.97)), talker=talker
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ack.wav")
        with open(path, "wb") as f:
            f.write(b"RIFF")
        agent.ack_pool.ready = [Ack("Ugh, fine. On it.", path)]
        agent.ack_pool.size = 1
        outputs = await collect(agent, user_turn("what's the weather in delhi"))
        assert len(outputs) == 1 and isinstance(outputs[0], AudioOutput), outputs
        assert outputs[0].transcript == "Ugh, fine. On it."
        assert outputs[0].audio_path == path
        assert not os.path.exists(path), "played ack audio is deleted"
    assert talker.cancelled == 1, "speculative reply was cancelled"
    start = hermes.calls("POST", "/v1/runs")[0][2]
    assert start["input"] == "what's the weather in delhi"
    assert agent.memory.messages[-1] == {
        "role": "assistant",
        "content": "Ugh, fine. On it.",
    }
    assert "thirty four" not in str(agent.memory.messages)


async def test_new_task_without_ready_ack_generates_one_without_user_words():
    talker = FakeTalker(ack="Fine, on it.")
    agent, _ = make_agent(
        router=FakeRouter(Route("new_task", p_task=0.9)), talker=talker
    )
    agent.ack_pool.size = 0
    outputs = await collect(agent, user_turn("remind me to call mom at 7"))
    assert "Fine, on it." in spoken(outputs), spoken(outputs)
    ack_messages = talker.completions[-1]
    assert "call mom" not in str(ack_messages), (
        "the ack prompt never sees the user's words"
    )
    assert ack_messages[-1] == {"role": "system", "content": ACK_INSTRUCTION}


async def test_prerendered_acks_use_the_tts_engine():
    class FakeTTS:
        def __init__(self, folder):
            self.folder = folder

        async def async_generate_audio(self, text, file_name_no_ext=None):
            path = os.path.join(self.folder, f"{file_name_no_ext}.wav")
            with open(path, "wb") as f:
                f.write(b"RIFF")
            return path

    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = make_agent()
        agent.set_tts_engine(FakeTTS(tmp))
        agent.ack_pool.fill()
        await wait_until(lambda: len(agent.ack_pool.ready) == 2)
        assert all(
            a.audio_path and os.path.exists(a.audio_path) for a in agent.ack_pool.ready
        )
        assert agent.ack_pool.ready[0].text == "Fine, on it."
        agent.ack_pool.clear()


async def test_prerendered_acks_say_hmph_as_a_word():
    class RecordingTTS:
        def __init__(self, folder):
            self.folder = folder
            self.texts = []

        async def async_generate_audio(self, text, file_name_no_ext=None):
            self.texts.append(text)
            path = os.path.join(self.folder, f"{file_name_no_ext}.wav")
            with open(path, "wb") as f:
                f.write(b"RIFF")
            return path

    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = make_agent(talker=FakeTalker(ack="Hmph, fine. On it."))
        tts = RecordingTTS(tmp)
        agent.set_tts_engine(tts)
        agent.ack_pool.fill()
        await wait_until(lambda: len(agent.ack_pool.ready) == 2)
        assert tts.texts == ["Humph, fine. On it."] * 2, tts.texts
        assert agent.ack_pool.ready[0].text == "Hmph, fine. On it.", (
            "the subtitle keeps the original spelling"
        )
        agent.ack_pool.clear()


async def test_spoken_hmph_keeps_its_subtitle():
    agent, _ = make_agent(talker=FakeTalker(reply="Hmph, hello there, dummy."))
    outputs = await collect(agent, user_turn("hi yuna"))
    display = " ".join(o.display_text.text for o in outputs)
    tts = " ".join(o.tts_text for o in outputs)
    assert "Hmph" in display and "Humph" not in display, display
    assert "Humph" in tts and "Hmph" not in tts, tts


async def test_failed_task_start_explains_instead_of_acking():
    hermes = FakeHermes(start_status=500)
    agent, _ = make_agent(
        hermes=hermes, router=FakeRouter(Route("new_task", p_task=0.9))
    )
    outputs = await collect(agent, user_turn("check my email"))
    assert not any(isinstance(o, AudioOutput) for o in outputs)
    instruction = trailing(agent.talker.calls[-1])
    assert "couldn't start (Hermes refused the task (HTTP 500))" in instruction, (
        instruction
    )


async def test_unsure_asks_and_accepting_the_offer_starts_the_task():
    router = FakeRouter(Route("unsure", p_task=0.5), Route("accept_offer", p_task=0.3))
    agent, hermes = make_agent(
        router=router, talker=FakeTalker(reply="Want me to look it up?")
    )
    await collect(agent, user_turn("is the new spiderman movie any good"))
    assert trailing(agent.talker.calls[-1]) == UNSURE_INSTRUCTION
    assert agent.pending_offer == "is the new spiderman movie any good"
    assert not hermes.calls("POST", "/v1/runs")
    await collect(agent, user_turn("yeah go ahead"))
    assert router.states[1].pending_offer == "is the new spiderman movie any good"
    start = hermes.calls("POST", "/v1/runs")[0][2]
    assert "spiderman" in start["input"] and "yeah go ahead" in start["input"], start
    assert agent.pending_offer is None


async def test_offer_is_dropped_when_the_user_moves_on():
    router = FakeRouter(Route("unsure", p_task=0.5), Route("chat", p_task=0.1))
    agent, hermes = make_agent(router=router)
    await collect(agent, user_turn("is the new spiderman movie any good"))
    await collect(agent, user_turn("anyway how was your day"))
    assert agent.pending_offer is None and not hermes.calls("POST", "/v1/runs")


async def test_followup_cancel_stops_the_run():
    agent, hermes = make_agent(
        router=FakeRouter(Route("followup", action="cancel", task_id="t1"))
    )
    rec = await agent.tasks.start("find flights to goa", [])
    await collect(agent, user_turn("actually forget the flights"))
    assert rec.status == "cancelled"
    assert hermes.calls("POST", "/v1/runs/run_1/stop")
    assert trailing(agent.talker.calls[-1]) == CANCEL_INSTRUCTION.format(
        request="find flights to goa"
    )
    assert agent.router.states[0].active_tasks[0][0] == "t1"


async def test_followup_change_steers_the_run():
    agent, hermes = make_agent(
        router=FakeRouter(Route("followup", action="change", task_id="t1"))
    )
    await agent.tasks.start("find flights to goa", [])
    await collect(agent, user_turn("only ones under five thousand"))
    assert hermes.calls("POST", "/steer")[0][2] == {
        "input": "only ones under five thousand"
    }
    assert trailing(agent.talker.calls[-1]) == CHANGE_INSTRUCTION.format(
        request="find flights to goa"
    )
    assert len(hermes.calls("POST", "/v1/runs")) == 1


async def test_followup_change_restarts_when_steering_is_refused():
    hermes = FakeHermes(steer_accepted=False)
    agent, _ = make_agent(
        hermes=hermes,
        router=FakeRouter(Route("followup", action="change", task_id="t1")),
    )
    first = await agent.tasks.start("find flights to goa", [])
    await collect(agent, user_turn("only ones under five thousand"))
    assert first.status == "cancelled"
    runs = hermes.calls("POST", "/v1/runs")
    assert (
        len(runs) == 2
        and "update from the user: only ones under five thousand" in runs[1][2]["input"]
    )


async def test_followup_status_on_a_finished_task_speaks_and_marks_it_told():
    talker = FakeTalker(reply="Thirty four degrees, happy?")
    agent, hermes = make_agent(
        router=FakeRouter(Route("followup", action="status", task_id="t1")),
        talker=talker,
    )
    rec = await agent.tasks.start("weather in delhi", [])
    hermes.emit("run_1", event="run.completed", output="34 degrees and humid")
    await wait_until(lambda: rec.status == "done")
    outputs = await collect(agent, user_turn("so what did you find"))
    assert "Thirty four" in spoken(outputs)
    assert len(talker.calls) == 1, "status answers come from the speculative reply"
    task_state = talker.calls[0]["messages"][-2]
    assert task_state["role"] == "system" and task_state["content"].startswith(
        TASK_STATE_HEADER
    )
    assert "34 degrees and humid" in task_state["content"]
    assert rec.told


async def test_task_result_turn_announces_untold_results():
    talker = FakeTalker(reply="It's thirty four degrees, dummy.")
    agent, hermes = make_agent(talker=talker)
    notified = []
    agent.set_task_listener(lambda: notified.append(True))
    rec = await agent.tasks.start("weather in delhi", [])
    hermes.emit("run_1", event="run.completed", output="34 degrees and humid")
    await wait_until(lambda: notified)
    assert agent.has_untold_results()
    outputs = await collect(
        agent, user_turn("", task_result=True, skip_memory=True, skip_history=True)
    )
    assert "thirty four" in spoken(outputs)
    instruction = trailing(talker.calls[-1])
    assert "weather in delhi: 34 degrees and humid" in instruction, instruction
    assert rec.told and not agent.has_untold_results()
    assert agent.memory.messages[-1] == {
        "role": "assistant",
        "content": "It's thirty four degrees, dummy.",
    }
    assert not any(m["role"] == "user" for m in agent.memory.messages)
    agent.set_task_listener(None)


async def test_task_result_turn_with_nothing_to_tell_is_silent():
    agent, _ = make_agent()
    outputs = await collect(agent, user_turn("", task_result=True, skip_memory=True))
    assert outputs == [] and agent.talker.calls == []


async def test_failed_results_are_announced_as_failures():
    agent, hermes = make_agent()
    rec = await agent.tasks.start("send the email", [])
    hermes.emit("run_1", event="run.failed", error="gmail auth expired")
    await wait_until(lambda: rec.status == "failed")
    await collect(agent, user_turn("", task_result=True, skip_memory=True))
    assert "send the email: failed (gmail auth expired)" in trailing(
        agent.talker.calls[-1]
    )


async def test_listener_fires_on_connect_when_results_are_waiting():
    agent, hermes = make_agent()
    rec = await agent.tasks.start("weather", [])
    hermes.emit("run_1", event="run.completed", output="sunny")
    await wait_until(lambda: rec.status == "done")
    notified = []
    listener = lambda: notified.append(1)  # noqa: E731
    agent.set_task_listener(listener)
    assert notified == [1]
    agent.remove_task_listener(lambda: None)
    agent.remove_task_listener(listener)
    rec2 = await agent.tasks.start("more", [])
    hermes.emit("run_2", event="run.completed", output="x")
    await wait_until(lambda: rec2.status == "done")
    assert notified == [1], "removed listeners are not called"


APPROVAL_EVENT = {
    "event": "approval.request",
    "command": "hermes gateway restart",
    "description": "restart a system service",
    "request_id": "req_1",
}


async def waiting_task(agent, hermes):
    rec = await agent.tasks.start("restart the gateway", [])
    hermes.emit("run_1", **APPROVAL_EVENT)
    await wait_until(lambda: rec.approval is not None)
    return rec


async def test_approval_request_is_announced_as_a_question():
    talker = FakeTalker(reply="Hermes wants to restart the gateway. Allow it once?")
    agent, hermes = make_agent(talker=talker)
    notified = []
    agent.set_task_listener(lambda: notified.append(True))
    await waiting_task(agent, hermes)
    assert notified and agent.has_untold_results()
    outputs = await collect(
        agent, user_turn("", task_result=True, skip_memory=True, skip_history=True)
    )
    assert "Allow it once?" in spoken(outputs)
    assert trailing(talker.calls[-1]) == APPROVAL_REQUEST_INSTRUCTION.format(
        approvals="- restart the gateway: it wants to restart a system service "
        "(command: hermes gateway restart)"
    )
    assert not agent.has_untold_results(), "asked once, not on every quiet gap"
    agent.set_task_listener(None)


async def test_yes_allows_the_pending_command_once():
    router = FakeRouter(Route("approval", action="once"))
    agent, hermes = make_agent(router=router)
    rec = await waiting_task(agent, hermes)
    await collect(agent, user_turn("yeah go ahead"))
    assert router.states[0].pending_approval == (
        "restart a system service: `hermes gateway restart`"
    )
    assert hermes.calls("POST", "/v1/runs/run_1/approval")[0][2] == {
        "choice": "once",
        "request_id": "req_1",
    }
    assert trailing(agent.talker.calls[-1]) == APPROVED_INSTRUCTION.format(
        description="restart a system service"
    )
    assert rec.approval is None


async def test_no_denies_the_pending_command():
    agent, hermes = make_agent(router=FakeRouter(Route("approval", action="deny")))
    await waiting_task(agent, hermes)
    await collect(agent, user_turn("no, don't"))
    assert hermes.calls("POST", "/approval")[0][2]["choice"] == "deny"
    assert trailing(agent.talker.calls[-1]) == DENIED_INSTRUCTION.format(
        description="restart a system service"
    )


async def test_answer_after_hermes_stopped_waiting_says_nothing_ran():
    agent, hermes = make_agent(router=FakeRouter(Route("approval", action="once")))
    await waiting_task(agent, hermes)
    hermes.approval_pending = False
    await collect(agent, user_turn("yes do it"))
    assert trailing(agent.talker.calls[-1]) == APPROVAL_EXPIRED_INSTRUCTION.format(
        description="restart a system service"
    )


async def test_no_pending_approval_means_no_approval_state_for_the_router():
    router = FakeRouter(Route("chat"))
    agent, _ = make_agent(router=router)
    await agent.tasks.start("restart the gateway", [])
    await collect(agent, user_turn("yes"))
    assert router.states[0].pending_approval is None


async def test_talker_failure_speaks_the_fallback_line():
    agent, _ = make_agent(
        router=FakeRouter(Route("chat")), talker=FakeTalker(fail=True)
    )
    outputs = await collect(agent, user_turn("hello"))
    assert FALLBACK_LINE in spoken(outputs)
    assert agent.memory.messages[-1] == {"role": "assistant", "content": FALLBACK_LINE}


async def test_unknown_expression_tags_are_stripped():
    agent, _ = make_agent(
        router=FakeRouter(Route("chat")),
        talker=FakeTalker(reply="Fine. [soft] I'll help. [joy]"),
    )
    outputs = await collect(agent, user_turn("help me"))
    text = spoken(outputs)
    assert "[soft]" not in text and "[joy]" in text, text
    assert any(o.actions.expressions == [3] for o in outputs), [
        o.actions for o in outputs
    ]


async def test_interrupt_records_what_was_heard():
    agent, _ = make_agent(
        router=FakeRouter(Route("chat")), talker=FakeTalker(reply="Once upon a time.")
    )
    await collect(agent, user_turn("tell me a story"))
    agent.handle_interrupt("Once upon")
    agent.handle_interrupt("Once upon a")  # only the first call per turn counts
    assert agent.memory.messages[-2:] == [
        {"role": "assistant", "content": "Once upon..."},
        {"role": "system", "content": "[Interrupted by user]"},
    ]


async def test_cancelled_turn_leaves_no_unheard_reply_in_memory():
    talker = FakeTalker(reply="a b c d e f g h", delay=0.05)
    agent, _ = make_agent(router=FakeRouter(Route("chat")), talker=talker)

    async def consume():
        async for _ in agent.chat(user_turn("go on")):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.12)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    agent.handle_interrupt("a b")
    await asyncio.sleep(0.05)
    assert agent.memory.messages[-2:] == [
        {"role": "assistant", "content": "a b..."},
        {"role": "system", "content": "[Interrupted by user]"},
    ], agent.memory.messages
    assert talker.cancelled == 1


async def test_interrupted_announcement_keeps_the_previous_reply():
    talker = FakeTalker(reply="Hmph, fine.")
    agent, hermes = make_agent(router=FakeRouter(Route("chat")), talker=talker)
    await collect(agent, user_turn("hey"))
    rec = await agent.tasks.start("weather in delhi", [])
    hermes.emit("run_1", event="run.completed", output="34 degrees")
    await wait_until(lambda: rec.status == "done")
    talker.reply, talker.delay = "It's thirty four degrees and humid today.", 0.05

    async def announce():
        batch = user_turn("", task_result=True, skip_memory=True, skip_history=True)
        async for _ in agent.chat(batch):
            pass

    task = asyncio.create_task(announce())
    await asyncio.sleep(0.12)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    agent.handle_interrupt("It's thirty")
    assert agent.memory.messages[-3:] == [
        {"role": "assistant", "content": "Hmph, fine."},
        {"role": "assistant", "content": "It's thirty..."},
        {"role": "system", "content": "[Interrupted by user]"},
    ], agent.memory.messages


async def test_interrupt_after_the_reply_finished_replaces_it_with_what_was_heard():
    agent, _ = make_agent(
        router=FakeRouter(Route("chat")), talker=FakeTalker(reply="A long story.")
    )
    await collect(
        agent, user_turn("", task_result=False, proactive_speak=True, skip_memory=True)
    )
    agent.handle_interrupt("A long")  # audio was still playing when the user spoke
    assert agent.memory.messages[-2:] == [
        {"role": "assistant", "content": "A long..."},
        {"role": "system", "content": "[Interrupted by user]"},
    ]


async def test_proactive_turn_skips_routing():
    router = FakeRouter()
    agent, _ = make_agent(router=router)
    await collect(
        agent,
        user_turn(
            "Say something.", proactive_speak=True, skip_memory=True, skip_history=True
        ),
    )
    assert router.states == []
    assert agent.talker.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "Say something.",
    }
    assert not any(m["role"] == "user" for m in agent.memory.messages)


async def test_talker_context_holds_the_last_eight_exchanges():
    agent, _ = make_agent(router=FakeRouter(Route("chat")), max_turns=8)
    for i in range(12):
        agent.memory.add("user", f"u{i}")
        agent.memory.add("assistant", f"a{i}")
    await collect(agent, user_turn("now"))
    messages = agent.talker.calls[0]["messages"]
    assert len(messages) == 1 + 16 + 1, len(messages)
    assert messages[1] == {"role": "user", "content": "u4"}
    assert messages[-1] == {"role": "user", "content": "now"}


def test_factory_builds_a_realtime_agent():
    from src.open_llm_vtuber.agent.agent_factory import AgentFactory

    agent = AgentFactory.create_agent(
        conversation_agent_choice="realtime_agent",
        agent_settings={
            "realtime_agent": {
                "openrouter_api_key": "or-key",
                "hermes_api_key": "hermes-key",
                "max_turns": 6,
                "prerender_acks": False,
            }
        },
        llm_configs={},
        system_prompt="You are Yuna.",
        live2d_model=FakeLive2D(),
        character_name="Yuna",
    )
    assert isinstance(agent, RealtimeAgent)
    assert agent.max_turns == 6 and agent.ai_name == "Yuna"
    assert agent.talker.model == "deepseek/deepseek-v4.1-flash"
    assert agent.router.fallback is not None


if __name__ == "__main__":
    run_module(globals())
