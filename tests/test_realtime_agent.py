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
    MEMORY_REVIEW_INSTRUCTIONS,
    SUMMARY_CONDENSE_INSTRUCTIONS,
    SUMMARY_HEADER,
    SUMMARY_INSTRUCTIONS,
    TASK_REMEMBER_HINT,
    TASK_STATE_HEADER,
    UNSURE_INSTRUCTION,
    WORKER_SUMMARY,
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
        summary="They talked about exams.",
        summary_error=None,
        summary_delay=0.0,
        condensed="- Shorter notes.",
        condense_error=None,
    ):
        self.reply = reply
        self.delay = delay
        self.fail = fail
        self.ack = ack
        self.summary = summary
        self.summary_error = summary_error
        self.summary_delay = summary_delay
        self.condensed = condensed
        self.condense_error = condense_error
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
        if messages[0]["content"] == SUMMARY_INSTRUCTIONS:
            await asyncio.sleep(self.summary_delay)
            if self.summary_error is not None:
                raise self.summary_error
            return self.summary
        if messages[0]["content"] == SUMMARY_CONDENSE_INSTRUCTIONS:
            if self.condense_error is not None:
                raise self.condense_error
            return self.condensed
        return self.ack


def summary_requests(talker):
    return [m for m in talker.completions if m[0]["content"] == SUMMARY_INSTRUCTIONS]


def condense_requests(talker):
    return [
        m
        for m in talker.completions
        if m[0]["content"] == SUMMARY_CONDENSE_INSTRUCTIONS
    ]


def fill(agent, exchanges):
    for i in range(exchanges):
        agent.memory.add("user", f"u{i}")
        agent.memory.add("assistant", f"a{i}")


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


def ready_ack(agent, folder, text="Ugh, fine. On it."):
    path = os.path.join(folder, "ack.wav")
    with open(path, "wb") as f:
        f.write(b"RIFF")
    agent.ack_pool.ready = [Ack(text, path)]
    agent.ack_pool.size = 1


async def test_slow_task_start_is_acknowledged_without_waiting():
    hermes = FakeHermes(start_delay=0.5)
    agent, _ = make_agent(
        hermes=hermes, router=FakeRouter(Route("new_task", p_task=0.9))
    )
    with tempfile.TemporaryDirectory() as tmp:
        ready_ack(agent, tmp)
        loop = asyncio.get_running_loop()
        started = loop.time()
        stream = agent.chat(user_turn("check my email"))
        first = await stream.__anext__()
        assert loop.time() - started < 0.3, "the ack doesn't wait for Hermes"
        assert (
            isinstance(first, AudioOutput) and first.transcript == "Ugh, fine. On it."
        )
        rest = [out async for out in stream]
        assert loop.time() - started >= 0.5, "the turn still waits for the start"
    assert rest == [], rest
    assert agent.tasks.active(), "the task was started"


async def test_slow_task_start_that_fails_is_corrected_after_the_ack():
    hermes = FakeHermes(start_delay=0.3, start_status=500)
    agent, _ = make_agent(
        hermes=hermes, router=FakeRouter(Route("new_task", p_task=0.9))
    )
    with tempfile.TemporaryDirectory() as tmp:
        ready_ack(agent, tmp)
        outputs = await collect(agent, user_turn("check my email"))
    assert isinstance(outputs[0], AudioOutput), outputs
    assert any(isinstance(o, SentenceOutput) for o in outputs[1:]), outputs
    instruction = trailing(agent.talker.calls[-1])
    assert "couldn't start (Hermes refused the task (HTTP 500))" in instruction


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


async def test_without_summaries_the_talker_holds_the_last_eight_exchanges():
    agent, _ = make_agent(
        router=FakeRouter(Route("chat")), max_turns=8, summarize_history=False
    )
    fill(agent, 12)
    await collect(agent, user_turn("now"))
    messages = agent.talker.calls[0]["messages"]
    assert len(messages) == 1 + 16 + 1, len(messages)
    assert messages[1] == {"role": "user", "content": "u4"}
    assert messages[-1] == {"role": "user", "content": "now"}
    assert summary_requests(agent.talker) == []


async def test_without_summaries_the_window_trims_in_steps_of_a_fifth():
    agent, _ = make_agent(
        router=FakeRouter(Route("chat")), max_turns=50, summarize_history=False
    )
    fill(agent, 55)
    await collect(agent, user_turn("now"))
    messages = agent.talker.calls[0]["messages"]
    # 55 exchanges, 5 over -> the start jumps a whole step of 10, to u10.
    assert messages[1] == {"role": "user", "content": "u10"}, messages[1]
    assert len(messages) == 1 + 90 + 1


async def test_old_exchanges_are_summarized_in_the_background():
    talker = FakeTalker(reply="Hmph.", summary="They talked about exams.")
    agent, _ = make_agent(
        router=FakeRouter(), talker=talker, max_turns=50, summary_keep_turns=10
    )
    fill(agent, 49)
    await collect(agent, user_turn("u49"))
    assert summary_requests(talker) == []  # 49 finished exchanges: not yet

    await collect(agent, user_turn("now"))
    # This turn still sees everything; the summary is made off the critical path.
    assert talker.calls[1]["messages"][1] == {"role": "user", "content": "u0"}
    await wait_until(lambda: agent.memory.summary)
    request = summary_requests(talker)[0][1]["content"]
    assert "Sam: u0\nYuna: a0" in request and "Sam: u39\nYuna: a39" in request
    assert "u40" not in request

    await collect(agent, user_turn("and now"))
    messages = talker.calls[2]["messages"]
    assert messages[1] == {
        "role": "system",
        "content": f"{SUMMARY_HEADER}\nThey talked about exams.",
    }
    assert messages[2] == {"role": "user", "content": "u40"}
    assert messages[-1] == {"role": "user", "content": "and now"}
    assert len(summary_requests(talker)) == 1


async def test_new_notes_are_added_after_the_old_ones_word_for_word():
    talker = FakeTalker(summary="- They planned a trip to Goa.")
    agent, _ = make_agent(router=FakeRouter(), talker=talker, max_turns=50)
    fill(agent, 90)
    agent.memory.set_summary("- They talked about exams.", 40)
    await collect(agent, user_turn("now"))
    await wait_until(lambda: agent.memory.summarized_exchanges == 80)
    assert agent.memory.summary == (
        "- They talked about exams.\n- They planned a trip to Goa."
    )
    request = summary_requests(talker)[0][1]["content"]
    assert "- They talked about exams." in request  # so the notes don't repeat it
    assert "Sam: u40\nYuna: a40" in request and "u80" not in request
    assert condense_requests(talker) == []


async def test_notes_are_condensed_once_they_run_long():
    talker = FakeTalker(summary="- They planned a trip to Goa.")
    agent, _ = make_agent(router=FakeRouter(), talker=talker, max_turns=50)
    fill(agent, 90)
    old = "\n".join(f"- Topic {i} went on for a while." for i in range(60))
    agent.memory.set_summary(old, 40)
    await collect(agent, user_turn("now"))
    await wait_until(lambda: agent.memory.summarized_exchanges == 80)
    assert agent.memory.summary == "- Shorter notes."
    condense = condense_requests(talker)[0][1]["content"]
    assert condense == f"{old}\n- They planned a trip to Goa."


async def test_long_notes_stay_whole_if_condensing_fails():
    talker = FakeTalker(
        summary="- They planned a trip to Goa.",
        condense_error=TalkerError("both providers down"),
    )
    agent, _ = make_agent(router=FakeRouter(), talker=talker, max_turns=50)
    fill(agent, 90)
    old = "\n".join(f"- Topic {i} went on for a while." for i in range(60))
    agent.memory.set_summary(old, 40)
    await collect(agent, user_turn("now"))
    await wait_until(lambda: agent.memory.summarized_exchanges == 80)
    assert agent.memory.summary == f"{old}\n- They planned a trip to Goa."


async def test_a_failed_summary_keeps_the_whole_window_and_is_retried():
    talker = FakeTalker(summary_error=TalkerError("both providers down"))
    agent, _ = make_agent(router=FakeRouter(), talker=talker, max_turns=50)
    fill(agent, 50)
    await collect(agent, user_turn("now"))
    await wait_until(lambda: summary_requests(talker))
    await asyncio.sleep(0.01)
    await collect(agent, user_turn("again"))
    assert talker.calls[1]["messages"][1] == {"role": "user", "content": "u0"}
    await wait_until(lambda: len(summary_requests(talker)) == 2)
    assert agent.memory.summary == ""


async def test_loading_another_history_drops_an_unfinished_summary():
    from src.open_llm_vtuber.agent.agents.realtime import context as context_module

    talker = FakeTalker(summary_delay=0.2)
    agent, _ = make_agent(talker=talker, max_turns=50, share_transcripts=False)
    fill(agent, 50)
    await collect(agent, user_turn("now"))
    await wait_until(lambda: summary_requests(talker))
    original = context_module.get_history
    context_module.get_history = lambda conf, uid: []
    try:
        agent.set_memory_from_history("mao_pro_001", "no_such_history")
    finally:
        context_module.get_history = original
    await asyncio.sleep(0.3)
    assert agent.memory.summary == ""


async def test_tasks_get_the_summary_too():
    router = FakeRouter(Route("new_task", p_task=0.95))
    agent, hermes = make_agent(router=router, max_turns=50)
    fill(agent, 20)
    agent.memory.set_summary("They picked a ramen place.", 10)
    await collect(agent, user_turn("book it"))
    instructions = hermes.calls("POST", "/v1/runs")[0][2]["instructions"]
    assert WORKER_SUMMARY.format(summary="They picked a ramen place.") in instructions
    agent.tasks.close()


async def test_workers_get_fewer_exchanges_than_the_talker():
    router = FakeRouter(Route("new_task", p_task=0.95))
    agent, hermes = make_agent(router=router, max_turns=50, worker_max_turns=8)
    for i in range(20):
        agent.memory.add("user", f"u{i}")
        agent.memory.add("assistant", f"a{i}")
    await collect(agent, user_turn("weather in delhi"))
    history = hermes.calls("POST", "/v1/runs")[0][2]["conversation_history"]
    assert len(history) == 16 and history[0]["content"] == "u12", history[:2]
    agent.tasks.close()


async def test_flagged_chat_turn_asks_hermes_whether_to_remember_it():
    router = FakeRouter(Route("chat"), Route("chat", p_remember=0.8))
    agent, hermes = make_agent(router=router, talker=FakeTalker(reply="What now?"))
    await collect(agent, user_turn("hey"))
    outputs = await collect(agent, user_turn("my sister's name is ayesha"))
    assert "What now?" in spoken(outputs)
    await wait_until(lambda: hermes.calls("POST", "/v1/runs"))
    runs = hermes.calls("POST", "/v1/runs")
    assert len(runs) == 1
    body = runs[0][2]
    assert body["instructions"] == MEMORY_REVIEW_INSTRUCTIONS
    assert "Sam: hey\nYuna: What now?" in body["input"]
    assert body["input"].endswith("Sam: my sister's name is ayesha")
    assert agent.tasks.tasks == {}
    agent.tasks.close()


async def test_unflagged_turns_leave_memory_alone():
    router = FakeRouter(Route("chat", p_remember=0.3), Route("chat"))
    agent, hermes = make_agent(router=router)
    await collect(agent, user_turn("ugh i'm so tired today"))
    await collect(agent, user_turn("tell me a joke"))
    await asyncio.sleep(0.05)
    assert hermes.calls("POST", "/v1/runs") == []


async def test_remember_threshold_zero_turns_the_check_off():
    router = FakeRouter(Route("chat", p_remember=0.99))
    agent, hermes = make_agent(router=router, remember_threshold=0)
    await collect(agent, user_turn("i'm allergic to cashews"))
    await asyncio.sleep(0.05)
    assert hermes.calls("POST", "/v1/runs") == []


async def test_flagged_task_turn_passes_the_hint_instead_of_a_second_run():
    router = FakeRouter(Route("new_task", p_task=0.95, p_remember=0.9))
    agent, hermes = make_agent(router=router)
    agent.ack_pool.take = lambda: Ack("On it.")
    await collect(agent, user_turn("i moved to koramangala, find gyms near me"))
    await asyncio.sleep(0.05)
    runs = hermes.calls("POST", "/v1/runs")
    assert len(runs) == 1
    assert runs[0][2]["instructions"].endswith(TASK_REMEMBER_HINT)
    assert runs[0][2]["input"] == "i moved to koramangala, find gyms near me"
    agent.tasks.close()


async def test_router_gets_the_known_facts():
    with tempfile.TemporaryDirectory() as tmp:
        user = os.path.join(tmp, "USER.md")
        with open(user, "w", encoding="utf-8") as f:
            f.write("Likes Pokemon.")
        router = FakeRouter(Route("chat"))
        agent, _ = make_agent(router=router)
        agent.context = ContextBuilder("You are Yuna.", "", user)
        await collect(agent, user_turn("hey"))
    assert router.states[0].known_facts == "- Likes Pokemon."


def test_loading_a_history_points_tasks_at_its_transcript():
    from src.open_llm_vtuber.agent.agents.realtime import context as context_module

    agent, _ = make_agent()
    original = context_module.get_history
    context_module.get_history = lambda conf, uid: []
    try:
        agent.set_memory_from_history("mao_pro_001", "2026-09-24_01-39-48_abc")
    finally:
        context_module.get_history = original
    assert agent.tasks.transcripts_dir == os.path.abspath(
        os.path.join("chat_history", "mao_pro_001")
    )
    assert agent.tasks.current_transcript == "2026-09-24_01-39-48_abc.json"


def test_transcripts_can_be_kept_from_hermes():
    from src.open_llm_vtuber.agent.agents.realtime import context as context_module

    agent, _ = make_agent(share_transcripts=False)
    original = context_module.get_history
    context_module.get_history = lambda conf, uid: []
    try:
        agent.set_memory_from_history("mao_pro_001", "2026-09-24_01-39-48_abc")
    finally:
        context_module.get_history = original
    assert agent.tasks.transcripts_dir == ""


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
    assert agent.router.remember is True
    assert agent.tasks.reasoning_effort == "high"
    assert agent.worker_max_turns == 8 and agent.remember_threshold == 0.5
    # Summaries on; with only 6 turns, at most 5 can be kept word for word.
    assert agent.summarize_history is True and agent.summary_keep_turns == 5
    # The talker is told it's Yuna's voice and that Hermes is her background helper.
    prompt = agent.context.system_prompt()
    assert prompt.startswith("You are Yuna.")
    assert "Hermes Agent" in prompt and "deepseek/deepseek-v4.1-flash" in prompt


if __name__ == "__main__":
    run_module(globals())
