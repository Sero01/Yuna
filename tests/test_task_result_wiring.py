"""Glue between the real-time agent and the websocket/service layers.

Run directly:  .venv/Scripts/python.exe tests/test_task_result_wiring.py
"""

import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber import service_context as service_context_module  # noqa: E402
from src.open_llm_vtuber import websocket_handler as websocket_handler_module  # noqa: E402
from src.open_llm_vtuber.conversations import conversation_handler  # noqa: E402
from src.open_llm_vtuber.conversations.task_announcer import TaskAnnouncer  # noqa: E402
from src.open_llm_vtuber.service_context import ServiceContext  # noqa: E402
from src.open_llm_vtuber.websocket_handler import WebSocketHandler  # noqa: E402


class FakeAgent:
    quiet_gap_s = 0.0

    def __init__(self, untold=True):
        self.untold = untold
        self.listener = None
        self.tts_engine = None
        self.removed = []

    def set_task_listener(self, listener):
        self.listener = listener

    def remove_task_listener(self, listener):
        self.removed.append(listener)
        if self.listener == listener:
            self.listener = None

    def has_untold_results(self):
        return self.untold

    def set_tts_engine(self, engine):
        self.tts_engine = engine


class NoGroups:
    client_group_map = {}

    def get_client_group(self, client_uid):
        return None


async def test_task_result_trigger_starts_a_turn_with_task_metadata():
    calls = []

    async def fake_single_conversation(**kwargs):
        calls.append(kwargs)

    original = conversation_handler.process_single_conversation
    conversation_handler.process_single_conversation = fake_single_conversation
    tasks = {}
    try:
        await conversation_handler.handle_conversation_trigger(
            msg_type="task-result",
            data={},
            client_uid="c1",
            context=SimpleNamespace(),
            websocket=SimpleNamespace(send_text=None),
            client_contexts={},
            client_connections={},
            chat_group_manager=NoGroups(),
            received_data_buffers={},
            current_conversation_tasks=tasks,
            broadcast_to_group=None,
        )
        await tasks["c1"]
    finally:
        conversation_handler.process_single_conversation = original
    assert calls[0]["user_input"] == ""
    assert calls[0]["metadata"] == {
        "task_result": True,
        "skip_memory": True,
        "skip_history": True,
    }


async def trigger(msg_type, tasks, data=None):
    await conversation_handler.handle_conversation_trigger(
        msg_type=msg_type,
        data=data or {},
        client_uid="c1",
        context=SimpleNamespace(),
        websocket=SimpleNamespace(send_text=None),
        client_contexts={},
        client_connections={},
        chat_group_manager=NoGroups(),
        received_data_buffers={"c1": None},
        current_conversation_tasks=tasks,
        broadcast_to_group=None,
    )


async def test_a_user_turn_cuts_off_a_running_announcement():
    started = []

    async def slow_conversation(**kwargs):
        started.append(kwargs["metadata"])
        await asyncio.sleep(5)

    original = conversation_handler.process_single_conversation
    conversation_handler.process_single_conversation = slow_conversation
    tasks = {}
    try:
        await trigger("task-result", tasks)
        announcement = tasks["c1"]
        await asyncio.sleep(0.01)
        await trigger("text-input", tasks, {"text": "wait, one more thing"})
        await asyncio.sleep(0.01)
        assert announcement.cancelled(), "the user's turn replaces the announcement"
        user_turn = tasks["c1"]
        assert user_turn is not announcement and not user_turn.done()

        await trigger("text-input", tasks, {"text": "and another"})
        await asyncio.sleep(0.01)
        assert not user_turn.done(), "ordinary turns are not cancelled here"
        for task in (user_turn, tasks["c1"]):
            task.cancel()
    finally:
        conversation_handler.process_single_conversation = original


def make_handler(agent):
    handler = WebSocketHandler(default_context_cache=None)
    context = SimpleNamespace(agent_engine=agent, task_listener=None)
    handler.client_connections["c1"] = SimpleNamespace(send_text=None)
    handler.client_contexts["c1"] = context
    handler.chat_group_manager = NoGroups()
    return handler, context


async def test_connect_registers_an_announcer_with_the_agent():
    agent = FakeAgent()
    handler, context = make_handler(agent)
    handler._attach_task_announcer("c1", context)
    announcer = handler._task_announcers["c1"]
    assert isinstance(announcer, TaskAnnouncer)
    assert agent.listener == announcer.notify
    assert context.task_listener == announcer.notify
    handler._detach_task_announcer("c1")
    assert agent.listener is None and context.task_listener is None
    assert "c1" not in handler._task_announcers


async def test_results_start_a_task_result_turn_when_idle():
    triggered = []

    async def fake_trigger(**kwargs):
        triggered.append(kwargs)

    original = websocket_handler_module.handle_conversation_trigger
    websocket_handler_module.handle_conversation_trigger = fake_trigger
    try:
        agent = FakeAgent(untold=True)
        handler, context = make_handler(agent)
        handler._attach_task_announcer("c1", context)
        busy = asyncio.get_running_loop().create_future()
        handler.current_conversation_tasks["c1"] = busy  # a turn is running
        agent.listener()
        await asyncio.sleep(0.4)
        assert triggered == [], "waits for the running turn"
        busy.set_result(None)
        await asyncio.sleep(0.6)
        assert len(triggered) == 1 and triggered[0]["msg_type"] == "task-result"
        assert triggered[0]["context"] is context

        agent.untold = False
        agent.listener()
        await asyncio.sleep(0.6)
        assert len(triggered) == 1, "no turn when there is nothing left to tell"
        handler._detach_task_announcer("c1")
    finally:
        websocket_handler_module.handle_conversation_trigger = original


def service_context_with(agent):
    context = ServiceContext()
    context.agent_engine = agent
    context.system_config = SimpleNamespace(tool_prompts={}, model_dump=lambda: {})
    context.character_config = SimpleNamespace(
        agent_config=None,
        persona_prompt="You are Yuna.",
        avatar="",
        character_name="Yuna",
        tts_preprocessor_config=None,
        tts_config=SimpleNamespace(tts_model="old"),
    )
    return context


async def test_new_agent_gets_the_listener_tts_engine_and_character_name():
    created = []
    new_agent = FakeAgent()

    def fake_create_agent(**kwargs):
        created.append(kwargs)
        return new_agent

    old_agent = FakeAgent()
    context = service_context_with(old_agent)
    context.tts_engine = "tts-engine"
    context.task_listener = lambda: None
    original = service_context_module.AgentFactory.create_agent
    service_context_module.AgentFactory.create_agent = staticmethod(fake_create_agent)
    try:
        agent_config = SimpleNamespace(
            conversation_agent_choice="realtime_agent",
            agent_settings=SimpleNamespace(model_dump=lambda: {}),
            llm_configs=SimpleNamespace(model_dump=lambda: {}),
        )
        await context.init_agent(agent_config, "You are Yuna.")
    finally:
        service_context_module.AgentFactory.create_agent = original
    assert created[0]["tts_engine"] == "tts-engine"
    assert created[0]["character_name"] == "Yuna"
    assert context.agent_engine is new_agent
    assert new_agent.listener == context.task_listener
    assert old_agent.removed == [context.task_listener]


def test_tts_switch_reaches_the_agent():
    agent = FakeAgent()
    context = service_context_with(agent)
    original = service_context_module.TTSFactory.get_tts_engine
    service_context_module.TTSFactory.get_tts_engine = staticmethod(
        lambda model, **kw: f"engine:{model}"
    )
    try:
        context.init_tts(
            SimpleNamespace(tts_model="new", new=SimpleNamespace(model_dump=lambda: {}))
        )
    finally:
        service_context_module.TTSFactory.get_tts_engine = original
    assert agent.tts_engine == "engine:new"


if __name__ == "__main__":
    run_module(globals())
