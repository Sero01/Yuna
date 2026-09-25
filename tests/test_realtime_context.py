"""Memory window, prompt assembly, SOUL/USER file reloads, interrupts, history loading.

Run directly:  .venv/Scripts/python.exe tests/test_realtime_context.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime import context as context_module  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.context import (  # noqa: E402
    ContextBuilder,
    ConversationMemory,
    format_user_facts,
    task_state_message,
    transcript,
    worker_history,
)
from src.open_llm_vtuber.agent.agents.realtime.prompts import (  # noqa: E402
    SUMMARY_HEADER,
    TASK_STATE_HEADER,
    USER_FACTS_HEADER,
)


def memory_with(exchanges: int) -> ConversationMemory:
    memory = ConversationMemory()
    for i in range(exchanges):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
    return memory


def test_window_keeps_the_last_n_exchanges():
    window = memory_with(12).window(8)
    assert len(window) == 16, window
    assert window[0] == {"role": "user", "content": "u4"}
    assert window[-1] == {"role": "assistant", "content": "a11"}


def test_window_with_fewer_exchanges_returns_everything():
    assert len(memory_with(3).window(8)) == 6
    assert memory_with(3).window(0) == []


def test_window_start_moves_in_steps_so_the_prompt_prefix_stays_cached():
    memory = memory_with(50)
    window = memory.window(50, step=10)
    assert len(window) == 100 and window[0]["content"] == "u0"
    memory.add("user", "u50")
    # One exchange too many drops a whole step, so the start doesn't move every turn.
    assert memory.window(50, step=10)[0]["content"] == "u10"
    for i in range(51, 60):
        memory.add("assistant", f"a{i - 1}")
        memory.add("user", f"u{i}")
    assert memory.window(50, step=10)[0]["content"] == "u10"
    memory.add("assistant", "a59")
    memory.add("user", "u60")
    assert memory.window(50, step=10)[0]["content"] == "u20"


def test_window_steps_count_exchanges_already_trimmed_from_the_buffer():
    memory = ConversationMemory(max_messages=30)
    for i in range(25):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
    assert memory.messages[0]["content"] == "u10"
    # 25 exchanges so far, 13 over the limit -> drop 16 (a multiple of 4) -> start at u16.
    assert memory.window(12, step=4)[0]["content"] == "u16"


def test_window_returns_copies():
    memory = memory_with(1)
    memory.window(8)[0]["content"] = "changed"
    assert memory.messages[0]["content"] == "u0"


def test_nothing_is_due_for_a_summary_before_max_turns():
    assert memory_with(49).due_for_summary(50, keep=10) is None


def test_the_oldest_exchanges_are_due_once_max_turns_are_unsummarized():
    memory = ConversationMemory()
    memory.add(
        "assistant", "Hmph, you're back."
    )  # a greeting before the first user turn
    for i in range(50):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
    messages, upto = memory.due_for_summary(50, keep=10)
    assert upto == 40
    assert messages[0] == {"role": "assistant", "content": "Hmph, you're back."}
    assert messages[-1] == {"role": "assistant", "content": "a39"}
    assert len(messages) == 81


def test_window_starts_after_the_summary_and_shows_it_first():
    memory = memory_with(50)
    memory.set_summary("They talked about exams.", 40)
    window = memory.window(100, step=10)
    assert window[0] == {
        "role": "system",
        "content": f"{SUMMARY_HEADER}\nThey talked about exams.",
    }
    assert window[1] == {"role": "user", "content": "u40"}
    assert len(window) == 1 + 20


def test_the_next_summary_covers_only_what_came_after_the_last_one():
    memory = memory_with(50)
    memory.set_summary("Part one.", 40)
    for i in range(50, 89):
        memory.add("user", f"u{i}")
        memory.add("assistant", f"a{i}")
    assert memory.due_for_summary(50, keep=10) is None  # 49 unsummarized
    memory.add("user", "u89")
    memory.add("assistant", "a89")
    messages, upto = memory.due_for_summary(50, keep=10)
    assert upto == 80
    assert messages[0]["content"] == "u40" and messages[-1]["content"] == "a79"


def test_an_older_summary_never_replaces_a_newer_one():
    memory = memory_with(60)
    memory.set_summary("Newer.", 50)
    memory.set_summary("Older.", 40)
    assert (memory.summary, memory.summarized_exchanges) == ("Newer.", 50)


def test_a_capped_window_still_slides_if_summaries_keep_failing():
    memory = memory_with(105)
    memory.set_summary("Old.", 40)
    # 65 unsummarized exchanges fit under a cap of 60 only by sliding past the summary.
    window = memory.window(60, step=10)
    assert window[1]["content"] == "u50"


def test_summary_survives_a_reload_through_history_metadata():
    saved = {}
    originals = (
        context_module.get_history,
        context_module.get_metadata,
        context_module.update_metadate,
    )
    context_module.get_history = lambda conf, uid: [
        m
        for i in range(50)
        for m in (
            {"role": "human", "content": f"u{i}"},
            {"role": "ai", "content": f"a{i}"},
        )
    ]
    context_module.get_metadata = lambda conf, uid: {"role": "metadata", **saved}

    def update(conf, uid, metadata):
        assert (conf, uid) == ("conf", "uid")
        saved.update(metadata)
        return True

    context_module.update_metadate = update
    try:
        memory = ConversationMemory()
        memory.load("conf", "uid")
        memory.set_summary("They talked about exams.", 40)
        reloaded = ConversationMemory()
        reloaded.load("conf", "uid")
    finally:
        (
            context_module.get_history,
            context_module.get_metadata,
            context_module.update_metadate,
        ) = originals
    assert reloaded.summary == "They talked about exams."
    assert reloaded.summarized_exchanges == 40
    assert reloaded.window(100, step=10)[1]["content"] == "u40"


def test_loading_a_history_without_a_summary_clears_the_old_one():
    originals = (context_module.get_history, context_module.get_metadata)
    context_module.get_history = lambda conf, uid: []
    context_module.get_metadata = lambda conf, uid: {}
    try:
        memory = memory_with(50)
        memory.set_summary("Old session.", 40)
        memory.load("conf", "other")
    finally:
        context_module.get_history, context_module.get_metadata = originals
    assert (memory.summary, memory.summarized_exchanges) == ("", 0)


def test_add_skips_empty_and_duplicate_messages_and_trims():
    memory = ConversationMemory(max_messages=4)
    memory.add("user", "  ")
    memory.add("user", "hi")
    memory.add("user", "hi")
    assert memory.messages == [{"role": "user", "content": "hi"}]
    for i in range(5):
        memory.add("assistant", f"a{i}")
    assert len(memory.messages) == 4 and memory.messages[-1]["content"] == "a4"


def test_interrupt_replaces_the_unheard_reply():
    memory = memory_with(1)
    memory.handle_interrupt("Once upon")
    assert memory.messages[-2:] == [
        {"role": "assistant", "content": "Once upon..."},
        {"role": "system", "content": "[Interrupted by user]"},
    ]


def test_interrupt_can_leave_the_previous_reply_alone():
    memory = memory_with(1)
    memory.handle_interrupt("It's thirty", replace_last=False)
    assert memory.messages[-3:] == [
        {"role": "assistant", "content": "a0"},
        {"role": "assistant", "content": "It's thirty..."},
        {"role": "system", "content": "[Interrupted by user]"},
    ]


def test_interrupt_before_any_reply_appends_what_was_heard():
    memory = ConversationMemory()
    memory.add("user", "hello")
    memory.handle_interrupt("Hm")
    assert memory.messages[-2]["content"] == "Hm..."
    memory.handle_interrupt("")
    assert memory.messages[-1] == {"role": "system", "content": "[Interrupted by user]"}


def test_load_maps_history_roles():
    original = context_module.get_history
    context_module.get_history = lambda conf, uid: [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "Hmph."},
        {"role": "system", "content": "[Interrupted by user]"},
        {"role": "human", "content": ""},
    ]
    try:
        memory = ConversationMemory()
        memory.load("conf", "uid")
    finally:
        context_module.get_history = original
    assert memory.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hmph."},
    ]


def test_transcript_and_worker_history():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "[Interrupted by user]"},
        {"role": "assistant", "content": "Hmph."},
    ]
    assert transcript(messages, "Sam", "Yuna") == "Sam: hi\nYuna: Hmph."
    assert worker_history(messages) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hmph."},
    ]


def test_worker_history_can_keep_only_the_last_exchanges():
    messages = memory_with(10).window(10)
    kept = worker_history(messages, max_turns=3)
    assert len(kept) == 6, kept
    assert kept[0] == {"role": "user", "content": "u7"}


def test_task_state_message():
    assert task_state_message([]) == ""
    text = task_state_message([("t1", "t1 [running 3s] weather")])
    assert text == f"{TASK_STATE_HEADER}\nt1 [running 3s] weather"


def test_build_orders_the_messages():
    builder = ContextBuilder("You are Yuna.")
    history = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
    ]
    messages = builder.build(history, "TASKS", current_user="now", trailing="DO THIS")
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "system",
        "user",
        "system",
    ]
    assert messages[0]["content"] == "You are Yuna."
    assert messages[3]["content"] == "TASKS"
    assert messages[4]["content"] == "now"
    assert messages[5]["content"] == "DO THIS"
    assert builder.build([], "")[1:] == []


def test_format_user_facts_turns_sections_into_bullets():
    text = "Likes being called Boss Boy.\n§\nHas a younger sister.\n§\n"
    assert (
        format_user_facts(text)
        == "- Likes being called Boss Boy.\n- Has a younger sister."
    )
    assert format_user_facts("") == ""


def test_system_prompt_includes_soul_and_user_facts_and_reloads():
    with tempfile.TemporaryDirectory() as tmp:
        soul = os.path.join(tmp, "SOUL.md")
        user = os.path.join(tmp, "USER.md")
        with open(soul, "w", encoding="utf-8") as f:
            f.write("You are Yuna, a tsundere.")
        with open(user, "w", encoding="utf-8") as f:
            f.write("Likes Pokemon.")
        builder = ContextBuilder("Keep replies short.", soul, user)
        prompt = builder.system_prompt()
        assert prompt.startswith("You are Yuna, a tsundere.\n\nKeep replies short.")
        assert f"{USER_FACTS_HEADER}\n- Likes Pokemon." in prompt

        with open(user, "w", encoding="utf-8") as f:
            f.write("Likes Pokemon.\n§\nAllergic to peanuts.")
        future = time.time() + 5
        os.utime(user, (future, future))
        assert "- Allergic to peanuts." in builder.system_prompt()


def test_user_facts_on_their_own():
    with tempfile.TemporaryDirectory() as tmp:
        user = os.path.join(tmp, "USER.md")
        with open(user, "w", encoding="utf-8") as f:
            f.write("Likes Pokemon.\n§\nHas a younger sister.")
        builder = ContextBuilder("Base.", "", user)
        assert builder.user_facts() == "- Likes Pokemon.\n- Has a younger sister."
    assert ContextBuilder("Base.").user_facts() == ""


def test_missing_files_are_skipped():
    builder = ContextBuilder("Base.", "Z:/nope/SOUL.md", "Z:/nope/USER.md")
    assert builder.system_prompt() == "Base."


if __name__ == "__main__":
    run_module(globals())
