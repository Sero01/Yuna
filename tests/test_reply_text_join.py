"""The reply text saved to chat history is rebuilt from sentence chunks, which the
sentence divider strips, so the join has to put the spaces back.

Run directly:  .venv/Scripts/python.exe tests/test_reply_text_join.py
"""

import os
import sys
from functools import reduce

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.conversations.conversation_utils import join_reply_text  # noqa: E402


def joined(chunks):
    return reduce(join_reply_text, chunks, "")


def test_sentences_are_joined_with_a_space():
    chunks = [
        "[smirk] Mostly,",
        "yeah — sentence-at-a-time is the bottleneck, that part's true.",
        "Anyway, Boss Boy, it's late.",
        "Sleep.",
    ]
    assert joined(chunks) == (
        "[smirk] Mostly, yeah — sentence-at-a-time is the bottleneck, that part's"
        " true. Anyway, Boss Boy, it's late. Sleep."
    )
    assert joined(["안녕하세요.", "반가워요."]) == "안녕하세요. 반가워요."


def test_chinese_and_japanese_sentences_are_joined_without_a_space():
    assert joined(["你好，", "今天怎么样？"]) == "你好，今天怎么样？"
    assert joined(["こんにちは。", "元気？"]) == "こんにちは。元気？"


def test_think_parentheses_hug_their_contents():
    assert joined(["(", "Let me think.", "Hmm.", ")", "Okay."]) == (
        "(Let me think. Hmm.) Okay."
    )


def test_empty_parts_add_no_space():
    assert joined(["", "Hi.", ""]) == "Hi."


if __name__ == "__main__":
    run_module(globals())
