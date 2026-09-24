"""Conversation memory and the talker's prompt."""

import os
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ....chat_history_manager import get_history
from .prompts import TASK_STATE_HEADER, USER_FACTS_HEADER

Message = Dict[str, str]
INTERRUPT_NOTE = "[Interrupted by user]"


class ConversationMemory:
    """The whole conversation for this agent; the talker only sees a window of it."""

    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self.messages: List[Message] = []

    def add(self, role: str, content: str) -> None:
        content = (content or "").strip()
        if not content:
            return
        if (
            self.messages
            and self.messages[-1]["role"] == role
            and self.messages[-1]["content"] == content
        ):
            return
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            del self.messages[: len(self.messages) - self.max_messages]

    def window(self, max_turns: int) -> List[Message]:
        """The last `max_turns` exchanges; each exchange starts at a user message."""
        if max_turns <= 0:
            return []
        user_indexes = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        start = user_indexes[-max_turns] if len(user_indexes) > max_turns else 0
        return [dict(m) for m in self.messages[start:]]

    def handle_interrupt(self, heard_response: str, replace_last: bool = True) -> None:
        """Record what the user heard before interrupting. `replace_last=False` when the
        interrupted reply was never recorded (so the last assistant message is older)."""
        heard = (heard_response or "").strip()
        if replace_last and self.messages and self.messages[-1]["role"] == "assistant":
            self.messages[-1]["content"] = heard + "..."
        elif heard:
            self.messages.append({"role": "assistant", "content": heard + "..."})
        self.messages.append({"role": "system", "content": INTERRUPT_NOTE})

    def load(self, conf_uid: str, history_uid: str) -> None:
        self.messages = []
        for msg in get_history(conf_uid, history_uid):
            role = {"human": "user", "ai": "assistant"}.get(msg.get("role"))
            content = msg.get("content")
            if role and isinstance(content, str) and content:
                self.add(role, content)
        logger.info(f"Loaded {len(self.messages)} messages from history.")


def transcript(messages: List[Message], user_name: str, ai_name: str) -> str:
    names = {"user": user_name, "assistant": ai_name}
    return "\n".join(
        f"{names[m['role']]}: {m['content']}" for m in messages if m["role"] in names
    )


def worker_history(messages: List[Message]) -> List[Message]:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]


def task_state_message(lines: List[Tuple[str, str]]) -> str:
    if not lines:
        return ""
    return TASK_STATE_HEADER + "\n" + "\n".join(line for _, line in lines)


def format_user_facts(text: str) -> str:
    """Hermes separates memory entries with lines containing only '§'."""
    entries = [
        e.strip() for e in re.split(r"^\s*§\s*$", text or "", flags=re.MULTILINE)
    ]
    return "\n".join(f"- {e}" for e in entries if e)


class _FileText:
    """Reads a text file, re-reading it only when its modification time changes."""

    def __init__(self, path: str):
        self.path = os.path.expanduser(path) if path else ""
        self._mtime: Optional[float] = None
        self._text = ""
        self._warned = False

    def read(self) -> str:
        if not self.path:
            return ""
        try:
            mtime = os.path.getmtime(self.path)
            if mtime != self._mtime:
                with open(self.path, encoding="utf-8") as f:
                    self._text = f.read().strip()
                self._mtime = mtime
        except OSError as e:
            if not self._warned:
                logger.warning(f"Real-time agent can't read {self.path}: {e}")
                self._warned = True
        return self._text


class ContextBuilder:
    """Assembles the talker's messages: persona, user facts, history, task state."""

    def __init__(
        self, system_prompt: str, soul_path: str = "", user_profile_path: str = ""
    ):
        self.base_prompt = (system_prompt or "").strip()
        self._soul = _FileText(soul_path)
        self._user = _FileText(user_profile_path)

    def system_prompt(self) -> str:
        parts = [self._soul.read(), self.base_prompt]
        facts = format_user_facts(self._user.read())
        if facts:
            parts.append(f"{USER_FACTS_HEADER}\n{facts}")
        return "\n\n".join(p for p in parts if p)

    def build(
        self,
        history: List[Message],
        task_state: str = "",
        current_user: Optional[str] = None,
        trailing: Optional[str] = None,
    ) -> List[Message]:
        messages = [{"role": "system", "content": self.system_prompt()}]
        messages += [dict(m) for m in history]
        if task_state:
            messages.append({"role": "system", "content": task_state})
        if current_user is not None:
            messages.append({"role": "user", "content": current_user})
        if trailing:
            messages.append({"role": "system", "content": trailing})
        return messages
