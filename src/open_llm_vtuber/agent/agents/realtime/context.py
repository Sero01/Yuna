"""Conversation memory and the talker's prompt."""

import os
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ....chat_history_manager import get_history, get_metadata, update_metadate
from .prompts import SUMMARY_HEADER, TASK_STATE_HEADER, USER_FACTS_HEADER

Message = Dict[str, str]
INTERRUPT_NOTE = "[Interrupted by user]"
# Keys in the history file's metadata entry.
SUMMARY_KEY = "summary"
SUMMARY_EXCHANGES_KEY = "summarized_exchanges"


class ConversationMemory:
    """The whole conversation for this agent; the talker only sees a window of it,
    after a running summary of the exchanges before that window."""

    def __init__(self, max_messages: int = 300):
        self.max_messages = max_messages
        self.messages: List[Message] = []
        self._trimmed_exchanges = 0  # user messages dropped from the front so far
        self.summary = ""
        self.summarized_exchanges = 0  # exchanges from the start the summary covers
        # The loaded history (conf_uid, history_uid), where the summary is saved.
        self._history_ids: Optional[Tuple[str, str]] = None

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
            cut = len(self.messages) - self.max_messages
            self._trimmed_exchanges += sum(
                1 for m in self.messages[:cut] if m["role"] == "user"
            )
            del self.messages[:cut]

    def window(self, max_turns: int, step: int = 1) -> List[Message]:
        """The summary (if any), then at most the last `max_turns` exchanges it doesn't
        cover; each exchange starts at a user message.

        With `step` > 1 the start moves forward `step` exchanges at a time, so between
        moves the history (and the cached prompt prefix it begins) stays the same.
        """
        if max_turns <= 0:
            return []
        step = max(1, min(step, max_turns))
        user_indexes = self._user_indexes()
        excess = self._trimmed_exchanges + len(user_indexes) - max_turns
        first = self.summarized_exchanges  # counted from the start of the conversation
        if excess > 0:
            first = max(first, -(-excess // step) * step)
        messages = self.messages
        if user_indexes and (excess > 0 or first > self._trimmed_exchanges):
            drop = first - self._trimmed_exchanges
            messages = messages[
                user_indexes[min(max(drop, 0), len(user_indexes) - 1)] :
            ]
        head = (
            [{"role": "system", "content": f"{SUMMARY_HEADER}\n{self.summary}"}]
            if self.summary
            else []
        )
        return head + [dict(m) for m in messages]

    def due_for_summary(
        self, max_turns: int, keep: int
    ) -> Optional[Tuple[List[Message], int]]:
        """Once `max_turns` exchanges aren't in the summary, all but the last `keep` of
        them, and how many exchanges the summary covers once it includes them."""
        user_indexes = self._user_indexes()
        start = max(self.summarized_exchanges, self._trimmed_exchanges)
        pending = self._trimmed_exchanges + len(user_indexes) - start
        keep = max(0, min(keep, max_turns - 1))
        if max_turns <= 0 or pending < max_turns:
            return None
        first = start - self._trimmed_exchanges
        begin = user_indexes[first] if first > 0 else 0
        end = user_indexes[len(user_indexes) - keep] if keep else len(self.messages)
        return [dict(m) for m in self.messages[begin:end]], start + pending - keep

    def set_summary(self, summary: str, exchanges: int) -> None:
        """Use `summary` for the first `exchanges` exchanges (unless a summary already
        covers more) and save it with the loaded history."""
        if exchanges <= self.summarized_exchanges:
            return
        self.summary = (summary or "").strip()
        self.summarized_exchanges = exchanges
        if self._history_ids is not None:
            update_metadate(
                *self._history_ids,
                {SUMMARY_KEY: self.summary, SUMMARY_EXCHANGES_KEY: exchanges},
            )

    def _user_indexes(self) -> List[int]:
        return [i for i, m in enumerate(self.messages) if m["role"] == "user"]

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
        self._trimmed_exchanges = 0
        for msg in get_history(conf_uid, history_uid):
            role = {"human": "user", "ai": "assistant"}.get(msg.get("role"))
            content = msg.get("content")
            if role and isinstance(content, str) and content:
                self.add(role, content)
        metadata = get_metadata(conf_uid, history_uid)
        summary = metadata.get(SUMMARY_KEY)
        exchanges = metadata.get(SUMMARY_EXCHANGES_KEY)
        if isinstance(summary, str) and isinstance(exchanges, int) and summary:
            self.summary, self.summarized_exchanges = summary, exchanges
        else:
            self.summary, self.summarized_exchanges = "", 0
        self._history_ids = (conf_uid, history_uid)
        logger.info(f"Loaded {len(self.messages)} messages from history.")
        if self.summary:
            logger.info(f"Loaded a summary of the first {exchanges} exchanges.")


def transcript(messages: List[Message], user_name: str, ai_name: str) -> str:
    names = {"user": user_name, "assistant": ai_name}
    return "\n".join(
        f"{names[m['role']]}: {m['content']}" for m in messages if m["role"] in names
    )


def worker_history(
    messages: List[Message], max_turns: Optional[int] = None
) -> List[Message]:
    """User and assistant messages only; with `max_turns`, just the last that many
    exchanges."""
    kept = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    if max_turns is not None:
        starts = [i for i, m in enumerate(kept) if m["role"] == "user"]
        if len(starts) > max_turns:
            kept = kept[starts[-max_turns] :] if max_turns > 0 else []
    return kept


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

    def user_facts(self) -> str:
        """The user profile as bullet points ('' if there is none)."""
        return format_user_facts(self._user.read())

    def system_prompt(self) -> str:
        parts = [self._soul.read(), self.base_prompt]
        facts = self.user_facts()
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
