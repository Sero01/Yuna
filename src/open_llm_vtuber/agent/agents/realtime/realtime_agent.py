"""Real-time conversation agent.

Each turn starts Jev (routing) and the talker (a speculative reply) at the same moment.
Chat turns release the talker's reply as soon as Jev agrees; task turns discard it, start
a Hermes run, and speak a pre-made acknowledgement instead. Finished tasks, and commands
Hermes needs permission for, are announced through a separate `task-result` turn started
by the websocket layer; the user's spoken yes/no answers the approval (once or deny only).

Jev also says whether a turn is worth remembering; if so, a silent Hermes run decides what,
if anything, to save to memory. Task runs get a pointer to the saved voice transcripts.

Once `max_turns` exchanges pile up, all but the latest few are folded into a running
summary in the background; the talker and Hermes tasks see it before the recent exchanges.
"""

import asyncio
import os
import time
import uuid
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import httpx
from loguru import logger

from ....chat_history_manager import _get_safe_history_path
from ....config_manager import TTSPreprocessorConfig
from ....config_manager.agent import RealtimeAgentConfig
from ....utils.tts_preprocessor import fix_pronunciation
from ...input_types import BatchInput
from ...output_types import Actions, AudioOutput, DisplayText, SentenceOutput
from ...transformers import (
    actions_extractor,
    display_processor,
    sentence_divider,
    tts_filter,
)
from ..agent_interface import AgentInterface
from .ack_pool import Ack, AckPool, clean_ack, discard_audio
from .context import (
    ContextBuilder,
    ConversationMemory,
    Message,
    task_state_message,
    transcript,
    worker_history,
)
from .loop_client import LoopBoundClient, warm_up_pings
from .openers import (
    UNDECIDED,
    OpenerLibrary,
    OpenerMatch,
    engine_renderer,
    match_opener,
    opener_instruction,
    opener_reminder,
)
from .prompts import (
    ACK_FALLBACK_TEXT,
    EARLY_OPENER_HINT,
    ACK_INSTRUCTION,
    ACK_USER_TURN,
    APPROVAL_EXPIRED_INSTRUCTION,
    APPROVAL_REQUEST_INSTRUCTION,
    APPROVED_INSTRUCTION,
    CANCEL_INSTRUCTION,
    CHANGE_INSTRUCTION,
    DENIED_INSTRUCTION,
    FALLBACK_LINE,
    SUMMARY_CONDENSE_INSTRUCTIONS,
    SUMMARY_INPUT,
    SUMMARY_INSTRUCTIONS,
    TALKER_SELF_NOTE,
    TASK_RESULTS_INSTRUCTION,
    TASK_START_FAILED_INSTRUCTION,
    UNSURE_INSTRUCTION,
)
from .router import FallbackRouter, JevRouter, RouterState
from .tags import strip_unknown_tags
from .talker import Talker, TalkerError
from .tasks import DONE, RUNNING, TaskManager, TaskRecord, TaskStartError

INSTRUCTED_MAX_TOKENS = 80  # short replies driven by a trailing system instruction
ACK_MAX_TOKENS = 40
# Hermes usually accepts a run within this; past it, "on it" is said without waiting
# (a refused localhost connection takes ~2 s to fail on Windows).
TASK_START_GRACE_S = 0.1
# If the route isn't back by then (Jev slow, so the fallback router is racing it), the
# reply's opener is spoken without waiting: any kind of turn can start with one.
EARLY_OPENER_AFTER_S = 0.7
WARM_INTERVAL_S = 30.0
RECENT_MESSAGES_FOR_ROUTER = 6
MESSAGES_FOR_MEMORY_REVIEW = 4  # context before the user's message
SUMMARY_MAX_TOKENS = 800  # prompts ask for 150-250 words; DeepSeek overshoots a bit
SUMMARY_CONDENSE_AFTER_WORDS = 400  # past this, all notes are condensed to ~250
# With summaries on, the talker's window only grows past max_turns while a summary is being
# made, or if they keep failing; this many times max_turns is where it starts sliding.
WINDOW_CAP_WITH_SUMMARIES = 2

_END = object()
Output = Union[SentenceOutput, AudioOutput, Dict[str, Any]]


class SpeculativeReply:
    """Streams the talker in the background, holding its tokens until released."""

    def __init__(self, tokens: AsyncIterator[str]):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._head: List[Any] = []  # items already read by peek_opener, replayed first
        self._task = asyncio.create_task(self._pump(tokens))

    async def _pump(self, tokens: AsyncIterator[str]) -> None:
        try:
            async for piece in tokens:
                self._queue.put_nowait(piece)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._queue.put_nowait(
                e if isinstance(e, TalkerError) else TalkerError(str(e))
            )
            return
        self._queue.put_nowait(_END)

    async def peek_opener(self, keys) -> Optional[OpenerMatch]:
        """Read just enough of the reply to tell whether it starts with an opener.
        Everything read is still released later (safe to cancel part-way)."""
        buffer, ended, found = "", False, UNDECIDED
        while found is UNDECIDED:
            item = await self._queue.get()
            self._head.append(item)
            if isinstance(item, str):
                buffer += item
            else:
                ended = True
            found = match_opener(buffer, keys, final=ended)
        return found if isinstance(found, OpenerMatch) else None

    def drop_opener(self, match: OpenerMatch) -> None:
        """The opener found by peek_opener was spoken already; release only the rest."""
        ends = [item for item in self._head if not isinstance(item, str)]
        self._head = ([match.rest] if match.rest else []) + ends

    async def release(self) -> AsyncIterator[str]:
        while True:
            item = self._head.pop(0) if self._head else await self._queue.get()
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def cancel(self) -> None:
        self._task.cancel()


async def _single(text: str) -> AsyncIterator[str]:
    yield text


def _log_start_failure(start: "asyncio.Future") -> None:
    if not start.cancelled() and start.exception() is not None:
        logger.warning(f"Could not start a background task: {start.exception()}")


class RealtimeAgent(AgentInterface):
    """Fast talker + Jev routing + Hermes background tasks."""

    def __init__(
        self,
        *,
        router,
        talker,
        tasks: TaskManager,
        context: ContextBuilder,
        memory: Optional[ConversationMemory] = None,
        live2d_model=None,
        tts_preprocessor_config: Optional[TTSPreprocessorConfig] = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        max_turns: int = 8,
        quiet_gap_s: float = 1.0,
        prerender_acks: bool = True,
        ai_name: str = "Yuna",
        warm_up: Optional[Callable[[], Awaitable[None]]] = None,
        openers: Optional[OpenerLibrary] = None,
        worker_max_turns: int = 8,
        remember_threshold: float = 0.5,
        share_transcripts: bool = True,
        summarize_history: bool = True,
        summary_keep_turns: int = 10,
    ):
        super().__init__()
        self.router = router
        self.talker = talker
        self.tasks = tasks
        self.context = context
        self.memory = memory or ConversationMemory()
        self.max_turns = max_turns
        # The window's start moves a fifth of it at a time, so the talker's cached prompt
        # prefix survives between moves.
        self._window_step = max(1, max_turns // 5)
        self.summarize_history = summarize_history
        self.summary_keep_turns = max(0, min(summary_keep_turns, max_turns - 1))
        self._summary_task: Optional[asyncio.Task] = None
        self._user_name = "User"
        self.worker_max_turns = worker_max_turns
        self.remember_threshold = remember_threshold
        self.share_transcripts = share_transcripts
        self.quiet_gap_s = quiet_gap_s
        self.ai_name = ai_name
        self.pending_offer: Optional[str] = None
        self.openers = openers
        self._opener_hint = opener_reminder(openers.phrases) if openers else ""
        self.ack_pool = AckPool(self.generate_ack_text)
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._prerender_acks = prerender_acks
        self._tts_engine = None
        self._warm_up = warm_up
        self._warm_task: Optional[asyncio.Task] = None
        self._task_listener: Optional[Callable[[], None]] = None
        self._interrupt_handled = False
        self._reply_recorded = False  # has this turn's reply been added to memory?
        self._early_opener = ""  # opener spoken before this turn's route was known
        self.tasks.listener = self._on_task_update
        self._pipeline = self._build_pipeline()

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Dict[str, Any]],
        system_prompt: str,
        live2d_model=None,
        tts_preprocessor_config: Optional[TTSPreprocessorConfig] = None,
        tts_engine=None,
        ai_name: str = "Yuna",
    ) -> "RealtimeAgent":
        cfg = RealtimeAgentConfig(**(settings or {}))
        openrouter_key = cfg.openrouter_api_key or os.environ.get(
            "OPENROUTER_API_KEY", ""
        )
        hermes_key = cfg.hermes_api_key or os.environ.get("HERMES_API_KEY", "")
        if not openrouter_key:
            logger.warning("realtime_agent: no OpenRouter API key configured")
        if not hermes_key:
            logger.warning("realtime_agent: no Hermes API key configured")
        openers = (
            OpenerLibrary(cfg.openers, None, carrier=cfg.opener_carrier)
            if cfg.openers
            else None
        )
        if openers is not None:
            system_prompt = f"{system_prompt}\n\n{opener_instruction(cfg.openers)}"
        self_note = TALKER_SELF_NOTE.format(talker_model=cfg.talker_model)
        system_prompt = f"{system_prompt}\n\n{self_note}"
        http = LoopBoundClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120.0),
        )
        fallback = (
            FallbackRouter(
                http.get,
                cfg.openrouter_base_url,
                openrouter_key,
                cfg.fallback_router_model,
                cfg.fallback_router_provider,
            )
            if cfg.fallback_router_model
            else None
        )
        agent = cls(
            router=JevRouter(
                http.get,
                cfg.jev_url,
                openrouter_key,
                model=cfg.jev_model,
                timeout_s=cfg.jev_timeout_s,
                unsure_low=cfg.unsure_low,
                unsure_high=cfg.unsure_high,
                fallback=fallback,
                fallback_after_s=cfg.jev_fallback_after_s,
                remember=cfg.remember_threshold > 0,
            ),
            talker=Talker(
                http.get,
                cfg.openrouter_base_url,
                openrouter_key,
                cfg.talker_model,
                provider=cfg.talker_provider,
                temperature=cfg.talker_temperature,
                max_tokens=cfg.talker_max_tokens,
                hedge_after_s=cfg.hedge_after_s,
                race_provider=cfg.talker_race_provider,
            ),
            tasks=TaskManager(
                http.get,
                cfg.hermes_base_url,
                hermes_key,
                timeout_s=cfg.task_timeout_s,
                max_active=cfg.max_active_tasks,
                reasoning_effort=cfg.hermes_reasoning_effort,
            ),
            context=ContextBuilder(system_prompt, cfg.soul_path, cfg.user_profile_path),
            live2d_model=live2d_model,
            tts_preprocessor_config=tts_preprocessor_config,
            faster_first_response=cfg.faster_first_response,
            segment_method=cfg.segment_method,
            max_turns=cfg.max_turns,
            quiet_gap_s=cfg.quiet_gap_s,
            prerender_acks=cfg.prerender_acks,
            ai_name=ai_name,
            warm_up=warm_up_pings(
                http,
                f"{cfg.openrouter_base_url.rstrip('/')}/key",
                {"Authorization": f"Bearer {openrouter_key}"},
            ),
            openers=openers,
            worker_max_turns=cfg.worker_max_turns,
            remember_threshold=cfg.remember_threshold,
            share_transcripts=cfg.share_transcripts,
            summarize_history=cfg.summarize_history,
            summary_keep_turns=cfg.summary_keep_turns,
        )
        agent.set_tts_engine(tts_engine)
        return agent

    # ---- AgentInterface ----

    async def chat(self, input_data: BatchInput) -> AsyncIterator[Output]:
        self._interrupt_handled = False
        self._reply_recorded = False
        self._early_opener = ""
        self.ack_pool.fill()
        if self.openers is not None:
            self.openers.ensure_built()
        user_name = next((t.from_name for t in input_data.texts if t.from_name), None)
        if user_name:
            self._user_name = user_name
        self._maybe_summarize()
        metadata = input_data.metadata or {}
        if metadata.get("task_result"):
            async for out in self._task_result_turn():
                yield out
            return

        text = "\n".join(t.content for t in input_data.texts if t.content).strip()
        if input_data.images:
            logger.info("realtime_agent ignores images; the talker is text-only")
        history = self._history()
        if not metadata.get("skip_memory"):
            self.memory.add("user", text)

        if metadata.get("proactive_speak"):
            messages = self.context.build(
                history,
                task_state_message(self.tasks.recent_lines()),
                current_user=text,
                trailing=self._with_opener_hint(None),
            )
            async for out in self._speak(self.talker.stream(messages)):
                yield out
            return

        async for out in self._routed_turn(text, history, self._user_name):
            yield out

    def handle_interrupt(self, heard_response: str) -> None:
        if self._interrupt_handled:
            return
        self._interrupt_handled = True
        self.memory.handle_interrupt(heard_response, replace_last=self._reply_recorded)

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        self._cancel_summary()  # it belongs to the conversation being replaced
        self.memory.load(conf_uid, history_uid)
        if not self.share_transcripts:
            return
        try:
            path = os.path.abspath(_get_safe_history_path(conf_uid, history_uid))
        except ValueError as e:
            logger.warning(f"Not sharing transcripts with Hermes: {e}")
            return
        self.tasks.set_transcripts(os.path.dirname(path), os.path.basename(path))

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        logger.warning("realtime_agent does not support group conversations")

    # ---- turns ----

    async def _routed_turn(
        self, text: str, history: List[Message], user_name: str
    ) -> AsyncIterator[Output]:
        task_lines = self.tasks.recent_lines()
        task_state = task_state_message(task_lines)
        waiting = self.tasks.awaiting_approval()
        asking = waiting[-1] if waiting else None
        state = RouterState(
            latest_user_message=text,
            recent_conversation=transcript(
                history[-RECENT_MESSAGES_FOR_ROUTER:], user_name, self.ai_name
            ),
            active_tasks=task_lines,
            pending_offer=self.pending_offer,
            pending_approval=(
                f"{asking.approval.description}: `{asking.approval.command}`"
                if asking is not None
                else None
            ),
            known_facts=self.context.user_facts(),
        )
        speculative = SpeculativeReply(
            self.talker.stream(
                self.context.build(
                    history,
                    task_state,
                    current_user=text,
                    trailing=self._with_opener_hint(None),
                )
            )
        )
        decide = asyncio.ensure_future(self.router.decide(state))
        try:
            started = time.perf_counter()
            async for out in self._opener_before_route(speculative, decide):
                yield out
            route = await decide
            logger.info(
                f"Route: {route.kind} action={route.action} task={route.task_id} "
                f"p_task={route.p_task:.2f} via {route.source} "
                f"in {time.perf_counter() - started:.2f}s"
            )
            offer, self.pending_offer = self.pending_offer, None
            remember = self._worth_remembering(route)
            if remember and route.kind not in ("new_task", "accept_offer"):
                self.tasks.remember(self._memory_excerpt(history, text, user_name))
            rec = self.tasks.get(route.task_id) if route.kind == "followup" else None
            speak_speculative = route.kind == "chat" or (
                route.kind == "followup"
                and (rec is None or rec.status != RUNNING or route.action == "status")
            )
            if speak_speculative:
                if rec is not None:
                    self.tasks.mark_told(rec.id)
                async for out in self._speak(speculative.release()):
                    yield out
                return

            speculative.cancel()
            if route.kind == "approval":
                async for out in self._answer_approval(
                    asking, route.action, text, history
                ):
                    yield out
            elif route.kind == "unsure":
                self.pending_offer = text
                async for out in self._instructed(
                    history, task_state, text, UNSURE_INSTRUCTION
                ):
                    yield out
            elif route.kind in ("new_task", "accept_offer"):
                request = (
                    f"{offer} (the user then said: {text})"
                    if route.kind == "accept_offer" and offer
                    else text
                )
                async for out in self._start_task(
                    request, history, task_state, text, remember
                ):
                    yield out
            elif route.action == "cancel":
                await self.tasks.cancel(rec.id)
                instruction = CANCEL_INSTRUCTION.format(request=rec.request)
                task_state = task_state_message(self.tasks.recent_lines())
                async for out in self._instructed(
                    history, task_state, text, instruction
                ):
                    yield out
            else:
                async for out in self._change_task(rec, text, history):
                    yield out
        finally:
            speculative.cancel()
            decide.cancel()  # only still running if the turn was cut off

    async def _opener_before_route(
        self, speculative: SpeculativeReply, decide: "asyncio.Future"
    ) -> AsyncIterator[Output]:
        """If the route is slow, speak the reply's opener clip before it arrives."""
        keys = self.openers.keys() if self.openers is not None else set()
        if not keys:
            return
        done, _ = await asyncio.wait({decide}, timeout=EARLY_OPENER_AFTER_S)
        if done:
            return
        peek = asyncio.ensure_future(speculative.peek_opener(keys))
        try:
            await asyncio.wait({decide, peek}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not peek.done():
                peek.cancel()  # what it read so far is still released later
        if decide.done() or peek.cancelled() or peek.result() is None:
            return
        match = peek.result()
        speculative.drop_opener(match)
        self._early_opener = match.text
        logger.info(
            f"Route not back after {EARLY_OPENER_AFTER_S}s; "
            f"speaking the opener {match.text!r} first"
        )
        yield self._opener_clip(match)

    def _worth_remembering(self, route) -> bool:
        return (
            self.remember_threshold > 0
            and route.p_remember is not None
            and route.p_remember >= self.remember_threshold
        )

    def _memory_excerpt(self, history: List[Message], text: str, user_name: str) -> str:
        before = transcript(
            history[-MESSAGES_FOR_MEMORY_REVIEW:], user_name, self.ai_name
        )
        latest = f"{user_name}: {text}"
        return f"{before}\n{latest}" if before else latest

    async def _start_task(
        self,
        request: str,
        history: List[Message],
        task_state: str,
        text: str,
        remember: bool = False,
    ) -> AsyncIterator[Output]:
        """`remember`: Jev thinks the message is also worth remembering, so the task is
        asked to consider saving it (instead of a separate memory review).

        A start that fails within TASK_START_GRACE_S is explained instead of
        acknowledged; a slower one is acknowledged first and corrected if it fails."""
        start = asyncio.ensure_future(
            self.tasks.start(
                request,
                worker_history(history, self.worker_max_turns),
                remember,
                summary=self.memory.summary,
            )
        )
        start.add_done_callback(_log_start_failure)  # also if the turn is cut off
        acked = False
        done, _ = await asyncio.wait({start}, timeout=TASK_START_GRACE_S)
        if not done:
            acked = True
            async for out in self._acknowledge():
                yield out
        try:
            await asyncio.shield(start)
        except TaskStartError as e:
            instruction = TASK_START_FAILED_INSTRUCTION.format(reason=e)
            async for out in self._instructed(history, task_state, text, instruction):
                yield out
            return
        if not acked:
            async for out in self._acknowledge():
                yield out

    async def _change_task(
        self, rec: TaskRecord, text: str, history: List[Message]
    ) -> AsyncIterator[Output]:
        instruction = CHANGE_INSTRUCTION.format(request=rec.request)
        if not await self.tasks.steer(rec.id, text):
            await self.tasks.cancel(rec.id)
            try:
                await self.tasks.start(
                    f"{rec.request} (update from the user: {text})",
                    worker_history(history, self.worker_max_turns),
                    summary=self.memory.summary,
                )
            except TaskStartError as e:
                instruction = TASK_START_FAILED_INSTRUCTION.format(reason=e)
        task_state = task_state_message(self.tasks.recent_lines())
        async for out in self._instructed(history, task_state, text, instruction):
            yield out

    async def _answer_approval(
        self,
        rec: Optional[TaskRecord],
        choice: str,
        text: str,
        history: List[Message],
    ) -> AsyncIterator[Output]:
        description = rec.approval.description if rec is not None else "go ahead"
        if rec is not None and await self.tasks.respond(rec.id, choice):
            template = APPROVED_INSTRUCTION if choice == "once" else DENIED_INSTRUCTION
        else:
            template = APPROVAL_EXPIRED_INSTRUCTION
        task_state = task_state_message(self.tasks.recent_lines())
        async for out in self._instructed(
            history, task_state, text, template.format(description=description)
        ):
            yield out

    async def _task_result_turn(self) -> AsyncIterator[Output]:
        results = self.tasks.untold_results()
        approvals = self.tasks.untold_approvals()
        if not results and not approvals:
            return
        instructions = []
        if results:
            for rec in results:
                rec.told = True
            summary = "\n".join(
                f"- {r.request}: {r.result}"
                if r.status == DONE
                else f"- {r.request}: failed ({r.error})"
                for r in results
            )
            instructions.append(TASK_RESULTS_INSTRUCTION.format(results=summary))
        if approvals:
            for rec in approvals:
                self.tasks.mark_approval_told(rec.id)
            asks = "\n".join(
                f"- {r.request}: it wants to {r.approval.description} "
                f"(command: {r.approval.command})"
                for r in approvals
            )
            instructions.append(APPROVAL_REQUEST_INSTRUCTION.format(approvals=asks))
        messages = self.context.build(
            self._history(),
            task_state_message(self.tasks.recent_lines()),
            trailing=self._with_opener_hint("\n\n".join(instructions)),
        )
        async for out in self._speak(self.talker.stream(messages)):
            yield out

    async def _acknowledge(self) -> AsyncIterator[Output]:
        ack = self.ack_pool.take()
        if ack is None:
            try:
                text = clean_ack(await self.generate_ack_text())
            except Exception as e:
                logger.warning(f"Could not generate an acknowledgement: {e}")
                text = ""
            ack = Ack(text or ACK_FALLBACK_TEXT)
        self._record_reply(ack.text)
        if ack.audio_path:
            try:
                yield AudioOutput(
                    audio_path=ack.audio_path,
                    display_text=DisplayText(text=ack.text),
                    transcript=ack.text,
                    actions=Actions(),
                )
            finally:
                discard_audio(ack)
        else:
            async for out in self._pipeline(_single(ack.text)):
                yield out

    async def _instructed(
        self, history: List[Message], task_state: str, text: str, instruction: str
    ) -> AsyncIterator[Output]:
        messages = self.context.build(
            history,
            task_state,
            current_user=text,
            trailing=self._with_opener_hint(instruction),
        )
        async for out in self._speak(
            self.talker.stream(messages, max_tokens=INSTRUCTED_MAX_TOKENS)
        ):
            yield out

    def _with_opener_hint(self, trailing: Optional[str]) -> Optional[str]:
        """`trailing` plus the opener reminder, for replies that are spoken."""
        hint = self._opener_hint
        if self._early_opener:
            hint = EARLY_OPENER_HINT.format(opener=self._early_opener)
        if not hint:
            return trailing
        return f"{trailing}\n\n{hint}" if trailing else hint

    async def _speak(self, tokens: AsyncIterator[str]) -> AsyncIterator[Output]:
        """Speak a token stream; record it in memory only if it finished (an interrupt
        records what was heard instead, via handle_interrupt)."""
        spoken: List[str] = []

        async def source() -> AsyncIterator[str]:
            try:
                async for piece in tokens:
                    spoken.append(piece)
                    yield piece
            except TalkerError as e:
                logger.error(f"Talker failed: {e}")
                if not spoken:
                    spoken.append(FALLBACK_LINE)
                    yield FALLBACK_LINE

        opener, rest = await self._split_opener(source())
        if opener is not None:
            yield opener
        async for out in self._pipeline(rest):
            yield out
        self._record_reply("".join(spoken))

    async def _split_opener(
        self, tokens: AsyncIterator[str]
    ) -> Tuple[Optional[AudioOutput], AsyncIterator[str]]:
        """If the reply starts with an opener that has a ready clip, return that clip and
        the tokens after it; otherwise no clip and the tokens unchanged."""
        keys = self.openers.keys() if self.openers is not None else set()
        if not keys:
            return None, tokens
        stream = tokens.__aiter__()
        buffer, ended, found = "", False, UNDECIDED
        while found is UNDECIDED:
            try:
                buffer += await stream.__anext__()
            except StopAsyncIteration:
                ended = True
            found = match_opener(buffer, keys, final=ended)
        head = found.rest if isinstance(found, OpenerMatch) else buffer

        async def rest() -> AsyncIterator[str]:
            if head:
                yield head
            if not ended:
                async for piece in stream:
                    yield piece

        if not isinstance(found, OpenerMatch):
            return None, rest()
        return self._opener_clip(found), rest()

    def _opener_clip(self, found: OpenerMatch) -> AudioOutput:
        expressions = (
            self._live2d_model.extract_emotion(found.tags)
            if found.tags and self._live2d_model is not None
            else []
        )
        return AudioOutput(
            audio_path=self.openers.clip(found.key),
            display_text=DisplayText(text=found.text),
            transcript=found.text,
            actions=Actions(expressions=expressions or None),
        )

    def _history(self) -> List[Message]:
        if self.summarize_history:
            cap = WINDOW_CAP_WITH_SUMMARIES * self.max_turns
            return self.memory.window(cap, self._window_step)
        return self.memory.window(self.max_turns, self._window_step)

    # ---- running summary ----

    def _maybe_summarize(self) -> None:
        """Start summarizing the oldest exchanges in the background once `max_turns` of
        them aren't in the summary yet. A failed attempt is retried on the next turn."""
        if not self.summarize_history:
            return
        if self._summary_task is not None and not self._summary_task.done():
            return
        due = self.memory.due_for_summary(self.max_turns, self.summary_keep_turns)
        if due is not None:
            self._summary_task = asyncio.create_task(self._summarize(*due))

    async def _summarize(self, messages: List[Message], exchanges: int) -> None:
        """Add notes on `messages` after the existing ones, which are kept word for word
        unless all the notes together run long and get condensed."""
        started = time.perf_counter()
        previous = self.memory.summary
        notes = await self._summary_call(
            SUMMARY_INSTRUCTIONS,
            SUMMARY_INPUT.format(
                facts=self.context.user_facts() or "(nothing yet)",
                summary=previous or "(nothing yet)",
                conversation=transcript(messages, self._user_name, self.ai_name),
            ),
        )
        if not notes:
            return
        summary = f"{previous}\n{notes}" if previous else notes
        if len(summary.split()) > SUMMARY_CONDENSE_AFTER_WORDS:
            shorter = await self._summary_call(SUMMARY_CONDENSE_INSTRUCTIONS, summary)
            if shorter and len(shorter.split()) < len(summary.split()):
                summary = shorter
        self.memory.set_summary(summary, exchanges)
        logger.info(
            f"Summarized the first {exchanges} exchanges in {len(summary.split())} words "
            f"({time.perf_counter() - started:.1f}s)"
        )

    async def _summary_call(self, instructions: str, content: str) -> str:
        """The talker's answer, or '' (logged) if it failed or was empty."""
        request = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ]
        try:
            text = await self.talker.complete(request, max_tokens=SUMMARY_MAX_TOKENS)
        except Exception as e:
            logger.warning(f"Conversation summary failed: {e}")
            return ""
        text = (text or "").strip()
        if not text:
            logger.warning("Conversation summary came back empty")
        return text

    def _cancel_summary(self) -> None:
        if self._summary_task is not None:
            try:
                self._summary_task.cancel()
            except RuntimeError:  # its event loop is already closed
                pass
            self._summary_task = None

    def _record_reply(self, text: str) -> None:
        """Store this turn's reply; an interrupt from now on edits it instead of the
        previous one."""
        if self._early_opener:
            text = f"{self._early_opener} {(text or '').strip()}".strip()
            self._early_opener = ""
        self.memory.add("assistant", text)
        self._reply_recorded = True

    def _build_pipeline(self):
        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @strip_unknown_tags(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def pipeline(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
            async for piece in tokens:
                yield piece

        return pipeline

    # ---- acknowledgements ----

    async def generate_ack_text(self) -> str:
        """An 'on it' line that never sees the user's words, so it can't invent a result."""
        messages = [
            {"role": "system", "content": self.context.system_prompt()},
            {"role": "user", "content": ACK_USER_TURN},
            {"role": "system", "content": ACK_INSTRUCTION},
        ]
        return await self.talker.complete(messages, max_tokens=ACK_MAX_TOKENS)

    def set_tts_engine(self, engine) -> None:
        self._tts_engine = engine
        if self._prerender_acks:
            self.ack_pool.set_synthesizer(
                self._synthesize_ack if engine is not None else None
            )
        if self.openers is not None:
            if engine is None:
                self.openers.use(None, "")
            else:
                self.openers.use(*engine_renderer(engine, self.openers.carrier))

    async def _synthesize_ack(self, text: str) -> Optional[str]:
        engine = self._tts_engine
        if engine is None:
            return None
        return await engine.async_generate_audio(
            fix_pronunciation(text), file_name_no_ext=f"ack_{uuid.uuid4().hex[:8]}"
        )

    # ---- task results listener (set by the websocket layer) ----

    def set_task_listener(self, listener: Optional[Callable[[], None]]) -> None:
        self._task_listener = listener
        if listener is None:
            self._stop_warm()
            return
        self._start_warm()
        self.ack_pool.fill()
        if self.tasks.untold_results():
            listener()

    def remove_task_listener(self, listener: Callable[[], None]) -> None:
        if self._task_listener == listener:
            self.set_task_listener(None)

    def has_untold_results(self) -> bool:
        return bool(self.tasks.untold_results() or self.tasks.untold_approvals())

    def _on_task_update(self, rec: TaskRecord) -> None:
        """A task finished or is waiting for the user's approval."""
        if self._task_listener is not None:
            self._task_listener()

    # ---- connection warm-up ----

    def _start_warm(self) -> None:
        if self._warm_up is None:
            return
        loop = asyncio.get_running_loop()
        if (
            self._warm_task is not None
            and not self._warm_task.done()
            and self._warm_task.get_loop() is loop
        ):
            return
        self._warm_task = loop.create_task(self._keep_warm())

    def _stop_warm(self) -> None:
        if self._warm_task is not None:
            try:
                self._warm_task.cancel()
            except RuntimeError:  # its event loop is already closed
                pass
            self._warm_task = None

    async def _keep_warm(self) -> None:
        while True:
            await self._warm_up()
            await asyncio.sleep(WARM_INTERVAL_S)

    async def close(self) -> None:
        self._stop_warm()
        self._cancel_summary()
        self.tasks.close()
        self.ack_pool.clear()
