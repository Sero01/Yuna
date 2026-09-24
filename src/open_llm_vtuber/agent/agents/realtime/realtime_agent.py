"""Real-time conversation agent.

Each turn starts Jev (routing) and the talker (a speculative reply) at the same moment.
Chat turns release the talker's reply as soon as Jev agrees; task turns discard it, start
a Hermes run, and speak a pre-made acknowledgement instead. Finished tasks, and commands
Hermes needs permission for, are announced through a separate `task-result` turn started
by the websocket layer; the user's spoken yes/no answers the approval (once or deny only).
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
)
from .prompts import (
    ACK_FALLBACK_TEXT,
    ACK_INSTRUCTION,
    ACK_USER_TURN,
    APPROVAL_EXPIRED_INSTRUCTION,
    APPROVAL_REQUEST_INSTRUCTION,
    APPROVED_INSTRUCTION,
    CANCEL_INSTRUCTION,
    CHANGE_INSTRUCTION,
    DENIED_INSTRUCTION,
    FALLBACK_LINE,
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
WARM_INTERVAL_S = 30.0
RECENT_MESSAGES_FOR_ROUTER = 6

_END = object()
Output = Union[SentenceOutput, AudioOutput, Dict[str, Any]]


class SpeculativeReply:
    """Streams the talker in the background, holding its tokens until released."""

    def __init__(self, tokens: AsyncIterator[str]):
        self._queue: asyncio.Queue = asyncio.Queue()
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

    async def release(self) -> AsyncIterator[str]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def cancel(self) -> None:
        self._task.cancel()


async def _single(text: str) -> AsyncIterator[str]:
    yield text


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
    ):
        super().__init__()
        self.router = router
        self.talker = talker
        self.tasks = tasks
        self.context = context
        self.memory = memory or ConversationMemory()
        self.max_turns = max_turns
        self.quiet_gap_s = quiet_gap_s
        self.ai_name = ai_name
        self.pending_offer: Optional[str] = None
        self.openers = openers
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
            ),
            tasks=TaskManager(
                http.get,
                cfg.hermes_base_url,
                hermes_key,
                timeout_s=cfg.task_timeout_s,
                max_active=cfg.max_active_tasks,
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
        )
        agent.set_tts_engine(tts_engine)
        return agent

    # ---- AgentInterface ----

    async def chat(self, input_data: BatchInput) -> AsyncIterator[Output]:
        self._interrupt_handled = False
        self._reply_recorded = False
        self.ack_pool.fill()
        if self.openers is not None:
            self.openers.ensure_built()
        metadata = input_data.metadata or {}
        if metadata.get("task_result"):
            async for out in self._task_result_turn():
                yield out
            return

        text = "\n".join(t.content for t in input_data.texts if t.content).strip()
        if input_data.images:
            logger.info("realtime_agent ignores images; the talker is text-only")
        history = self.memory.window(self.max_turns)
        if not metadata.get("skip_memory"):
            self.memory.add("user", text)

        if metadata.get("proactive_speak"):
            messages = self.context.build(
                history,
                task_state_message(self.tasks.recent_lines()),
                current_user=text,
            )
            async for out in self._speak(self.talker.stream(messages)):
                yield out
            return

        user_name = next((t.from_name for t in input_data.texts if t.from_name), None)
        async for out in self._routed_turn(text, history, user_name or "User"):
            yield out

    def handle_interrupt(self, heard_response: str) -> None:
        if self._interrupt_handled:
            return
        self._interrupt_handled = True
        self.memory.handle_interrupt(heard_response, replace_last=self._reply_recorded)

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        self.memory.load(conf_uid, history_uid)

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
        )
        speculative = SpeculativeReply(
            self.talker.stream(
                self.context.build(history, task_state, current_user=text)
            )
        )
        try:
            started = time.perf_counter()
            route = await self.router.decide(state)
            logger.info(
                f"Route: {route.kind} action={route.action} task={route.task_id} "
                f"p_task={route.p_task:.2f} via {route.source} "
                f"in {time.perf_counter() - started:.2f}s"
            )
            offer, self.pending_offer = self.pending_offer, None
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
                async for out in self._start_task(request, history, task_state, text):
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

    async def _start_task(
        self, request: str, history: List[Message], task_state: str, text: str
    ) -> AsyncIterator[Output]:
        try:
            await self.tasks.start(request, worker_history(history))
        except TaskStartError as e:
            logger.warning(f"Could not start a background task: {e}")
            instruction = TASK_START_FAILED_INSTRUCTION.format(reason=e)
            async for out in self._instructed(history, task_state, text, instruction):
                yield out
            return
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
                    worker_history(history),
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
            self.memory.window(self.max_turns),
            task_state_message(self.tasks.recent_lines()),
            trailing="\n\n".join(instructions),
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
            history, task_state, current_user=text, trailing=instruction
        )
        async for out in self._speak(
            self.talker.stream(messages, max_tokens=INSTRUCTED_MAX_TOKENS)
        ):
            yield out

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
        expressions = (
            self._live2d_model.extract_emotion(found.tags)
            if found.tags and self._live2d_model is not None
            else []
        )
        clip = AudioOutput(
            audio_path=self.openers.clip(found.key),
            display_text=DisplayText(text=found.text),
            transcript=found.text,
            actions=Actions(expressions=expressions or None),
        )
        return clip, rest()

    def _record_reply(self, text: str) -> None:
        """Store this turn's reply; an interrupt from now on edits it instead of the
        previous one."""
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
        self.tasks.close()
        self.ack_pool.clear()
