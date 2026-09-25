"""Openers: the reply's first interjection ("Hmph,") plays from a pre-rendered clip.

Run directly:  .venv/Scripts/python.exe tests/test_realtime_openers.py
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydub import AudioSegment  # noqa: E402
from pydub.generators import Sine  # noqa: E402

from _harness import run_module  # noqa: E402
from realtime_fakes import wait_until  # noqa: E402
from test_realtime_agent import (  # noqa: E402
    FakeRouter,
    FakeTalker,
    collect,
    make_agent,
    ready_ack,
    spoken,
    trailing,
    user_turn,
)
from src.open_llm_vtuber.agent.agents.realtime import realtime_agent  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.router import Route  # noqa: E402
from src.open_llm_vtuber.agent.agents.realtime.openers import (  # noqa: E402
    UNDECIDED,
    OpenerLibrary,
    cut_after_words,
    match_opener,
    opener_instruction,
    opener_key,
)
from src.open_llm_vtuber.agent.output_types import AudioOutput, SentenceOutput  # noqa: E402

KEYS = {"hmph,", "huh?", "oh,", "fine,"}


# ---- matching the start of the reply ----


def test_opener_keys_ignore_case_and_keep_only_question_marks():
    assert opener_key("Hmph,") == "hmph,"
    assert opener_key("Hmph!") == "hmph,"
    assert opener_key("Huh?") == "huh?"
    assert opener_key("Oh...") == "oh,"


def test_matches_a_known_interjection_and_keeps_the_rest():
    m = match_opener("Hmph, fine then", KEYS)
    assert (m.key, m.text, m.rest) == ("hmph,", "Hmph,", " fine then"), m


def test_question_mark_openers_match_their_own_clip():
    m = match_opener("Huh? What", KEYS)
    assert (m.key, m.text) == ("huh?", "Huh?"), m


def test_other_punctuation_uses_the_comma_clip_but_shows_what_was_written():
    m = match_opener("Hmph! No way", KEYS)
    assert (m.key, m.text) == ("hmph,", "Hmph!"), m


def test_waits_for_more_text_while_the_first_word_is_incomplete():
    for buffer in ("", "Hm", "Hmph", "[jo", "[joy] Hmp", "Hmph,"):
        assert match_opener(buffer, KEYS) is UNDECIDED, buffer


def test_decides_at_the_end_of_the_stream():
    m = match_opener("Hmph.", KEYS, final=True)
    assert (m.key, m.rest) == ("hmph,", ""), m
    assert match_opener("Hm", KEYS, final=True) is None


def test_unknown_or_unpunctuated_first_words_are_not_openers():
    assert match_opener("Hello, there", KEYS) is None
    assert match_opener("Hmph fine", KEYS) is None
    assert match_opener("It's raining", KEYS) is None


def test_leading_expression_tags_belong_to_the_opener():
    m = match_opener("[joy] Oh, nice", KEYS)
    assert (m.key, m.tags.strip(), m.rest) == ("oh,", "[joy]", " nice"), m


def test_opener_instruction_lists_the_phrases():
    text = opener_instruction(["Hmph,", "Huh?"])
    assert "Hmph," in text and "Huh?" in text


# ---- cutting an interjection out of a carrier sentence ----


def test_cut_keeps_the_first_word_and_part_of_the_pause():
    audio = Sine(300).to_audio_segment(duration=2000)
    words = [(0.1, 0.4), (0.8, 0.1), (0.9, 0.4)]  # (offset s, duration s)
    clip = cut_after_words(audio, words, 1)
    # 0.07 s .. 0.5 s + min(0.12, half of the 0.3 s gap) = 0.07 .. 0.62
    assert abs(len(clip) - 550) <= 5, len(clip)


def test_cutting_the_last_word_treats_the_audio_end_as_the_next_word():
    audio = Sine(300).to_audio_segment(duration=1000)
    clip = cut_after_words(audio, [(0.1, 0.5)], 1)
    # 0.07 s .. 0.6 s + min(0.12, half of the 0.4 s left) = 0.07 .. 0.72
    assert abs(len(clip) - 650) <= 5, len(clip)


# ---- the clip library ----


def tone_file(folder, name):
    path = os.path.join(folder, name + ".wav")
    Sine(440).to_audio_segment(duration=300).export(path, format="wav")
    return path


async def test_library_renders_each_phrase_once_and_caches_it_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        rendered = []

        async def render(phrase):
            rendered.append(phrase)
            return tone_file(tmp, f"r{len(rendered)}")

        cache = os.path.join(tmp, "openers")
        library = OpenerLibrary(["Hmph,", "Huh?"], render, cache, cache_key="voice-a")
        await library.build()
        assert rendered == ["Hmph,", "Huh?"]
        assert library.keys() == {"hmph,", "huh?"}
        path = library.clip("hmph,")
        assert path and os.path.exists(path) and path.startswith(cache)
        assert AudioSegment.from_file(path).duration_seconds > 0.2

        again = OpenerLibrary(["Hmph,", "Huh?"], render, cache, cache_key="voice-a")
        await again.build()
        assert rendered == ["Hmph,", "Huh?"], "cached clips are not re-rendered"

        other_voice = OpenerLibrary(["Hmph,"], render, cache, cache_key="voice-b")
        await other_voice.build()
        assert len(rendered) == 3, "a different voice renders its own clips"


async def test_library_skips_phrases_that_fail_to_render():
    with tempfile.TemporaryDirectory() as tmp:

        async def render(phrase):
            if phrase == "Huh?":
                raise RuntimeError("tts down")
            return tone_file(tmp, "ok")

        library = OpenerLibrary(["Hmph,", "Huh?"], render, tmp, cache_key="v")
        await library.build()
        assert library.keys() == {"hmph,"}


# ---- in a turn ----


def ready_library(folder, *phrases):
    library = OpenerLibrary(list(phrases), None, folder, cache_key="v")
    for phrase in phrases:
        library._clips[opener_key(phrase)] = tone_file(folder, opener_key(phrase)[:-1])
    return library


async def test_reply_opener_plays_its_clip_first():
    with tempfile.TemporaryDirectory() as tmp:
        library = ready_library(tmp, "Hmph,")
        agent, _ = make_agent(
            talker=FakeTalker(reply="Hmph, hello there, dummy."), openers=library
        )
        outputs = await collect(agent, user_turn("hi yuna"))
        first = outputs[0]
        assert isinstance(first, AudioOutput), outputs
        assert first.audio_path == library.clip("hmph,")
        assert first.transcript == "Hmph," and first.display_text.text == "Hmph,"
        assert os.path.exists(first.audio_path), "cached clips are never deleted"
        rest = [o for o in outputs[1:] if isinstance(o, SentenceOutput)]
        assert rest and "Hmph" not in " ".join(o.display_text.text for o in rest)
        assert spoken(outputs).startswith("Hmph, hello there"), spoken(outputs)
        assert agent.memory.messages[-1]["content"] == "Hmph, hello there, dummy."


async def test_spoken_replies_are_reminded_to_open_with_an_interjection():
    with tempfile.TemporaryDirectory() as tmp:
        talker = FakeTalker(reply="Hmph, hi.")
        agent, _ = make_agent(talker=talker, openers=ready_library(tmp, "Hmph,"))
        await collect(agent, user_turn("hi yuna"))
        messages = talker.calls[0]["messages"]
        assert messages[-2] == {"role": "user", "content": "hi yuna"}, messages
        assert messages[-1]["role"] == "system", "the reminder follows the user turn"
        assert "after any expression keyword" in messages[-1]["content"]
        assert "Hmph," in messages[-1]["content"]


async def test_no_reminder_without_openers():
    talker = FakeTalker(reply="Hello.")
    agent, _ = make_agent(talker=talker)
    await collect(agent, user_turn("hi yuna"))
    assert talker.calls[0]["messages"][-1] == {"role": "user", "content": "hi yuna"}


class quick_early_opener:
    """Shrinks EARLY_OPENER_AFTER_S so a slow route takes milliseconds, not seconds."""

    def __enter__(self):
        self.saved = realtime_agent.EARLY_OPENER_AFTER_S
        realtime_agent.EARLY_OPENER_AFTER_S = 0.05

    def __exit__(self, *exc):
        realtime_agent.EARLY_OPENER_AFTER_S = self.saved


async def test_slow_route_speaks_the_opener_before_it_arrives():
    with tempfile.TemporaryDirectory() as tmp, quick_early_opener():
        router = FakeRouter(Route("chat"), delay=0.3)
        agent, _ = make_agent(
            router=router,
            talker=FakeTalker(reply="Hmph, fine, hello there."),
            openers=ready_library(tmp, "Hmph,", "Fine,"),
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        stream = agent.chat(user_turn("hi yuna"))
        first = await stream.__anext__()
        assert loop.time() - started < 0.2, "the opener didn't wait for the route"
        assert isinstance(first, AudioOutput) and first.transcript == "Hmph,"
        rest = [out async for out in stream]
        assert spoken([first] + rest) == "Hmph, fine, hello there.", spoken(rest)
        assert agent.memory.messages[-1] == {
            "role": "assistant",
            "content": "Hmph, fine, hello there.",
        }


async def test_opener_spoken_early_is_not_repeated_by_an_instructed_reply():
    with tempfile.TemporaryDirectory() as tmp, quick_early_opener():
        talker = FakeTalker(reply="Huh? what do you mean?")
        agent, _ = make_agent(
            router=FakeRouter(Route("unsure", p_task=0.5), delay=0.3),
            talker=talker,
            openers=ready_library(tmp, "Huh?"),
        )
        outputs = await collect(agent, user_turn("the thing"))
        assert isinstance(outputs[0], AudioOutput), outputs
        hint = trailing(talker.calls[-1])
        assert 'already said "Huh?"' in hint, hint
        assert agent.memory.messages[-1]["content"].startswith("Huh? "), (
            agent.memory.messages
        )


async def test_early_opener_then_task_ack():
    with tempfile.TemporaryDirectory() as tmp, quick_early_opener():
        agent, _ = make_agent(
            router=FakeRouter(Route("new_task", p_task=0.9), delay=0.3),
            talker=FakeTalker(reply="Oh, the weather is sunny."),
            openers=ready_library(tmp, "Oh,"),
        )
        ready_ack(agent, tmp)
        outputs = await collect(agent, user_turn("weather in delhi?"))
        assert [o.transcript for o in outputs] == ["Oh,", "Ugh, fine. On it."], outputs
        assert agent.memory.messages[-1]["content"] == "Oh, Ugh, fine. On it."
        assert "sunny" not in str(agent.memory.messages)


async def test_fast_route_waits_as_before():
    with tempfile.TemporaryDirectory() as tmp, quick_early_opener():
        agent, _ = make_agent(
            router=FakeRouter(Route("new_task", p_task=0.9)),
            talker=FakeTalker(reply="Oh, the weather is sunny."),
            openers=ready_library(tmp, "Oh,"),
        )
        ready_ack(agent, tmp)
        outputs = await collect(agent, user_turn("weather in delhi?"))
        assert [o.transcript for o in outputs] == ["Ugh, fine. On it."], outputs


async def test_no_early_opener_when_the_reply_has_none():
    with tempfile.TemporaryDirectory() as tmp, quick_early_opener():
        agent, _ = make_agent(
            router=FakeRouter(Route("chat"), delay=0.2),
            talker=FakeTalker(reply="Well hello there."),
            openers=ready_library(tmp, "Hmph,"),
        )
        outputs = await collect(agent, user_turn("hi"))
        assert all(isinstance(o, SentenceOutput) for o in outputs), outputs
        assert "Well hello there." in spoken(outputs)


async def test_opener_takes_the_expression_of_leading_tags():
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = make_agent(
            talker=FakeTalker(reply="[joy] Hmph, fine."),
            openers=ready_library(tmp, "Hmph,"),
        )
        outputs = await collect(agent, user_turn("you like it?"))
        assert isinstance(outputs[0], AudioOutput)
        assert outputs[0].actions.expressions == [3], outputs[0].actions
        assert "[joy]" not in spoken(outputs)


async def test_replies_without_a_known_opener_are_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = make_agent(
            talker=FakeTalker(reply="Well, hello there."),
            openers=ready_library(tmp, "Hmph,"),
        )
        outputs = await collect(agent, user_turn("hi"))
        assert all(isinstance(o, SentenceOutput) for o in outputs), outputs
        assert "Well, hello there." in spoken(outputs)


async def test_a_reply_that_is_only_the_opener_is_just_the_clip():
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = make_agent(
            talker=FakeTalker(reply="Hmph."), openers=ready_library(tmp, "Hmph,")
        )
        outputs = await collect(agent, user_turn("say something"))
        assert len(outputs) == 1 and isinstance(outputs[0], AudioOutput), outputs
        assert outputs[0].transcript == "Hmph."


async def test_openers_render_through_the_tts_engine_once_it_is_set():
    class FakeTTS:
        voice = "en-GB-MaisieNeural"

        def __init__(self, folder):
            self.folder = folder
            self.texts = []

        async def async_generate_audio(self, text, file_name_no_ext=None):
            self.texts.append(text)
            return tone_file(self.folder, file_name_no_ext)

    with tempfile.TemporaryDirectory() as tmp:
        library = OpenerLibrary(["Hmph,"], None, os.path.join(tmp, "c"), "v")
        agent, _ = make_agent(openers=library)
        tts = FakeTTS(tmp)
        agent.set_tts_engine(tts)
        await wait_until(lambda: library.keys() == {"hmph,"})
        assert tts.texts == ["Humph,"], "rendered with the pronunciation fix"


async def test_engines_without_word_timings_render_the_opener_alone():
    from src.open_llm_vtuber.agent.agents.realtime.openers import engine_renderer

    class NoTimingsTTS:
        voice = "v"

        def __init__(self, folder):
            self.folder = folder
            self.texts = []

        def generate_prefix_audio(self, prefix, carrier, file_name_no_ext):
            return None  # e.g. Azure without an edge-tts fallback

        async def async_generate_audio(self, text, file_name_no_ext=None):
            self.texts.append(text)
            return tone_file(self.folder, file_name_no_ext)

    with tempfile.TemporaryDirectory() as tmp:
        tts = NoTimingsTTS(tmp)
        render, _ = engine_renderer(tts)
        path = await render("Hmph,")
        assert path and os.path.exists(path), path
        assert tts.texts == ["Humph,"]


if __name__ == "__main__":
    run_module(globals())
