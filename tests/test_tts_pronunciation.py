"""Respelling words that TTS engines read out letter by letter (edge-tts says "Hmph" as
H-M-P-H). Only the TTS text changes; subtitles and memory keep the original.

Run directly:  .venv/Scripts/python.exe tests/test_tts_pronunciation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.utils.tts_preprocessor import (  # noqa: E402
    fix_pronunciation,
    tts_filter,
)


def filtered(text):
    return tts_filter(
        text,
        remove_special_char=False,
        ignore_brackets=False,
        ignore_parentheses=False,
        ignore_asterisks=False,
        ignore_angle_brackets=False,
    )


def test_hmph_is_respelled_as_humph():
    assert fix_pronunciation("Hmph, fine.") == "Humph, fine."
    assert fix_pronunciation("fine... hmph.") == "fine... humph."
    assert fix_pronunciation("Hmph.Anyway, hmph!") == "Humph.Anyway, humph!"


def test_spelled_out_variants_are_respelled_too():
    for word in ["HMPH", "Hmmph", "Hmmmph", "Hmpf", "Hmf", "Hmff"]:
        assert fix_pronunciation(f"{word}, fine.") == "Humph, fine.", word


def test_words_edge_tts_already_says_are_left_alone():
    for text in ["Hmm, fine.", "Humph.", "hmm", "Oh, I see.", "Thompson helped."]:
        assert fix_pronunciation(text) == text, text


def test_tts_filter_applies_it():
    assert filtered("Hmph, sure you didn't.") == "Humph, sure you didn't."


if __name__ == "__main__":
    run_module(globals())
