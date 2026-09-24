"""Drop expression tags the Live2D model doesn't have (models invent [soft], [sigh], ...)."""

import re
from dataclasses import replace
from functools import wraps
from typing import Set

from ....utils.sentence_divider import SentenceWithTags, TagState

_TAG = re.compile(r"\[([^\[\]]{1,30})\]")


def strip_unknown_tags_from(text: str, known: Set[str]) -> str:
    cleaned = _TAG.sub(
        lambda m: m.group(0) if m.group(1).strip().lower() in known else "", text
    )
    return re.sub(r" {2,}", " ", cleaned) if cleaned != text else text


def strip_unknown_tags(live2d_model):
    """Decorator for SentenceWithTags streams; sentences left empty are dropped."""
    known = {str(k).lower() for k in (getattr(live2d_model, "emo_map", None) or {})}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async for item in func(*args, **kwargs):
                if isinstance(item, SentenceWithTags) and not any(
                    tag.state in (TagState.START, TagState.END) for tag in item.tags
                ):
                    text = strip_unknown_tags_from(item.text, known)
                    if text != item.text:
                        if not text.strip():
                            continue
                        item = replace(item, text=text)
                yield item

        return wrapper

    return decorator
