# config_manager/filler.py
from pydantic import Field, model_validator
from typing import Dict, ClassVar, List
from .i18n import I18nMixin, Description


class FillerConfig(I18nMixin):
    """Pre-rendered backchannel clips played while the real response is generated."""

    enabled: bool = Field(False, alias="enabled")
    probability: float = Field(0.7, alias="probability")
    no_repeat_window: int = Field(2, alias="no_repeat_window")
    escalation_delay: float = Field(4.0, alias="escalation_delay")
    max_tier0_ms: int = Field(900, alias="max_tier0_ms")
    tier0_phrases: List[str] = Field(
        # edge-tts spells out "Mm." / "Mm-hm." letter by letter; these read as sounds.
        default_factory=lambda: [
            "Hmm.",
            "Okay.",
            "Mmm.",
            "Mmm...",
            "Hmmm.",
            "Hmm hmm.",
            "Uh-huh.",
            "Ah.",
            "Oh.",
            "Right.",
            "I see.",
        ],
        alias="tier0_phrases",
    )
    escalation_phrases: List[str] = Field(
        default_factory=lambda: [
            "Hmm, give me a sec.",
            "Hmm, let me think.",
            "One moment.",
        ],
        alias="escalation_phrases",
    )
    tool_phrases: List[str] = Field(
        default_factory=lambda: ["Let me look that up.", "Let me check."],
        alias="tool_phrases",
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Play short filler sounds while the response is being generated",
            zh="在生成回复时播放简短的语气词",
        ),
        "probability": Description(
            en="Chance (0-1) of playing the instant filler on a voice turn",
            zh="语音对话中播放即时语气词的概率（0-1）",
        ),
        "no_repeat_window": Description(
            en="Number of recent clips that won't be picked again",
            zh="最近播放过、不会被再次选中的语气词数量",
        ),
        "escalation_delay": Description(
            en="Seconds without a response before playing one 'still thinking' clip",
            zh="多少秒仍无回复时播放一次“还在想”语句",
        ),
        "max_tier0_ms": Description(
            en="Instant filler clips longer than this (ms) are dropped",
            zh="超过此时长（毫秒）的即时语气词将被丢弃",
        ),
        "tier0_phrases": Description(
            en="Neutral phrases played instantly when the user stops speaking",
            zh="用户停止说话时立即播放的中性语气词",
        ),
        "escalation_phrases": Description(
            en="Phrases played once if the response is slow",
            zh="回复较慢时播放一次的语句",
        ),
        "tool_phrases": Description(
            en="Phrases played when a tool call starts",
            zh="工具调用开始时播放的语句",
        ),
    }

    @model_validator(mode="after")
    def check_values(cls, values):
        if not 0.0 <= values.probability <= 1.0:
            raise ValueError("filler probability must be between 0 and 1")
        if values.no_repeat_window < 0:
            raise ValueError("filler no_repeat_window must be >= 0")
        if values.escalation_delay <= 0:
            raise ValueError("filler escalation_delay must be > 0")
        return values
