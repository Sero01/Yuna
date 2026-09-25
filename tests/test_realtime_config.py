"""Config for the real-time agent: defaults and both templates.

Run directly:  .venv/Scripts/python.exe tests/test_realtime_config.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.config_manager.agent import RealtimeAgentConfig  # noqa: E402
from src.open_llm_vtuber.config_manager.utils import read_yaml, validate_config  # noqa: E402

TEMPLATES = [
    os.path.join(ROOT, "config_templates", name)
    for name in ("conf.default.yaml", "conf.ZH.default.yaml")
]


def test_defaults_match_the_measured_choices():
    cfg = RealtimeAgentConfig()
    assert cfg.talker_model == "deepseek/deepseek-v4.1-flash"
    assert cfg.jev_model == "typesafe/jev-1.13"
    assert cfg.jev_url == "https://openrouter.ai/api/alpha/decisions"
    # 2026-09-24: 60 real exchanges (5.1K tokens) cost no measurable latency once cached.
    assert cfg.max_turns == 50
    # Past max_turns the oldest exchanges become a summary; the last 10 stay verbatim.
    assert cfg.summarize_history is True
    assert cfg.summary_keep_turns == 10
    assert cfg.worker_max_turns == 8
    # OpenRouter offers GLM 5.3 Flash only at low/high/max; "medium" was silently low.
    assert cfg.hermes_reasoning_effort == "high"
    assert cfg.remember_threshold == 0.5
    assert cfg.share_transcripts is True
    assert (cfg.unsure_low, cfg.unsure_high) == (0.35, 0.65)
    assert cfg.hedge_after_s == 1.5
    assert cfg.hermes_base_url == "http://localhost:8642"
    assert cfg.fallback_router_provider == "groq"
    # Jev's slowest successful answer was 0.61s; hedge the router just before that.
    assert cfg.jev_fallback_after_s == 0.6


def test_templates_expose_the_router_hedge_delay():
    for path in TEMPLATES:
        raw = read_yaml(path)["character_config"]["agent_config"]["agent_settings"]
        assert raw["realtime_agent"].get("jev_fallback_after_s") == 0.6, path


def test_templates_expose_the_summary_settings():
    for path in TEMPLATES:
        raw = read_yaml(path)["character_config"]["agent_config"]["agent_settings"]
        assert raw["realtime_agent"].get("summarize_history") is True, path
        assert raw["realtime_agent"].get("summary_keep_turns") == 10, path


def test_openers_are_off_unless_configured():
    assert RealtimeAgentConfig().openers == []
    for path in TEMPLATES:
        raw = read_yaml(path)["character_config"]["agent_config"]["agent_settings"]
        assert raw["realtime_agent"].get("openers") == [], path
        assert raw["realtime_agent"].get("opener_carrier"), path


def test_templates_contain_a_valid_realtime_block():
    for path in TEMPLATES:
        config = validate_config(read_yaml(path))
        rt = config.character_config.agent_config.agent_settings.realtime_agent
        assert rt is not None, f"{path}: missing realtime_agent block"
        assert rt.max_turns == 50, path
        assert rt.summarize_history is True, path
        assert rt.summary_keep_turns == 10, path
        assert rt.worker_max_turns == 8, path
        assert rt.hermes_reasoning_effort == "high", path
        assert rt.remember_threshold == 0.5, path
        assert rt.share_transcripts is True, path
        assert rt.talker_model == "deepseek/deepseek-v4.1-flash", path


def test_realtime_agent_is_an_accepted_choice():
    for path in TEMPLATES:
        data = read_yaml(path)
        data["character_config"]["agent_config"]["conversation_agent_choice"] = (
            "realtime_agent"
        )
        config = validate_config(data)
        assert (
            config.character_config.agent_config.conversation_agent_choice
            == "realtime_agent"
        )


if __name__ == "__main__":
    run_module(globals())
