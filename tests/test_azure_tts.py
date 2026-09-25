"""Azure TTS over REST: one kept-alive connection, raw PCM saved as trimmed WAV,
edge-tts fallback.

Run directly:  .venv/Scripts/python.exe tests/test_azure_tts.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
from pydub import AudioSegment  # noqa: E402
from pydub.generators import Sine  # noqa: E402

from _harness import run_module  # noqa: E402
from src.open_llm_vtuber.tts.azure_tts import TTSEngine  # noqa: E402


def pcm_bytes(lead_ms=0, tone_ms=200, tail_ms=0):
    """Raw 24 kHz 16-bit mono PCM, as Azure sends for raw-24khz-16bit-mono-pcm."""
    tone = Sine(440).to_audio_segment(duration=tone_ms, volume=-6)
    tone = tone.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    clip = (
        AudioSegment.silent(lead_ms, 24000) + tone + AudioSegment.silent(tail_ms, 24000)
    )
    return clip.raw_data


class FakeAzure:
    def __init__(self, status=200, headers=None, body=None):
        self.status = status
        self.headers = headers or {}
        self.body = body if body is not None else pcm_bytes()
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, headers=self.headers, text="nope")
        return httpx.Response(200, content=self.body)


class FakeEdge:
    def __init__(self, folder):
        self.folder = folder
        self.texts = []
        self.prefixes = []

    def generate_audio(self, text, file_name_no_ext=None):
        self.texts.append(text)
        path = os.path.join(self.folder, f"edge_{len(self.texts)}.mp3")
        with open(path, "wb") as f:
            f.write(b"ID3")
        return path

    def generate_prefix_audio(self, prefix, carrier, file_name_no_ext):
        self.prefixes.append((prefix, carrier))
        return os.path.join(self.folder, "prefix.wav")


def make_engine(fake, tmp, fallback=None, **kw):
    engine = TTSEngine(
        "azure-key",
        "centralindia",
        "en-GB-MaisieNeural",
        pitch="0",
        rate="1",
        transport=httpx.MockTransport(fake.handler),
        fallback=fallback,
        **kw,
    )
    engine.new_audio_dir = tmp
    return engine


def test_synthesizes_wav_with_the_configured_voice():
    fake = FakeAzure()
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(fake, tmp)
        path = engine.generate_audio("Fish & chips <now>", "s1")
        assert path.endswith(".wav") and os.path.exists(path), path
        with open(path, "rb") as f:
            assert f.read(4) == b"RIFF"
    request = fake.requests[0]
    assert str(request.url) == (
        "https://centralindia.tts.speech.microsoft.com/cognitiveservices/v1"
    )
    assert request.headers["ocp-apim-subscription-key"] == "azure-key"
    assert request.headers["x-microsoft-outputformat"] == "raw-24khz-16bit-mono-pcm"
    ssml = request.content.decode("utf-8")
    assert 'name="en-GB-MaisieNeural"' in ssml and 'xml:lang="en-GB"' in ssml
    assert "Fish &amp; chips &lt;now&gt;" in ssml, "text is escaped for SSML"


def test_padding_silence_is_trimmed():
    # Azure starts the voice ~140 ms in and pads ~850 ms after it.
    fake = FakeAzure(body=pcm_bytes(lead_ms=140, tone_ms=500, tail_ms=850))
    with tempfile.TemporaryDirectory() as tmp:
        path = make_engine(fake, tmp).generate_audio("hello", "s1")
        audio = AudioSegment.from_file(path)
        assert audio.frame_rate == 24000 and audio.channels == 1
        assert 500 + 25 + 190 <= len(audio) <= 500 + 25 + 210, len(audio)
        lead = audio[:40]
        assert lead[:20].dBFS < -60 and lead[25:].dBFS > -30, "25 ms kept before voice"


def test_reuses_one_http_client_so_the_connection_stays_warm():
    fake = FakeAzure()
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(fake, tmp)
        client = engine._client
        engine.generate_audio("one", "a")
        engine.generate_audio("two", "b")
        assert engine._client is client and len(fake.requests) == 2


def test_failure_falls_back_to_edge_tts():
    fake = FakeAzure(status=500)
    with tempfile.TemporaryDirectory() as tmp:
        edge = FakeEdge(tmp)
        engine = make_engine(fake, tmp, fallback=edge)
        path = engine.generate_audio("hello", "s1")
        assert path and path.endswith(".mp3"), path
        assert edge.texts == ["hello"]


def test_failure_without_fallback_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(FakeAzure(status=401), tmp)
        assert engine.generate_audio("hello", "s1") is None


def test_quota_or_rate_limit_pauses_azure_for_a_while():
    fake = FakeAzure(status=429, headers={"Retry-After": "30"})
    with tempfile.TemporaryDirectory() as tmp:
        edge = FakeEdge(tmp)
        engine = make_engine(fake, tmp, fallback=edge)
        engine.generate_audio("one", "a")
        engine.generate_audio("two", "b")
        assert len(fake.requests) == 1, "no Azure call while it's paused"
        assert edge.texts == ["one", "two"]


def test_prefix_audio_uses_the_edge_voice_with_word_timings():
    with tempfile.TemporaryDirectory() as tmp:
        edge = FakeEdge(tmp)
        engine = make_engine(FakeAzure(), tmp, fallback=edge)
        assert engine.generate_prefix_audio("Humph,", "I suppose.", "x")
        assert edge.prefixes == [("Humph,", "I suppose.")]
        no_edge = make_engine(FakeAzure(), tmp)
        assert not hasattr(no_edge, "generate_prefix_audio") or (
            no_edge.generate_prefix_audio("Humph,", "I suppose.", "x") is None
        )


def test_factory_adds_an_edge_fallback_with_the_configured_voice():
    from src.open_llm_vtuber.config_manager.tts import AzureTTSConfig
    from src.open_llm_vtuber.tts.edge_tts import TTSEngine as EdgeEngine
    from src.open_llm_vtuber.tts.tts_factory import TTSFactory

    cfg = AzureTTSConfig(
        api_key="k",
        region="centralindia",
        voice="en-GB-MaisieNeural",
        pitch="0",
        rate="1",
        edge_fallback_voice="en-GB-MaisieNeural",
    )
    engine = TTSFactory.get_tts_engine("azure_tts", **cfg.model_dump())
    assert isinstance(engine.fallback, EdgeEngine)
    assert engine.fallback.voice == "en-GB-MaisieNeural"

    cfg.edge_fallback_voice = ""
    assert TTSFactory.get_tts_engine("azure_tts", **cfg.model_dump()).fallback is None


if __name__ == "__main__":
    run_module(globals())
