"""Azure neural TTS over the REST API.

One kept-alive HTTPS connection is reused for every sentence (no TLS handshake per
sentence), and audio comes back as 24 kHz WAV, so no MP3 decode is needed downstream.

Azure serves the same neural voices as edge-tts (e.g. en-GB-MaisieNeural), so an edge-tts
engine with the same voice can stand in when Azure fails, hits its rate limit, or runs
out of free-tier quota.
"""

import os
import time
from typing import Optional
from xml.sax.saxutils import escape, quoteattr

import httpx
from loguru import logger

from .tts_interface import TTSInterface

OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"
RATE_LIMIT_PAUSE_S = 60.0  # after a 429 without Retry-After
AUTH_PAUSE_S = 600.0  # after 401/403 (bad key, or the free quota is used up)


class TTSEngine(TTSInterface):
    temp_audio_file = "temp"
    file_extension = "wav"
    new_audio_dir = "cache"

    def __init__(
        self,
        api_key,
        region,
        voice,
        pitch=0,
        rate=1.0,
        fallback: Optional[TTSInterface] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        """
        api_key, region: the Azure Speech resource's key and region (e.g. centralindia)
        voice: Azure voice name, e.g. en-GB-MaisieNeural
        pitch: pitch adjustment in percent; rate: speaking rate (1 = normal)
        fallback: engine used while Azure is failing (e.g. edge-tts with the same voice)
        """
        self.api_key = api_key
        self.voice = voice
        self.pitch = pitch
        self.rate = rate
        self.fallback = fallback
        self.url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(10.0, connect=3.0),
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0),
        )
        self._paused_until = 0.0

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Synthesize `text` to a wav file (or the fallback's file while Azure is failing).

        Returns:
        str: path to the audio file, or None if nothing could synthesize it
        """
        if time.monotonic() >= self._paused_until:
            path = self._azure(text, file_name_no_ext)
            if path:
                return path
        if self.fallback is not None:
            return self.fallback.generate_audio(text, file_name_no_ext)
        return None

    def generate_prefix_audio(self, prefix, carrier, file_name_no_ext):
        """Openers need word timings, which Azure's REST API doesn't give; the edge-tts
        fallback speaks with the same voice and does."""
        render = getattr(self.fallback, "generate_prefix_audio", None)
        return render(prefix, carrier, file_name_no_ext) if render else None

    def _azure(self, text, file_name_no_ext) -> Optional[str]:
        try:
            response = self._client.post(
                self.url,
                content=self._ssml(text).encode("utf-8"),
                headers={
                    "Ocp-Apim-Subscription-Key": self.api_key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
                    "User-Agent": "Open-LLM-VTuber",
                },
            )
        except httpx.HTTPError as e:
            logger.warning(f"Azure TTS request failed: {type(e).__name__}: {e}")
            return None
        if response.status_code != 200:
            self._pause_after(response)
            logger.warning(
                f"Azure TTS HTTP {response.status_code}: {response.text[:200]!r}"
            )
            return None
        path = os.path.join(
            self.new_audio_dir,
            f"{file_name_no_ext or self.temp_audio_file}.{self.file_extension}",
        )
        with open(path, "wb") as f:
            f.write(response.content)
        return path

    def _pause_after(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            try:
                pause = float(response.headers.get("Retry-After", RATE_LIMIT_PAUSE_S))
            except ValueError:
                pause = RATE_LIMIT_PAUSE_S
        elif response.status_code in (401, 403):
            pause = AUTH_PAUSE_S
        else:
            return
        self._paused_until = time.monotonic() + pause
        logger.warning(
            f"Azure TTS paused for {pause:.0f}s"
            + (" (using the fallback voice)" if self.fallback else "")
        )

    def _ssml(self, text: str) -> str:
        lang = "-".join(self.voice.split("-")[:2]) or "en-US"
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f"xml:lang={quoteattr(lang)}><voice name={quoteattr(self.voice)}>"
            f'<prosody pitch="{self.pitch}%" rate="{self.rate}">'
            f"{escape(text.strip())}</prosody></voice></speak>"
        )
