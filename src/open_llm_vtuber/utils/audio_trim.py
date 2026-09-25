"""Trimming the silence TTS engines pad their clips with.

Azure and edge-tts voices start ~110-150 ms after the clip does (heard as latency) and
end ~850 ms before it does (heard as a long gap before the next sentence). Their
padding is digital silence, so a -60 dBFS threshold finds the voice without clipping
soft onsets like the "h" in "Hmph".
"""

from pydub import AudioSegment
from pydub.silence import detect_leading_silence

FLOOR_DBFS = -60.0
LEAD_KEEP_MS = 25
TAIL_KEEP_MS = 200  # about a short spoken pause between sentences
CHUNK_MS = 5


def trim_padding(
    audio: AudioSegment,
    lead_keep_ms: int = LEAD_KEEP_MS,
    tail_keep_ms: int = TAIL_KEEP_MS,
    floor_dbfs: float = FLOOR_DBFS,
) -> AudioSegment:
    """`audio` without its leading and trailing silence beyond the amounts kept.
    Clips that are all silence come back unchanged."""
    start = detect_leading_silence(audio, floor_dbfs, CHUNK_MS)
    end = detect_leading_silence(audio.reverse(), floor_dbfs, CHUNK_MS)
    if start + end >= len(audio):
        return audio
    return audio[max(0, start - lead_keep_ms) : len(audio) - max(0, end - tail_keep_ms)]
