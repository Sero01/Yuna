import base64
import os
from typing import Dict, List, Optional, Tuple

from pydub import AudioSegment
from pydub.utils import make_chunks
from ..agent.output_types import Actions
from ..agent.output_types import DisplayText
from .audio_trim import trim_padding

# Clips sent over and over (openers), decoded once: path -> ((mtime_ns, size), base64, volumes)
_static_clips: Dict[str, Tuple[Tuple[int, int], str, List[float]]] = {}
STATIC_CHUNK_MS = 20


def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
    """
    Calculate the normalized volume (RMS) for each chunk of the audio.

    Parameters:
        audio (AudioSegment): The audio segment to process.
        chunk_length_ms (int): The length of each audio chunk in milliseconds.

    Returns:
        list: Normalized volumes for each chunk.
    """
    chunks = make_chunks(audio, chunk_length_ms)
    volumes = [chunk.rms for chunk in chunks]
    max_volume = max(volumes)
    if max_volume == 0:
        raise ValueError("Audio is empty or all zero.")
    return [volume / max_volume for volume in volumes]


def prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
) -> dict[str, any]:
    """
    Prepares the audio payload for sending to a broadcast endpoint.
    If audio_path is None, returns a payload with audio=None for silent display.

    Parameters:
        audio_path (str | None): The path to the audio file to be processed, or None for silent display
        chunk_length_ms (int): The length of each audio chunk in milliseconds
        display_text (DisplayText, optional): Text to be displayed with the audio
        actions (Actions, optional): Actions associated with the audio

    Returns:
        dict: The audio payload to be sent
    """
    if isinstance(display_text, DisplayText):
        display_text = display_text.to_dict()

    if not audio_path:
        # Return payload for silent display
        return {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "display_text": display_text,
            "actions": actions.to_dict() if actions else None,
            "forwarded": forwarded,
        }

    try:
        audio = AudioSegment.from_file(audio_path)
        audio_bytes = audio.export(format="wav").read()
    except Exception as e:
        raise ValueError(
            f"Error loading or converting generated audio file to wav file '{audio_path}': {e}"
        )
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    volumes = _get_volume_by_chunks(audio, chunk_length_ms)

    payload = {
        "type": "audio",
        "audio": audio_base64,
        "volumes": volumes,
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": actions.to_dict() if actions else None,
        "forwarded": forwarded,
    }

    return payload


def _stamp(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return st.st_mtime_ns, st.st_size


def preload_static_clip(audio_path: str, trim: bool = False) -> None:
    """Decode a clip that will be sent many times and keep its payload in memory, so
    sending it later costs no file read, WAV export or RMS pass. `trim` drops the TTS
    padding silence (the file itself is left as it is)."""
    audio = AudioSegment.from_file(audio_path)
    if trim:
        audio = trim_padding(audio)
    audio_base64 = base64.b64encode(audio.export(format="wav").read()).decode("utf-8")
    volumes = _get_volume_by_chunks(audio, STATIC_CHUNK_MS)
    _static_clips[os.path.abspath(audio_path)] = (
        _stamp(audio_path),
        audio_base64,
        volumes,
    )


def static_clip_payload(
    audio_path: str,
    display_text: DisplayText = None,
    actions: Actions = None,
) -> Optional[dict]:
    """The payload for a preloaded clip, or None if it isn't preloaded (or changed)."""
    entry = _static_clips.get(os.path.abspath(audio_path))
    if entry is None:
        return None
    try:
        if _stamp(audio_path) != entry[0]:
            return None
    except OSError:
        return None
    if isinstance(display_text, DisplayText):
        display_text = display_text.to_dict()
    return {
        "type": "audio",
        "audio": entry[1],
        "volumes": entry[2],
        "slice_length": STATIC_CHUNK_MS,
        "display_text": display_text,
        "actions": actions.to_dict() if actions else None,
        "forwarded": False,
    }


# Example usage:
# payload, duration = prepare_audio_payload("path/to/audio.mp3", display_text="Hello", expression_list=[0,1,2])
