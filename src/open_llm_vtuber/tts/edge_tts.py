import io
import sys
import os

import edge_tts
from loguru import logger
from pydub import AudioSegment

from ..utils.audio_cut import cut_after_words
from .tts_interface import TTSInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# Check out doc at https://github.com/rany2/edge-tts
# Use `edge-tts --list-voices` to list all available voices


class TTSEngine(TTSInterface):
    def __init__(self, voice="en-US-AvaMultilingualNeural"):
        self.voice = voice

        self.temp_audio_file = "temp"
        self.file_extension = "mp3"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext: str
            name of the file without extension


        Returns:
        str: the path to the generated audio file

        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        try:
            communicate = edge_tts.Communicate(text, self.voice)
            communicate.save_sync(file_name)
        except Exception as e:
            logger.critical(f"\nError: edge-tts unable to generate audio: {e}")
            logger.critical("It's possible that edge-tts is blocked in your region.")
            return None

        return file_name

    def generate_prefix_audio(self, prefix, carrier, file_name_no_ext):
        """
        Speak `prefix` followed by `carrier`, and keep only the prefix. Said at the start
        of a longer sentence, a word like "Hmph," keeps a natural mid-sentence pitch.

        Returns:
        str: path to a wav file with just the prefix, or None if it couldn't be made
        """
        words, audio = [], bytearray()
        try:
            communicate = edge_tts.Communicate(
                f"{prefix} {carrier}", self.voice, boundary="WordBoundary"
            )
            for chunk in communicate.stream_sync():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    words.append(
                        (chunk["offset"] / 10_000_000, chunk["duration"] / 10_000_000)
                    )
        except Exception as e:
            logger.warning(f"edge-tts could not render '{prefix}': {e}")
            return None
        n = len(prefix.split())
        if len(words) < n or not audio:
            logger.warning(f"edge-tts gave no word timings for '{prefix}'")
            return None
        clip = cut_after_words(
            AudioSegment.from_file(io.BytesIO(bytes(audio))), words, n
        )
        path = os.path.join(self.new_audio_dir, f"{file_name_no_ext}.wav")
        clip.export(path, format="wav")
        return path


# en-US-AvaMultilingualNeural
# en-US-EmmaMultilingualNeural
# en-US-JennyNeural
