import requests
from pathlib import Path
from pydub import AudioSegment
import subprocess
import re
from src.utils import get_temp_path

class AudioPreprocessor:
    @staticmethod
    def validate_and_fix_file(file_path: str) -> str:
        """
        Validates and preprocesses audio files for optimal transcription.
        Supports direct URLs (downloads to /content/input_downloaded.*).
        For MP4 files, converts to WAV for Whisper.
        """
        print(f"Validating file or URL: {file_path}")

        # If URL, download
        if isinstance(file_path, str) and re.match(r"^https?://", file_path.strip(), re.IGNORECASE):
            try:
                print("Detected URL — downloading...")
                resp = requests.get(file_path, stream=True, timeout=60)
                resp.raise_for_status()
                suffix = ".mp4" if ".mp4" in file_path.lower() else (".mp3" if ".mp3" in file_path.lower() else ".bin")
                dl_path = get_temp_path("input_downloaded" + suffix)
                with open(dl_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded to: {dl_path}")
                file_path = dl_path
            except Exception as e:
                raise RuntimeError(f"Failed to download media: {e}")

        # Local/Downloaded path must exist now
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.lower().endswith('.mp4'):
                print(f"Converting MP4 → Whisper-optimized WAV...")
                return AudioPreprocessor._convert_to_whisper_wav(file_path)
            elif file_path.lower().endswith(('.mp3', '.m4a', '.aac', '.ogg', '.wav')):
                print(f"Normalizing to Whisper-optimized WAV...")
                return AudioPreprocessor._convert_to_whisper_wav(file_path)
            else:
                print("Unknown format — attempting Python fallback decode...")
                return AudioPreprocessor._python_extract_audio(file_path)
        except Exception as e:
            print(f"Error during file processing: {str(e)}")
            raise

    @staticmethod
    def _convert_to_whisper_wav(audio_path: str) -> str:
        """Convert any audio file to WAV format optimized for Whisper model"""
        wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        try:
            result = subprocess.run([
                'ffmpeg','-y','-i', audio_path,
                '-acodec','pcm_s16le','-ar','16000','-ac','1', wav_path
            ], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(result.stderr or "ffmpeg failed")
            print(f"Created: {wav_path}")
            return wav_path
        except Exception as e:
            raise RuntimeError(f"Failed to convert {audio_path} → WAV: {e}")

    @staticmethod
    def _python_extract_audio(file_path: str) -> str:
        """
        Fallback: use PyDub to decode & write 16kHz mono WAV.
        """
        print("Attempting Python-based audio extraction...")
        wav_path = file_path.rsplit('.', 1)[0] + '_extracted.wav'
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")
        if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
            raise RuntimeError("Python audio extraction produced empty file")
        print(f"Created: {wav_path}")
        return wav_path
