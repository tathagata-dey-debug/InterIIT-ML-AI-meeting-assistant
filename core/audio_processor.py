"""
Audio validation, loudness normalization, and compression module.
Prepares audio streams for optimal cloud STT consumption without PyTorch or CUDA.
"""

from __future__ import annotations

import io
import math
import os
import shutil
from pathlib import Path
from typing import Any, List, Tuple, Union

# Auto-configure ffmpeg from imageio_ffmpeg before importing pydub to prevent RuntimeWarning
_ffmpeg_exe: str | None = None
try:
    import imageio_ffmpeg

    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if _ffmpeg_exe and os.path.exists(_ffmpeg_exe):
        _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
        _target_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        _target_exe = os.path.join(_ffmpeg_dir, _target_name)
        if not os.path.exists(_target_exe):
            try:
                shutil.copy2(_ffmpeg_exe, _target_exe)
            except Exception:
                pass
        if _ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", message=".*Couldn't find ffprobe or avprobe.*")

from pydub import AudioSegment
import pydub.utils
import pydub.audio_segment

if _ffmpeg_exe and os.path.exists(_ffmpeg_exe):
    AudioSegment.converter = _ffmpeg_exe
elif shutil.which("ffmpeg"):
    AudioSegment.converter = shutil.which("ffmpeg")

# Safely handle systems without ffprobe: fall back cleanly to ffmpeg decoding
_orig_mediainfo_json = getattr(pydub.utils, "mediainfo_json", None)

def _safe_mediainfo_json(*args: Any, **kwargs: Any) -> Any:
    if _orig_mediainfo_json is not None:
        try:
            return _orig_mediainfo_json(*args, **kwargs)
        except (FileNotFoundError, OSError, Exception):
            return None
    return None

pydub.utils.mediainfo_json = _safe_mediainfo_json
pydub.audio_segment.mediainfo_json = _safe_mediainfo_json


# ==========================================
# Custom Audio Processing Exceptions
# ==========================================

class AudioProcessingError(Exception):
    """Base exception for audio processing failures."""
    pass


class EmptyAudioError(AudioProcessingError):
    """Raised when the uploaded or provided audio file is 0 bytes."""
    pass


class UnsupportedAudioFormatError(AudioProcessingError):
    """Raised when an audio format is not supported."""
    pass


class CorruptAudioError(AudioProcessingError):
    """Raised when an audio file cannot be decoded or contains no audible speech data."""
    pass


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}
AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}
MAX_PAYLOAD_SIZE_BYTES = 24 * 1024 * 1024  # 24 MB payload ceiling for cloud STT endpoints


def get_audio_mime_type(filename_or_ext: str) -> str:
    """Return the MIME type for a supported audio file or extension."""
    if not filename_or_ext:
        return "audio/mpeg"
    ext = Path(filename_or_ext).suffix.lower() if "." in filename_or_ext else f".{filename_or_ext.lower().lstrip('.')}"
    return AUDIO_MIME_TYPES.get(ext, "audio/mpeg")


class AudioProcessor:
    """Production audio ingestion, validation, normalization, and compression utility."""

    @staticmethod
    def validate_file(file_path_or_bytes: Union[str, Path, bytes, bytearray, memoryview, io.BytesIO, Any], filename: str) -> None:
        """
        Validate audio file size and format extension.

        Args:
            file_path_or_bytes: Path to file, raw bytes, or BytesIO buffer.
            filename: Name of the audio file to inspect extension.

        Raises:
            EmptyAudioError: If file size is 0 bytes.
            UnsupportedAudioFormatError: If file extension is not in supported formats.
        """
        # 1. Validate file extension (case-insensitive whitelist)
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise UnsupportedAudioFormatError(
                f"Unsupported audio format: '{suffix}'. Allowed formats: WAV, MP3, M4A, OGG."
            )

        # 2. Validate file size
        size_bytes = 0
        if isinstance(file_path_or_bytes, (str, Path)):
            p = Path(file_path_or_bytes)
            if not p.exists():
                raise AudioProcessingError(f"Audio file does not exist: {p}")
            size_bytes = p.stat().st_size
        elif isinstance(file_path_or_bytes, (bytes, bytearray, memoryview)):
            size_bytes = len(file_path_or_bytes)
        elif hasattr(file_path_or_bytes, "nbytes"):
            size_bytes = file_path_or_bytes.nbytes
        elif hasattr(file_path_or_bytes, "size") and isinstance(file_path_or_bytes.size, int):
            size_bytes = file_path_or_bytes.size
        elif hasattr(file_path_or_bytes, "getbuffer"):
            size_bytes = len(file_path_or_bytes.getbuffer())
        elif hasattr(file_path_or_bytes, "getvalue"):
            size_bytes = len(file_path_or_bytes.getvalue())
        elif hasattr(file_path_or_bytes, "seek") and hasattr(file_path_or_bytes, "tell"):
            pos = file_path_or_bytes.tell()
            file_path_or_bytes.seek(0, io.SEEK_END)
            size_bytes = file_path_or_bytes.tell()
            file_path_or_bytes.seek(pos)
        elif hasattr(file_path_or_bytes, "__len__"):
            size_bytes = len(file_path_or_bytes)
        else:
            raise AudioProcessingError("Unrecognized audio payload type.")

        if size_bytes == 0:
            raise EmptyAudioError("The uploaded file is empty (0 bytes).")

    @classmethod
    def preprocess_audio(
        cls,
        input_file_path: str,
        output_path: str = "temp_processed.mp3",
    ) -> str:
        """
        Inspect audio, verify audible signal, normalize loudness to -20 dBFS,
        resample to 16,000 Hz mono PCM, and export a clean, standardized temporary file.

        Args:
            input_file_path: Path to source audio file.
            output_path: Path for output processed file.

        Returns:
            Resolved string path of the clean, processed audio file.

        Raises:
            CorruptAudioError: If audio is unreadable or contains no audible data.
            EmptyAudioError: If input file is empty.
            UnsupportedAudioFormatError: If format extension is unsupported.
        """
        in_path = Path(input_file_path).resolve()
        cls.validate_file(in_path, in_path.name)

        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        audio: AudioSegment | None = None
        ext = in_path.suffix.lower()

        try:
            audio = AudioSegment.from_file(str(in_path))
        except Exception as first_exc:
            # Resilient fallback for m4a and ogg if auto-detection fails
            if ext == ".m4a":
                for fmt in ("m4a", "mp4"):
                    try:
                        audio = AudioSegment.from_file(str(in_path), format=fmt)
                        break
                    except Exception:
                        pass
            elif ext == ".ogg":
                try:
                    audio = AudioSegment.from_file(str(in_path), format="ogg")
                except Exception:
                    pass

            if audio is None:
                raise CorruptAudioError(f"Failed to decode audio file '{in_path.name}': {first_exc}") from first_exc

        # Validate audio data: if audio.dBFS == -float('inf') or duration is less than 500 ms
        if math.isinf(audio.dBFS) or audio.dBFS == -float("inf") or len(audio) < 500:
            raise CorruptAudioError("Audio file contains no audible speech data.")

        # Loudness normalization to -20 dBFS
        target_dbfs = -20.0
        change_in_dbfs = target_dbfs - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dbfs)

        # Resample to 16,000 Hz mono PCM
        standardized_audio = (
            normalized_audio.set_channels(1)
            .set_frame_rate(16000)
        )

        out_fmt = out_path.suffix.lstrip(".").lower() or "mp3"
        if out_fmt == "wav":
            standardized_audio.export(str(out_path), format="wav")
        else:
            standardized_audio.export(str(out_path), format="mp3", bitrate="64k")

        return str(out_path)


# ==========================================
# Backwards-Compatibility Utility Wrappers
# ==========================================

def validate_audio_file(file_path: Path | str) -> Path:
    """Validate audio file path existence and supported format."""
    path = Path(file_path).resolve()
    AudioProcessor.validate_file(path, path.name)
    return path


def normalize_audio_loudness(
    input_path: Path | str,
    output_path: Path | str,
    target_dbfs: float = -20.0,
) -> Tuple[Path, float]:
    """Normalize loudness wrapper returning (output_path, duration_seconds)."""
    in_p = Path(input_path).resolve()
    AudioProcessor.validate_file(in_p, in_p.name)
    out_p = Path(output_path).resolve()

    processed_str = AudioProcessor.preprocess_audio(str(in_p), str(out_p))
    audio = AudioSegment.from_file(processed_str)
    return Path(processed_str), len(audio) / 1000.0


def split_audio_into_chunks(
    audio_path: Path | str,
    chunk_length_seconds: int = 600,
    output_dir: Path | str = "scratch/chunks",
) -> List[Path]:
    """Split audio into chunk segments."""
    valid_path = validate_audio_file(audio_path)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(str(valid_path))
    chunk_ms = chunk_length_seconds * 1000
    total_chunks = (len(audio) + chunk_ms - 1) // chunk_ms

    if total_chunks <= 1:
        return [valid_path]

    chunk_paths: List[Path] = []
    for idx in range(total_chunks):
        start_ms = idx * chunk_ms
        end_ms = min((idx + 1) * chunk_ms, len(audio))
        chunk = audio[start_ms:end_ms]

        chunk_filename = out_dir / f"{valid_path.stem}_part_{idx + 1:03d}.mp3"
        chunk.export(str(chunk_filename), format="mp3", bitrate="128k")
        chunk_paths.append(chunk_filename)

    return chunk_paths
