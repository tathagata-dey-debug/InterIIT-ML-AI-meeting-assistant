"""
Stage 1: Cloud Speech-to-Text transcriber with confidence-based uncertainty tagging.
Connects directly to OpenAI Whisper or Groq Whisper APIs.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, ConfigurationError
from .audio_processor import AudioProcessor
from .key_manager import AllKeysExhaustedError, GeminiKeyPool

logger = logging.getLogger(__name__)

UNCERTAINTY_CONFIDENCE_THRESHOLD = 0.35
LOGPROB_UNCERTAINTY_THRESHOLD = -1.0
STRIP_PUNCTUATION = '.,!?;:"\''

DIARIZATION_SYSTEM_PROMPT = """You are an expert meeting transcriptionist and acoustic-semantic speaker diarization system.
Transcribe the provided audio recording verbatim while accurately separating speakers.

CRITICAL DIARIZATION RULES:
1. MULTI-SPEAKER AWARENESS: This is a collaborative meeting with MULTIPLE speakers. Identify distinct speakers based on vocal timbre, tone, pitch shifts, and conversational turns. Do NOT label the entire transcript under a single speaker tag like [Speaker 1] unless it is truly an uninterrupted solo monologue.
2. CONVERSATIONAL TURN DETECTION: Even if speakers have similar vocal tones, split speakers at natural conversational boundaries:
   - Agreements and reactions (e.g., "Makes total sense.", "Confirmed.", "I agree.") are separate speaker turns.
   - When a speaker directs an action to someone (e.g., "Dev, please configure..."), the responding commitment (e.g., "Understood, I will deliver...") MUST be split into a new speaker turn.
3. SPEAKER IDENTIFICATION & NAMING:
   - Deduce speaker names from conversational context (e.g., if Speaker A says "Priya suggested...", and Speaker B responds "Makes total sense. Also, we will not deploy...", identify Speaker B as [Priya]).
   - If an explicit name cannot be resolved with certainty, use distinct generic labels: [Speaker 1], [Speaker 2], [Speaker 3]. Never collapse different conversational roles into [Speaker 1].
4. TIMESTAMPS: Include approximate timestamps [MM:SS] at the start of each turn.
5. FORMATTING: Output strictly in this format on new lines:
   [Speaker Name] (00:00): Transcribed speech text...
6. VERBATIM ACCURACY: Do not summarize, alter, or omit conversational filler or technical jargon.
7. UNCERTAINTY TAGGING: Tag low-confidence, muffled, or low-volume words as [unclear: word?]."""


class TranscriptionError(Exception):
    """Raised when cloud speech-to-text transcription fails."""
    pass


class CloudTranscriber:
    """Manages cloud STT calls with segment/word logprob evaluation and uncertainty tagging."""

    def __init__(self, config: Config):
        self.config = config
        self.provider = config.stt_provider.lower()

    def transcribe(self, audio_file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Submit audio file to cloud transcription endpoint, request verbose JSON
        with log probabilities, tag low-confidence tokens as [unclear: word?],
        and return (raw_transcript, segments).

        Args:
            audio_file_path: Path to target audio file.

        Returns:
            Tuple of (raw_transcript_string, list_of_segment_dicts).

        Raises:
            TranscriptionError: For authentication, rate limit, network, or decoding errors.
            FileNotFoundError: If the specified audio file cannot be found.
        """
        path = Path(audio_file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Validate file before uploading
        AudioProcessor.validate_file(path, path.name)

        if self.provider == "gemini":
            return self._transcribe_gemini(path)
        elif self.provider == "openai":
            # If user selected OpenAI as STT, provide graceful routing to Gemini for native acoustic diarization if key is present
            if self.config.gemini_api_key and "mock" not in self.config.gemini_api_key:
                try:
                    return self._transcribe_gemini(path)
                except Exception as exc:
                    logger.warning(
                        "Routing OpenAI STT to Gemini diarization failed (%s), falling back to OpenAI Whisper.", exc
                    )
            return self._transcribe_openai(path)
        elif self.provider == "groq":
            if self.config.gemini_api_key and "mock" not in self.config.gemini_api_key:
                try:
                    return self._transcribe_gemini(path)
                except Exception as exc:
                    logger.warning(
                        "Routing Groq STT to Gemini diarization failed (%s), falling back to Groq Whisper.", exc
                    )
            return self._transcribe_groq(path)
        else:
            raise ConfigurationError(
                f"Unsupported STT provider: '{self.provider}'. Supported: 'openai', 'groq', 'gemini'."
            )

    def _transcribe_openai(self, audio_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Execute transcription via OpenAI Whisper API."""
        try:
            from openai import (
                APIConnectionError,
                APIError,
                AuthenticationError,
                OpenAI,
                RateLimitError,
            )
        except ImportError as err:
            raise ImportError(
                "openai package is required for OpenAI STT. Install via requirements.txt"
            ) from err

        if not self.config.openai_api_key:
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )

        client = OpenAI(api_key=self.config.openai_api_key)
        model = self.config.stt_model or "whisper-1"

        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                )

            return self._parse_verbose_response(response)

        except AuthenticationError as exc:
            raise TranscriptionError(
                f"Cloud STT Authentication failed: Invalid OpenAI API key. Check keys/api_keys.json. Details: {exc}"
            ) from exc
        except RateLimitError as exc:
            raise TranscriptionError(
                f"Cloud STT rate limit exceeded on OpenAI. Please wait or check your account quota: {exc}"
            ) from exc
        except APIConnectionError as exc:
            raise TranscriptionError(
                f"Network connection failed or timed out while reaching OpenAI STT API: {exc}"
            ) from exc
        except APIError as exc:
            raise TranscriptionError(
                f"OpenAI Cloud STT API returned an error: {exc}"
            ) from exc
        except Exception as exc:
            raise TranscriptionError(
                f"Unexpected error during OpenAI transcription: {exc}"
            ) from exc

    def _transcribe_groq(self, audio_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Execute transcription via Groq Whisper API."""
        try:
            from groq import (
                APIConnectionError,
                APIError,
                AuthenticationError,
                Groq,
                RateLimitError,
            )
        except ImportError as err:
            raise ImportError(
                "groq package is required for Groq STT. Install via requirements.txt"
            ) from err

        if not self.config.groq_api_key:
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )

        client = Groq(api_key=self.config.groq_api_key)
        model = self.config.stt_model if "whisper" in self.config.stt_model else "whisper-large-v3"

        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                )

            return self._parse_verbose_response(response)

        except AuthenticationError as exc:
            raise TranscriptionError(
                f"Cloud STT Authentication failed: Invalid Groq API key. Check keys/api_keys.json. Details: {exc}"
            ) from exc
        except RateLimitError as exc:
            raise TranscriptionError(
                f"Cloud STT rate limit exceeded on Groq. Please wait or check your quota: {exc}"
            ) from exc
        except APIConnectionError as exc:
            raise TranscriptionError(
                f"Network connection failed or timed out while reaching Groq STT API: {exc}"
            ) from exc
        except APIError as exc:
            raise TranscriptionError(
                f"Groq Cloud STT API returned an error: {exc}"
            ) from exc
        except Exception as exc:
            raise TranscriptionError(
                f"Unexpected error during Groq transcription: {exc}"
            ) from exc

    def _transcribe_gemini(self, audio_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Execute transcription and acoustic speaker diarization via Google Gemini Multimodal Audio API."""
        try:
            from google import genai
            from google.genai.errors import APIError as GenAIErr
        except ImportError as err:
            raise ImportError(
                "google-genai is required for Gemini STT. Install via requirements.txt"
            ) from err

        key_pool = self.config.key_pool
        if key_pool is None:
            pool_keys = list(self.config.gemini_api_keys)
            if self.config.gemini_api_key and self.config.gemini_api_key not in pool_keys:
                pool_keys.insert(0, self.config.gemini_api_key)
            if not pool_keys:
                raise ConfigurationError(
                    "Missing API credentials. Please populate keys/api_keys.json with your API keys."
                )
            key_pool = GeminiKeyPool(pool_keys)

        def _do_transcribe(api_key: str) -> Tuple[str, List[Dict[str, Any]]]:
            client = genai.Client(api_key=api_key)
            uploaded = client.files.upload(file=str(audio_path))
            prompt = DIARIZATION_SYSTEM_PROMPT

            # Prioritize gemini-1.5-flash with graceful fallback to active models
            candidate_models = ["gemini-1.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]
            configured_model = self.config.stt_model if "gemini" in (self.config.stt_model or "") else None
            if configured_model and configured_model not in candidate_models:
                candidate_models.insert(0, configured_model)

            response = None
            last_err = None
            for model_name in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[uploaded, prompt],
                    )
                    if response and response.text:
                        break
                except Exception as exc:
                    last_err = exc
                    if "404" in str(exc) or "NOT_FOUND" in str(exc) or "no longer available" in str(exc):
                        logger.warning(
                            "Gemini model '%s' unavailable (%s), trying next candidate.", model_name, exc
                        )
                        continue
                    raise

            if response is None:
                if last_err:
                    raise last_err
                raise TranscriptionError("Gemini API returned an empty transcription response.")

            raw_text = response.text.strip() if response.text else ""

            # Parse speaker-diarized lines into structured segments
            segments: List[Dict[str, Any]] = []
            turn_pattern = re.compile(r"^\[([^\]]+)\](?:\s*\(([0-9]{1,2}:[0-9]{2})\))?:?\s*(.*)$")
            for idx, line in enumerate(raw_text.splitlines()):
                line_str = line.strip()
                if not line_str:
                    continue
                match = turn_pattern.match(line_str)
                if match:
                    spk, ts_str, text_content = match.groups()
                    start_sec = 0.0
                    if ts_str:
                        try:
                            parts = ts_str.split(":")
                            start_sec = float(parts[0]) * 60 + float(parts[1])
                        except Exception:
                            start_sec = 0.0
                    segments.append({
                        "id": idx,
                        "speaker": spk.strip(),
                        "start": start_sec,
                        "end": start_sec,
                        "text": text_content.strip() if text_content else line_str,
                        "avg_logprob": 0.0,
                    })
                else:
                    segments.append({
                        "id": idx,
                        "speaker": None,
                        "start": 0.0,
                        "end": 0.0,
                        "text": line_str,
                        "avg_logprob": 0.0,
                    })

            if not segments:
                segments = [{"id": 0, "start": 0.0, "end": 0.0, "text": raw_text, "avg_logprob": 0.0}]

            return raw_text, segments

        try:
            return key_pool.execute_with_fallback(_do_transcribe)
        except AllKeysExhaustedError as err:
            raise TranscriptionError(f"Cloud STT Quota Exceeded across key pool: {err}") from err
        except GenAIErr as exc:
            raise TranscriptionError(f"Gemini API returned an error during transcription: {exc}") from exc
        except Exception as exc:
            raise TranscriptionError(f"Unexpected error during Gemini transcription: {exc}") from exc

    def _parse_verbose_response(self, response: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse verbose JSON response from Whisper-compatible STT engines.
        Tags low-confidence tokens (confidence < 0.35 or avg_logprob < -1.0) with [unclear: word?].
        """
        # Convert response to dictionary representation
        if hasattr(response, "model_dump"):
            resp_dict = response.model_dump()
        elif hasattr(response, "to_dict"):
            resp_dict = response.to_dict()
        elif isinstance(response, dict):
            resp_dict = response
        else:
            resp_dict = getattr(response, "__dict__", {})

        raw_segments = resp_dict.get("segments", []) or []
        raw_words = resp_dict.get("words", []) or []

        parsed_segments: List[Dict[str, Any]] = []
        tagged_transcript_tokens: List[str] = []

        # If word-level tokens exist
        if raw_words:
            for w in raw_words:
                w_dict = w if isinstance(w, dict) else getattr(w, "__dict__", {})
                word_text = str(w_dict.get("word", "")).strip()
                if not word_text:
                    continue

                # Check confidence or logprob
                conf = w_dict.get("confidence")
                logprob = w_dict.get("logprob")

                is_low_confidence = False
                if conf is not None and float(conf) < UNCERTAINTY_CONFIDENCE_THRESHOLD:
                    is_low_confidence = True
                elif logprob is not None and float(logprob) < LOGPROB_UNCERTAINTY_THRESHOLD:
                    is_low_confidence = True

                if is_low_confidence:
                    clean_token = word_text.strip(".,!?;:\"'")
                    tagged_transcript_tokens.append(f"[unclear: {clean_token}?]")
                else:
                    tagged_transcript_tokens.append(word_text)

            raw_transcript = " ".join(tagged_transcript_tokens).strip()

            # Normalize segments
            for s in raw_segments:
                s_dict = s if isinstance(s, dict) else getattr(s, "__dict__", {})
                parsed_segments.append(s_dict)

            return raw_transcript, parsed_segments

        # If word-level tokens are not explicitly separate, evaluate segment log probabilities
        for s in raw_segments:
            s_dict = s if isinstance(s, dict) else getattr(s, "__dict__", {})
            seg_text = str(s_dict.get("text", "")).strip()
            avg_logprob = s_dict.get("avg_logprob")
            no_speech_prob = s_dict.get("no_speech_prob", 0.0)

            # If segment logprob indicates low confidence (avg_logprob < -1.0 or confidence < 0.35)
            # and no_speech_prob is not overriding
            is_low_confidence = False
            if avg_logprob is not None:
                try:
                    lp = float(avg_logprob)
                    if lp < LOGPROB_UNCERTAINTY_THRESHOLD:
                        is_low_confidence = True
                    # Also check equivalent probability: p = e^(logprob)
                    elif math.exp(lp) < UNCERTAINTY_CONFIDENCE_THRESHOLD:
                        is_low_confidence = True
                except (ValueError, OverflowError):
                    pass

            tokens = seg_text.split()
            if is_low_confidence and tokens:
                tagged_words = [f"[unclear: {t.strip(STRIP_PUNCTUATION)}?]" for t in tokens]
                tagged_segment = " ".join(tagged_words)
                tagged_transcript_tokens.append(tagged_segment)
            else:
                tagged_transcript_tokens.append(seg_text)

            parsed_segments.append(s_dict)

        raw_transcript = " ".join(tagged_transcript_tokens).strip()
        if not raw_transcript:
            raw_transcript = str(resp_dict.get("text", "")).strip()

        return raw_transcript, parsed_segments
