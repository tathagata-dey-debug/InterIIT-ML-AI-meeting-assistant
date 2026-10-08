"""
Unit and integration test suite for AI Meeting Assistant pipeline.
Validates typed configuration, edge case credentials handling, schema validation, and audio checking.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from core.config import Config, ConfigurationError
from core.key_manager import (
    AllKeysExhaustedError,
    GeminiKeyPool,
    is_exhaustion_error,
    mask_api_key,
)
from core.schemas import (
    ActionItem,
    KeyDecision,
    Decision,
    MeetingMinutes,
    MeetingRecord,
    TranscriptWord,
)
from core.audio_processor import (
    AudioProcessingError,
    AudioProcessor,
    EmptyAudioError,
    UnsupportedAudioFormatError,
    CorruptAudioError,
    SUPPORTED_AUDIO_EXTENSIONS,
    AUDIO_MIME_TYPES,
    get_audio_mime_type,
    validate_audio_file,
)
from core.transcriber import CloudTranscriber, TranscriptionError, DIARIZATION_SYSTEM_PROMPT
from core.refiner import REFINER_SYSTEM_PROMPT
from core.extractor import DOCUMENTER_SYSTEM_PROMPT
from core.exporter import Exporter, export_to_json, export_to_markdown


class TestConfiguration:
    """Test suite for credential loading, fallback, and validation."""

    def test_missing_credentials_raises_configuration_error(self, tmp_path: Path):
        """When keys file does not exist and no env vars are present, raise ConfigurationError."""
        fake_keys_path = tmp_path / "non_existent_keys.json"

        with pytest.raises(ConfigurationError) as exc_info:
            Config.load(keys_path=fake_keys_path, stt_provider="openai", openai_api_key=None)

        assert "Missing API credentials. Please populate keys/api_keys.json with your API keys." in str(exc_info.value)

    def test_placeholder_credentials_rejected(self, tmp_path: Path):
        """Placeholder strings like 'sk-your-openai-key-here' must be rejected."""
        keys_file = tmp_path / "api_keys.json"
        keys_file.write_text(
            json.dumps({
                "STT_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-your-openai-key-here",
                "LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "AIza-your-gemini-key-here"
            }),
            encoding="utf-8"
        )

        with pytest.raises(ConfigurationError) as exc_info:
            Config.load(keys_path=keys_file)

        assert "Missing API credentials. Please populate keys/api_keys.json with your API keys." in str(exc_info.value)

    def test_valid_explicit_credentials_loaded(self):
        """Valid explicit keys should load successfully without error."""
        cfg = Config(
            stt_provider="openai",
            openai_api_key="sk-proj-valid-mock-key-1234567890",
            llm_provider="openai",
        )
        assert cfg.stt_provider == "openai"
        assert cfg.openai_api_key == "sk-proj-valid-mock-key-1234567890"
        assert cfg.stt_model == "whisper-1"

    def test_gemini_provider_requires_gemini_key(self):
        """Selecting Gemini as LLM requires gemini_api_key."""
        with pytest.raises(ConfigurationError):
            Config(
                stt_provider="openai",
                openai_api_key="sk-proj-valid-mock-key-1234567890",
                llm_provider="gemini",
                gemini_api_key=None,
            )


class TestSchemas:
    """Test suite verifying Pydantic v2 schemas and anti-hallucination constraints."""

    def test_transcript_word_confidence_bounds(self):
        """Confidence score must strictly be between 0.0 and 1.0."""
        valid_word = TranscriptWord(
            word="architecture",
            start=1.2,
            end=1.9,
            confidence=0.88,
            is_uncertain=False,
        )
        assert valid_word.confidence == 0.88

        with pytest.raises(ValidationError):
            TranscriptWord(
                word="invalid",
                start=0.0,
                end=1.0,
                confidence=1.5,  # Out of range > 1.0
            )

    def test_action_item_schema_defaults(self):
        """Action items must default owner and deadline to 'Unspecified' when missing."""
        item = ActionItem(
            task_description="Deploy Kubernetes ingress controller",
        )
        assert item.task_description == "Deploy Kubernetes ingress controller"
        assert item.owner == "Unspecified"
        assert item.deadline == "Unspecified"

    def test_action_item_schema_explicit(self):
        """Action items correctly accept explicit owner and deadline."""
        item = ActionItem(
            task_description="Write end-to-end integration tests",
            owner="Alice",
            deadline="2026-10-15",
        )
        assert item.owner == "Alice"
        assert item.deadline == "2026-10-15"

    def test_key_decision_schema(self):
        """KeyDecision requires decision_text and supporting_quote."""
        decision = KeyDecision(
            decision_text="Migrate pipeline to cloud APIs only",
            supporting_quote="Team unanimously agreed to eliminate CUDA dependencies.",
        )
        assert decision.decision_text == "Migrate pipeline to cloud APIs only"
        assert "unanimously agreed" in decision.supporting_quote

    def test_meeting_record_and_minutes_serialization(self):
        """MeetingMinutes and MeetingRecord validate and export to JSON & Markdown properly."""
        minutes = MeetingMinutes(
            concise_summary="Sprint planning and architectural refactoring session.",
            discussion_points=[
                "Migrated audio processing to CPU-only pydub normalization",
                "Integrated cloud-based speech-to-text with uncertainty tags",
            ],
            key_decisions=[
                KeyDecision(
                    decision_text="Migrate pipeline to cloud APIs only",
                    supporting_quote="Let's drop local weights and use cloud endpoints entirely.",
                )
            ],
            action_items=[
                ActionItem(
                    task_description="Draft API specification",
                    owner="Alice",
                    deadline="Friday",
                )
            ],
        )

        record = MeetingRecord(
            raw_transcript="We will deploy [unclear: Kubernetes?] tomorrow.",
            refined_transcript="We will deploy Kubeflow tomorrow.",
            minutes=minutes,
        )

        assert record.raw_transcript.startswith("We will deploy")
        assert len(record.minutes.key_decisions) == 1
        assert len(record.minutes.action_items) == 1

        json_out = export_to_json(record)
        assert "Sprint planning and architectural refactoring session." in json_out
        assert "Draft API specification" in json_out

        md_out = export_to_markdown(record)
        assert "# Meeting Record" in md_out
        assert "## Executive Summary" in md_out
        assert "## Key Decisions" in md_out
        assert "Migrate pipeline to cloud APIs only" in md_out
        assert "## Action Items" in md_out
        assert "| Draft API specification | Alice | Friday |" in md_out
        assert "## Refined Transcript" in md_out
        assert "We will deploy Kubeflow tomorrow." in md_out
        assert "## Raw Transcript" in md_out
        assert "We will deploy [unclear: Kubernetes?] tomorrow." in md_out

    def test_exporter_empty_decisions_and_actions(self):
        """When decisions and action items are empty, markdown prints exact fallback text."""
        empty_minutes = MeetingMinutes(
            concise_summary="Brief status sync with no deliverables.",
            discussion_points=["Reviewed roadmap"],
            key_decisions=[],
            action_items=[],
        )
        empty_record = MeetingRecord(
            raw_transcript="Everything looks good.",
            refined_transcript="Everything looks good.",
            minutes=empty_minutes,
        )

        md = export_to_markdown(empty_record)
        assert "No explicit decisions were finalized." in md
        assert "No actionable tasks were assigned." in md


class TestAudioProcessor:
    """Test suite for audio format validation, custom exceptions, and preprocessing."""

    def test_validate_missing_file_raises_error(self, tmp_path: Path):
        """Missing audio file must raise AudioProcessingError."""
        non_existent = tmp_path / "ghost_meeting.mp3"
        with pytest.raises(AudioProcessingError) as exc_info:
            AudioProcessor.validate_file(non_existent, non_existent.name)
        assert "Audio file does not exist" in str(exc_info.value)

    def test_validate_empty_audio_raises_empty_audio_error(self, tmp_path: Path):
        """0-byte audio file must raise EmptyAudioError with exact required message."""
        empty_file = tmp_path / "empty_meeting.mp3"
        empty_file.write_bytes(b"")

        with pytest.raises(EmptyAudioError) as exc_info:
            AudioProcessor.validate_file(empty_file, empty_file.name)
        assert "The uploaded file is empty (0 bytes)." in str(exc_info.value)

    def test_validate_empty_files_rejected_across_all_four_formats(self, tmp_path: Path):
        """Rejection of empty 0-byte files across WAV, MP3, M4A, and OGG formats."""
        for ext in [".wav", ".mp3", ".m4a", ".ogg"]:
            empty_file = tmp_path / f"empty_test{ext}"
            empty_file.write_bytes(b"")
            with pytest.raises(EmptyAudioError) as exc_info:
                AudioProcessor.validate_file(empty_file, empty_file.name)
            assert "The uploaded file is empty (0 bytes)." in str(exc_info.value)

    def test_validate_all_four_supported_formats_including_case_insensitivity(self, tmp_path: Path):
        """All 4 supported formats (WAV, MP3, M4A, OGG) pass validation in lower and upper case."""
        for ext in [".wav", ".mp3", ".m4a", ".ogg", ".WAV", ".MP3", ".M4A", ".OGG"]:
            audio_f = tmp_path / f"valid_sample{ext}"
            audio_f.write_bytes(b"non-empty valid audio payload bytes")
            # Must not raise
            AudioProcessor.validate_file(audio_f, audio_f.name)

    def test_validate_unsupported_extensions_rejected(self, tmp_path: Path):
        """Unsupported file extensions (.txt, .flac, .aac, .pdf) must raise UnsupportedAudioFormatError."""
        for ext in [".txt", ".flac", ".aac", ".pdf", ".mp4", ".webm"]:
            bad_file = tmp_path / f"test{ext}"
            bad_file.write_bytes(b"non-empty payload")
            with pytest.raises(UnsupportedAudioFormatError) as exc_info:
                AudioProcessor.validate_file(bad_file, bad_file.name)
            assert f"Unsupported audio format: '{ext}'. Allowed formats: WAV, MP3, M4A, OGG." in str(exc_info.value)

    def test_mime_type_mapping(self):
        """Verify dynamic MIME type mapping for all four supported formats."""
        assert get_audio_mime_type("recording.wav") == "audio/wav"
        assert get_audio_mime_type("recording.WAV") == "audio/wav"
        assert get_audio_mime_type("recording.mp3") == "audio/mpeg"
        assert get_audio_mime_type("recording.MP3") == "audio/mpeg"
        assert get_audio_mime_type("recording.m4a") == "audio/mp4"
        assert get_audio_mime_type("recording.M4A") == "audio/mp4"
        assert get_audio_mime_type("recording.ogg") == "audio/ogg"
        assert get_audio_mime_type("recording.OGG") == "audio/ogg"

    def test_validate_bytes_buffer(self):
        """Bytes, bytearray, memoryview, and BytesIO payloads are properly validated by size and filename."""
        import io

        with pytest.raises(EmptyAudioError):
            AudioProcessor.validate_file(b"", "session.wav")

        with pytest.raises(EmptyAudioError):
            AudioProcessor.validate_file(memoryview(b""), "session.wav")

        # Non-empty valid bytes, bytearrays, memoryviews, and BytesIO
        AudioProcessor.validate_file(b"\x00" * 1024, "session.mp3")
        AudioProcessor.validate_file(bytearray(b"\x00" * 512), "session.wav")
        AudioProcessor.validate_file(memoryview(b"\x00" * 256), "session.wav")
        AudioProcessor.validate_file(io.BytesIO(b"\x00" * 128), "session.ogg")

    def test_corrupt_audio_silent_or_too_short_raises_corrupt_audio_error(self, tmp_path: Path, monkeypatch):
        """Audio files that are silent or shorter than 500ms raise CorruptAudioError."""
        from pydub import AudioSegment

        test_file = tmp_path / "silent.wav"
        test_file.write_bytes(b"\x00" * 1024)

        # Mock silent audio segment
        silent_audio = AudioSegment.silent(duration=1000)
        monkeypatch.setattr(AudioSegment, "from_file", lambda *args, **kwargs: silent_audio)

        with pytest.raises(CorruptAudioError) as exc_info:
            AudioProcessor.preprocess_audio(str(test_file))
        assert "Audio file contains no audible speech data." in str(exc_info.value)

        # Mock short audio segment (< 500ms)
        from pydub.generators import Sine
        short_audio = Sine(440).to_audio_segment(duration=300)
        monkeypatch.setattr(AudioSegment, "from_file", lambda *args, **kwargs: short_audio)

        with pytest.raises(CorruptAudioError) as exc_info:
            AudioProcessor.preprocess_audio(str(test_file))
        assert "Audio file contains no audible speech data." in str(exc_info.value)

    def test_preprocess_resilient_m4a_and_ogg_fallback(self, tmp_path: Path, monkeypatch):
        """Preprocess audio falls back to explicit format flags if auto-detection fails."""
        from pydub import AudioSegment
        from pydub.generators import Sine

        valid_tone = Sine(440).to_audio_segment(duration=1000)
        out_target = tmp_path / "out.mp3"

        # Test M4A fallback
        m4a_file = tmp_path / "meeting.m4a"
        m4a_file.write_bytes(b"\x00" * 512)

        calls = []
        def mock_from_file(path, format=None):
            calls.append(format)
            if format is None:
                raise Exception("Cannot auto-detect container")
            if format in ("m4a", "mp4"):
                return valid_tone
            raise Exception("Unsupported")

        monkeypatch.setattr(AudioSegment, "from_file", mock_from_file)
        result_path = AudioProcessor.preprocess_audio(str(m4a_file), str(out_target))
        assert Path(result_path).exists()
        assert None in calls
        assert "m4a" in calls


class TestTranscriberUncertainty:
    """Test suite for CloudTranscriber uncertainty tagging and logprob evaluation."""

    def test_parse_verbose_response_tags_low_confidence_words(self):
        """Words with confidence < 0.35 must be tagged with [unclear: word?]."""
        cfg = Config(
            stt_provider="openai",
            openai_api_key="sk-mock-valid-key-1234567890",
            llm_provider="openai",
        )
        transcriber = CloudTranscriber(cfg)

        mock_response = {
            "text": "We will deploy Kubernetes tomorrow",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": "We will deploy Kubernetes tomorrow",
                    "avg_logprob": -0.2,
                }
            ],
            "words": [
                {"word": "We", "start": 0.0, "end": 0.2, "confidence": 0.98},
                {"word": "will", "start": 0.2, "end": 0.4, "confidence": 0.95},
                {"word": "deploy", "start": 0.4, "end": 0.8, "confidence": 0.90},
                {"word": "Kubernetes", "start": 0.8, "end": 1.5, "confidence": 0.22},  # Low confidence < 0.35
                {"word": "tomorrow", "start": 1.5, "end": 2.0, "confidence": 0.96},
            ]
        }

        raw_transcript, segments = transcriber._parse_verbose_response(mock_response)

        assert "[unclear: Kubernetes?]" in raw_transcript
        assert "We will deploy [unclear: Kubernetes?] tomorrow" == raw_transcript
        assert len(segments) == 1

    def test_parse_verbose_response_tags_low_logprob_segments(self):
        """Segments with avg_logprob < -1.0 must tag tokens as unclear."""
        cfg = Config(
            stt_provider="openai",
            openai_api_key="sk-mock-valid-key-1234567890",
            llm_provider="openai",
        )
        transcriber = CloudTranscriber(cfg)

        mock_response = {
            "text": "Muffled speech here",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Muffled speech here",
                    "avg_logprob": -1.45,  # Below -1.0 threshold
                }
            ],
            "words": []
        }

        raw_transcript, segments = transcriber._parse_verbose_response(mock_response)
        assert "[unclear: Muffled?]" in raw_transcript
        assert "[unclear: speech?]" in raw_transcript
        assert "[unclear: here?]" in raw_transcript


class TestBenchmarkRunnerCLI:
    """Test suite for headless evaluation CLI runner and artifact persistence."""

    def test_benchmark_artifact_persistence_and_parity(self, tmp_path: Path):
        """Benchmark exports must generate matching JSON and Markdown artifacts."""
        minutes = MeetingMinutes(
            concise_summary="Quarterly Engineering Planning meeting.",
            discussion_points=[
                "Discussed zero-GPU cloud API strategy",
                "Agreed to enforce evidence quotes for decisions",
            ],
            key_decisions=[
                KeyDecision(
                    decision_text="Standardize on Pydantic v2 schemas",
                    supporting_quote="The team unanimously resolved to adopt Pydantic v2.",
                )
            ],
            action_items=[
                ActionItem(
                    task_description="Deploy headless benchmark runner",
                    owner="DevOps",
                    deadline="Next Monday",
                ),
                ActionItem(
                    task_description="Configure API credentials template",
                    owner="Unspecified",
                    deadline="Unspecified",
                )
            ],
        )

        record = MeetingRecord(
            raw_transcript="We will deploy [unclear: Kubernetes?] next Monday.",
            refined_transcript="We will deploy Kubeflow next Monday.",
            minutes=minutes,
        )

        out_dir = tmp_path / "benchmark_outputs"
        json_path, md_path = Exporter.save_artifacts(record, out_dir, base_filename="benchmark_test")

        assert json_path.exists()
        assert md_path.exists()

        # Check content parity
        json_data = json.loads(json_path.read_text(encoding="utf-8"))
        md_text = md_path.read_text(encoding="utf-8")

        assert json_data["minutes"]["concise_summary"] == "Quarterly Engineering Planning meeting."
        assert "Quarterly Engineering Planning meeting." in md_text
        assert "Standardize on Pydantic v2 schemas" in md_text
        assert "The team unanimously resolved to adopt Pydantic v2." in md_text
        assert "| Deploy headless benchmark runner | DevOps | Next Monday |" in md_text
        assert "| Configure API credentials template | Unspecified | Unspecified |" in md_text
        assert "We will deploy Kubeflow next Monday." in md_text
        assert "We will deploy [unclear: Kubernetes?] next Monday." in md_text


class TestSpeakerDiarization:
    """Test suite for cloud-native speaker diarization and stage guardrails."""

    def test_diarization_system_prompt_rules(self):
        """Diarization prompt must contain acoustic rules and formatting specification."""
        assert "vocal timbre" in DIARIZATION_SYSTEM_PROMPT
        assert "[Speaker Name] (00:00):" in DIARIZATION_SYSTEM_PROMPT
        assert "UNCERTAINTY TAGGING" in DIARIZATION_SYSTEM_PROMPT
        assert "VERBATIM ACCURACY" in DIARIZATION_SYSTEM_PROMPT
        assert "MULTI-SPEAKER AWARENESS" in DIARIZATION_SYSTEM_PROMPT
        assert "CONVERSATIONAL TURN DETECTION" in DIARIZATION_SYSTEM_PROMPT
        assert "SPEAKER IDENTIFICATION & NAMING" in DIARIZATION_SYSTEM_PROMPT

    def test_refiner_guardrail_preserves_speaker_labels(self):
        """Refiner system prompt must strictly guard speaker tags and split collapsed turns."""
        assert "PRESERVE SPEAKER LABELS" in REFINER_SYSTEM_PROMPT
        assert "Never merge distinct speaker turns or remove speaker identifiers." in REFINER_SYSTEM_PROMPT
        assert "SEMANTIC SPEAKER RESOLUTION & TURN SPLITTING" in REFINER_SYSTEM_PROMPT

    def test_extractor_guardrail_speaker_attribution(self):
        """Extractor system prompt must instruct model to attribute action items to identified speakers."""
        assert "SPEAKER ATTRIBUTION" in DOCUMENTER_SYSTEM_PROMPT
        assert "record their name as 'owner'" in DOCUMENTER_SYSTEM_PROMPT
        assert "Unspecified" in DOCUMENTER_SYSTEM_PROMPT

    def test_openai_routing_to_gemini_when_available(self, monkeypatch, tmp_path: Path):
        """When OpenAI STT is configured and Gemini API key is present, audio routes to Gemini diarization."""
        cfg = Config(
            stt_provider="openai",
            openai_api_key="sk-mock-valid-key-1234567890",
            gemini_api_key="AIza-valid-gemini-key-1234567890",
            llm_provider="gemini",
        )
        transcriber = CloudTranscriber(cfg)

        called = {}
        def mock_transcribe_gemini(path):
            called["gemini"] = True
            return "[Alex] (00:00): Hello team", [{"id": 0, "speaker": "Alex", "text": "Hello team"}]

        monkeypatch.setattr(transcriber, "_transcribe_gemini", mock_transcribe_gemini)

        # Create dummy audio file
        test_audio = tmp_path / "meeting.wav"
        test_audio.write_bytes(b"\x00" * 512)

        raw_text, segments = transcriber.transcribe(test_audio)
        assert called.get("gemini") is True
        assert "[Alex]" in raw_text
        assert len(segments) == 1
        assert segments[0]["speaker"] == "Alex"


class TestGeminiKeyPool:
    """Test suite for Gemini dynamic API key pooling, rotation, and failover."""

    def test_rotation_on_429_status_code(self):
        """When an API call fails with HTTP 429, pool rotates to next key and succeeds."""
        pool = GeminiKeyPool(["AIzaSy-primary-key-1111", "AIzaSy-secondary-key-2222"])

        call_log = []

        def mock_gemini_api(api_key: str, prompt: str) -> str:
            call_log.append(api_key)
            if api_key == "AIzaSy-primary-key-1111":
                class HTTP429Error(Exception):
                    status_code = 429
                raise HTTP429Error("Rate limit reached. Please retry.")
            return f"Processed prompt: {prompt}"

        result = pool.execute_with_fallback(mock_gemini_api, "Summarize meeting")

        assert result == "Processed prompt: Summarize meeting"
        assert call_log == ["AIzaSy-primary-key-1111", "AIzaSy-secondary-key-2222"]
        assert pool.available_count == 1
        masked = pool.get_masked_keys()
        assert masked[0]["exhausted"] is True
        assert masked[1]["is_active"] is True

    def test_rotation_on_resource_exhausted_message(self):
        """When error message contains quota/ResourceExhausted keywords, pool rotates."""
        pool = GeminiKeyPool(["AIzaSy-first-key-AAAA", "AIzaSy-second-key-BBBB"])

        call_log = []

        def mock_gemini_api(api_key: str) -> str:
            call_log.append(api_key)
            if api_key == "AIzaSy-first-key-AAAA":
                raise Exception("ResourceExhausted: Quota exceeded for model gemini-1.5-flash")
            return "Success"

        result = pool.execute_with_fallback(mock_gemini_api)
        assert result == "Success"
        assert len(call_log) == 2
        assert pool.get_active_key() == "AIzaSy-second-key-BBBB"

    def test_all_keys_exhausted_raises_error(self):
        """When all keys in the pool trigger 429/quota errors, AllKeysExhaustedError is raised."""
        pool = GeminiKeyPool(["AIzaSy-key-one-1111", "AIzaSy-key-two-2222"])

        def failing_api(api_key: str):
            class QuotaExceededError(Exception):
                pass
            raise QuotaExceededError("Rate limit exceeded 429")

        with pytest.raises(AllKeysExhaustedError) as exc_info:
            pool.execute_with_fallback(failing_api)

        assert "All Gemini API keys in the pool have exceeded their quota" in str(exc_info.value)
        assert pool.available_count == 0

    def test_non_quota_error_not_rotated(self):
        """Non-quota errors (syntax, network, ValueError) re-raise immediately without exhaustion."""
        pool = GeminiKeyPool(["AIzaSy-key-alpha-1111", "AIzaSy-key-beta-2222"])

        call_log = []

        def broken_api(api_key: str):
            call_log.append(api_key)
            raise ValueError("Malformed JSON input payload")

        with pytest.raises(ValueError) as exc_info:
            pool.execute_with_fallback(broken_api)

        assert "Malformed JSON input payload" in str(exc_info.value)
        assert len(call_log) == 1
        # Key must not be marked exhausted
        assert pool.available_count == 2

    def test_key_pool_deduplication_and_sanitization(self):
        """Pool trims whitespace, filters empty/short keys, and deduplicates."""
        pool = GeminiKeyPool([
            "  AIzaSy-valid-key-9999  ",
            "AIzaSy-valid-key-9999",  # duplicate
            "",                       # empty
            "short",                  # < 8 chars
            "AIzaSy-second-key-8888",
        ])

        assert pool.total_count == 2
        assert pool.get_active_key() == "AIzaSy-valid-key-9999"

    def test_get_masked_keys_format(self):
        """get_masked_keys returns correct dictionary format for UI rendering."""
        pool = GeminiKeyPool(["AIzaSy1234567890ABCDEF"])
        masked = pool.get_masked_keys()

        assert len(masked) == 1
        assert "masked" in masked[0]
        assert "is_active" in masked[0]
        assert "exhausted" in masked[0]
        assert masked[0]["is_active"] is True
        assert masked[0]["exhausted"] is False
        assert masked[0]["masked"].startswith("AIzaSy")

    def test_single_key_setup_backward_compatibility(self):
        """Existing single-key setup in Config continues to work seamlessly."""
        cfg = Config(
            stt_provider="openai",
            openai_api_key="sk-mock-valid-key-1234567890",
            gemini_api_key="AIzaSy-single-legacy-key-9999",
            llm_provider="gemini",
        )

        assert cfg.key_pool is not None
        assert cfg.key_pool.total_count == 1
        assert cfg.key_pool.get_active_key() == "AIzaSy-single-legacy-key-9999"
        assert cfg.gemini_api_keys == ["AIzaSy-single-legacy-key-9999"]

    def test_list_keys_config_initialization(self):
        """Config accepts list of gemini_api_keys."""
        cfg = Config(
            stt_provider="openai",
            openai_api_key="sk-mock-valid-key-1234567890",
            gemini_api_keys=[
                "AIzaSy-list-key-AAAA-1111",
                "AIzaSy-list-key-BBBB-2222",
            ],
            llm_provider="gemini",
        )

        assert cfg.key_pool is not None
        assert cfg.key_pool.total_count == 2
        assert cfg.gemini_api_key == "AIzaSy-list-key-AAAA-1111"
        assert cfg.key_pool.get_active_key() == "AIzaSy-list-key-AAAA-1111"



