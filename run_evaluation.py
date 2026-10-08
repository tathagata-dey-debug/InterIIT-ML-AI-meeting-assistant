"""
Headless Benchmark Runner for AI Meeting Assistant (Inter IIT Tech Meet 15.0).
Provides non-GUI automated evaluation across real meeting recordings.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path for robust imports regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Config, ConfigurationError
from core.audio_processor import (
    AudioProcessor,
    AudioProcessingError,
    EmptyAudioError,
    UnsupportedAudioFormatError,
    CorruptAudioError,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from core.transcriber import CloudTranscriber, TranscriptionError
from core.refiner import TranscriptRefiner
from core.extractor import MeetingDocumenter
from core.exporter import export_to_json, export_to_markdown
from core.schemas import MeetingRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark_runner")


def evaluate_single_recording(
    audio_path: Path,
    config: Config,
    output_dir: Path,
    domain_glossary: str = "",
) -> Tuple[Path, Path]:
    """
    Execute end-to-end multi-stage pipeline on a real meeting recording.

    Args:
        audio_path: Path to target audio file.
        config: Centralized pipeline configuration.
        output_dir: Target folder where benchmark outputs are saved.
        domain_glossary: Optional domain terminology hints.

    Returns:
        Tuple of (json_artifact_path, markdown_artifact_path).
    """
    logger.info("=" * 60)
    logger.info(f"BENCHMARK EVALUATION: {audio_path.name}")
    logger.info("=" * 60)

    start_total = time.time()

    # Step 0: Audio Validation
    logger.info("Step 0: Validating audio integrity...")
    AudioProcessor.validate_file(audio_path, audio_path.name)

    # Step 1: Audio Conditioning & Normalization
    t0 = time.time()
    scratch_dir = Path("scratch/benchmark")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    norm_audio_target = scratch_dir / f"norm_{audio_path.stem}.mp3"

    norm_audio_path = AudioProcessor.preprocess_audio(
        str(audio_path),
        str(norm_audio_target),
    )
    t_condition = time.time() - t0
    logger.info(f"[Stage 0] Conditioning completed in {t_condition:.2f}s -> {norm_audio_path}")

    # Step 2: Stage 1 - Cloud Speech-to-Text with Uncertainty Flagging
    t1 = time.time()
    transcriber = CloudTranscriber(config)
    raw_transcript, segments = transcriber.transcribe(norm_audio_path)
    t_stt = time.time() - t1
    uncertain_count = raw_transcript.count("[unclear:")
    total_words = len(raw_transcript.split())

    logger.info(
        f"[Stage 1 - STT] Completed in {t_stt:.2f}s | "
        f"Words: {total_words} | Uncertain Tokens: {uncertain_count} | Segments: {len(segments)}"
    )

    # Step 3: Stage 2 - Language Model 1 (Domain-Aware Refinement)
    t2 = time.time()
    refiner = TranscriptRefiner(config)
    refined_transcript = refiner.refine(
        raw_transcript,
        domain_context=domain_glossary.strip() if domain_glossary else None,
    )
    t_refine = time.time() - t2
    logger.info(
        f"[Stage 2 - LLM 1] Refinement completed in {t_refine:.2f}s | "
        f"Refined character length: {len(refined_transcript)}"
    )

    # Step 4: Stage 3 - Language Model 2 (Structured Minutes & Deliverables)
    t3 = time.time()
    documenter = MeetingDocumenter(config)
    minutes = documenter.document(refined_transcript)
    t_extract = time.time() - t3
    logger.info(
        f"[Stage 3 - LLM 2] Extraction completed in {t_extract:.2f}s | "
        f"Decisions: {len(minutes.key_decisions)} | Action Items: {len(minutes.action_items)}"
    )

    # Step 5: Deliverables Assembly & Parity Serialization
    record = MeetingRecord(
        raw_transcript=raw_transcript,
        refined_transcript=refined_transcript,
        minutes=minutes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{audio_path.stem}_record.json"
    md_path = output_dir / f"{audio_path.stem}_summary.md"

    json_content = export_to_json(record)
    md_content = export_to_markdown(record)

    json_path.write_text(json_content, encoding="utf-8")
    md_path.write_text(md_content, encoding="utf-8")

    total_time = time.time() - start_total

    logger.info("-" * 60)
    logger.info(f"SUCCESS: Pipeline finished in {total_time:.2f}s")
    logger.info(f"Artifact [JSON]:     {json_path.resolve()}")
    logger.info(f"Artifact [Markdown]: {md_path.resolve()}")
    logger.info("-" * 60)

    # Executive Summary Console Output
    print("\n--- EXECUTIVE SUMMARY ---")
    print(minutes.concise_summary)
    print("\n--- KEY DECISIONS ---")
    if minutes.key_decisions:
        for idx, dec in enumerate(minutes.key_decisions, start=1):
            print(f"{idx}. {dec.decision_text} (Quote: \"{dec.supporting_quote}\")")
    else:
        print("No explicit decisions were finalized.")

    print("\n--- ACTION ITEMS ---")
    if minutes.action_items:
        for idx, item in enumerate(minutes.action_items, start=1):
            print(f"{idx}. {item.task_description} | Owner: {item.owner} | Due: {item.deadline}")
    else:
        print("No actionable tasks were assigned.")
    print()

    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless Benchmark Runner for AI Meeting Assistant (Inter IIT Tech Meet 15.0)."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to specific real meeting recording (e.g., samples/test_meeting.wav).",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="samples",
        help="Directory containing benchmark samples when --audio is omitted (default: 'samples').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for output artifacts (defaults to same folder as audio).",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default="keys/api_keys.json",
        help="Path to API keys JSON file (default: 'keys/api_keys.json').",
    )
    parser.add_argument(
        "--glossary",
        type=str,
        default="",
        help="Optional domain terminology or glossary string.",
    )

    args = parser.parse_args()

    # Load and validate credentials
    try:
        config = Config.load(keys_path=args.keys)
        logger.info(
            f"Config Loaded: STT={config.stt_provider} ({config.stt_model}) | "
            f"LLM={config.llm_provider} ({config.llm_model})"
        )
    except ConfigurationError as err:
        logger.error(f"Configuration Error: {err}")
        sys.exit(1)

    # Determine files to process
    if args.audio:
        target_file = Path(args.audio).resolve()
        if not target_file.exists():
            logger.error(f"Specified audio file not found: {target_file}")
            sys.exit(1)
        suffix = target_file.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            logger.error(
                f"Unsupported audio format: '{suffix}'. "
                "Allowed formats: WAV, MP3, M4A, OGG."
            )
            sys.exit(1)
        out_dir = Path(args.output_dir) if args.output_dir else target_file.parent
        try:
            evaluate_single_recording(target_file, config, out_dir, args.glossary)
        except Exception as exc:
            logger.error(f"Evaluation failed for {target_file.name}: {exc}", exc_info=True)
            sys.exit(1)
    else:
        # Batch evaluation across samples directory
        samples_path = Path(args.samples_dir).resolve()
        files_to_process: List[Path] = [
            f for f in samples_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS and f.stat().st_size > 0
        ]

        if not files_to_process:
            logger.warning(
                f"No real audio files discovered in '{samples_path}'. "
                "Please place a pre-recorded meeting audio in 'samples/' (e.g. samples/test_meeting.wav) "
                "or specify via: python run_evaluation.py --audio samples/test_meeting.wav"
            )
            sys.exit(0)

        out_dir = Path(args.output_dir) if args.output_dir else samples_path
        logger.info(f"Discovered {len(files_to_process)} real meeting recordings in '{samples_path}'")

        for audio_file in files_to_process:
            try:
                evaluate_single_recording(audio_file, config, out_dir, args.glossary)
            except Exception as exc:
                logger.error(f"Failed on '{audio_file.name}': {exc}", exc_info=True)


if __name__ == "__main__":
    main()
