# Real Audio Testing & Benchmark Samples

This directory is designated for testing the AI Meeting Assistant with **real pre-recorded meeting audio files**.

> **Important**: This project does **NOT** rely on synthetic TTS audio or local PyTorch/CUDA weights. All transcription and extraction are executed via cloud APIs.

---

## 1. Adding Meeting Recordings

Copy your real pre-recorded meeting audio files into this directory:
```bash
# Example audio file placements:
samples/test_meeting.wav
samples/standup_meeting.mp3
samples/architecture_review.m4a
```

### Supported Formats
- `.wav`
- `.mp3`
- `.m4a`
- `.ogg`
- `.flac`
- `.webm`

---

## 2. Running Headless Evaluation

Execute the headless benchmark runner targeting your placed audio file:

```bash
# Run evaluation on a specific audio recording:
python run_evaluation.py --audio samples/test_meeting.wav

# Or run batch evaluation across all recordings in samples/:
python run_evaluation.py
```

---

## 3. Benchmark Outputs

Running the evaluation pipeline automatically writes the verified benchmark artifacts into this directory (or the configured `--output-dir`):
- `samples/<audio_filename>_record.json`: Complete serialized `MeetingRecord` with 2-space indentation.
- `samples/<audio_filename>_summary.md`: Standardized executive Markdown report containing the executive summary, key decisions with verbatim quotes, action items table with neutral 'Unspecified' badges, and full transcripts.

---

## 4. Privacy & Git Exclusions

Real meeting recordings (`*.mp3`, `*.wav`, `*.m4a`, etc.) are explicitly excluded in `.gitignore` to prevent proprietary audio or sensitive organizational discussions from being accidentally pushed to version control.
