# AI Meeting Assistant | Inter IIT Tech Meet 15.0

An end-to-end, multi-model meeting intelligence pipeline designed for verbatim spoken dialogue capture, technical domain refinement, and hallucination-free structured extraction. Built for reproducible local and headless evaluation without local GPU or heavy driver requirements.

---

## 1. Project Overview & Core Philosophy

### The Problem
Engineering and product meetings are notoriously messy. Real-world discussions are packed with technical abbreviations, unfinished sentences, conversational interruptions, and spoken jargon. Off-the-shelf automated meeting systems routinely fail in two distinct failure modes:
1. **Phonetic & Jargon Degradation**: Standard Speech-to-Text (STT) models stumble on specialized technical vocabulary—turning *"Kubeflow"* into *"cube flow"*, *"PostgreSQL"* into *"post grass"*, or mangling library names and architecture terms.
2. **Generative Hallucination**: Downstream LLMs tasked with creating minutes often hallucinate consensus where only an idea was pitched, fabricate arbitrary deadlines ("by Friday"), or assign ownership to participants who merely asked a question.

### Our Architectural Decision: Zero Heavy Local Weights
Most academic submissions attempt to bundle 15 GB+ local models (Whisper-large, PyTorch, Pyannote, CUDA runtimes). In practice, this leads to fragile environments, CUDA version mismatches, and minutes of setup time on unfamiliar evaluation machines.

Instead, we designed an **asynchronous, deterministic 3-stage pipeline powered by managed cloud API models**:
- **Zero local PyTorch or CUDA dependencies**: The repository clones and installs in under 60 seconds on standard CPU hardware (macOS, Linux, or Windows).
- **Decoupled model responsibilities**: Transcription, domain correction, and structured extraction are isolated into three independent stages to prevent prompt bleed and model confusion.
- **Strict deterministic validation**: Every structured field is strictly enforced through Pydantic v2 schemas with invariant preservation rules and evidence grounding.

---

## 2. How the Pipeline Works (The 3 Distinct Stages)

```
                         [ Audio Input ]
                 (.wav, .mp3, .m4a, .ogg)
                            │
                            ▼
      ┌──────────────────────────────────────────────┐
      │       Pre-Stage: Audio Conditioning          │
      │  - Strict format & size validation (core)    │
      │  - 0-byte & silence rejection                │
      │  - Loudness normalization to -20 dBFS        │
      │  - Dynamic re-encoding if payload > 24 MB    │
      └─────────────────────┬────────────────────────┘
                            │
                            ▼
      ┌──────────────────────────────────────────────┐
      │  Stage 1: Speech-to-Text & Diarization       │
      │  (Gemini Multimodal / OpenAI Whisper)        │
      │  - Verbatim multi-speaker transcription      │
      │  - Acoustic-semantic speaker turn tracking   │
      │  - Uncertainty Tagging (<0.35 confidence     │
      │    flagged as [unclear: word?])              │
      └─────────────────────┬────────────────────────┘
                            │ Raw Transcript + Uncertainty Flags
                            ▼
      ┌──────────────────────────────────────────────┐
      │  Stage 2 (LLM 1): Domain Refinement          │
      │  (Gemini 2.5 Flash / GPT-4o-mini)            │
      │  - Dedicated technical proofreader           │
      │  - Context-aware jargon & acronym alignment  │
      │  - Invariant preservation (numbers, names,   │
      │    negations like "not" / "cannot")          │
      │  - Disfluency cleaning without drops/adds    │
      └─────────────────────┬────────────────────────┘
                            │ Refined Verbatim Text
                            ▼
      ┌──────────────────────────────────────────────┐
      │  Stage 3 (LLM 2): Structured Extraction      │
      │  (Gemini 2.5 Flash / GPT-4o-mini)            │
      │  - Schema enforcement via Pydantic v2        │
      │  - Strict Consensus Rule (proposals ≠ done)  │
      │  - Mandatory verbatim quote grounding        │
      │  - Unspecified defaults for owner/deadline   │
      └─────────────────────┬────────────────────────┘
                            │
                            ▼
      ┌──────────────────────────────────────────────┐
      │         Verified Dual Deliverables           │
      │  - meeting_record.json (Machine-readable)    │
      │  - meeting_summary.md  (Human-readable)      │
      └──────────────────────────────────────────────┘
```

### Stage 1: Speech-to-Text & Acoustic Diarization
- **Model**: Gemini Multimodal Audio (primary) / OpenAI Whisper-1 (alternative).
- **Role**: Captures spoken dialogue verbatim without guessing or rewriting grammar. Employs acoustic turn tracking to output structured speaker turns (`[Speaker Name] (MM:SS): speech text...`).
- **Uncertainty Filter**: During transcription, phonetic segments or tokens with confidence scores below 0.35 (or $\text{avg\_logprob} < -1.0$) are flagged explicitly as `[unclear: word?]`. This prevents downstream components from treating garbled audio as verified fact.

### Stage 2: LLM 1 – Domain Refinement
- **Model**: Gemini 2.5 Flash / GPT-4o-mini.
- **Role**: Operates strictly as an expert technical proofreader and conversational discourse analyst. It analyzes the context of the discussion (e.g., cloud infrastructure, machine learning, databases, finance) to resolve phonetic slips and technical jargon (*"dock er"* $\rightarrow$ *"Docker"*, *"cube ctl"* $\rightarrow$ *"kubectl"*).
- **Invariant Preservation Guardrails**:
  - **Negations**: Words like "not", "cannot", "won't", and "never" must remain untouched.
  - **Factual Invariants**: Proper names, financial figures, dates, and version tags are strictly preserved.
  - **Speaker Boundary Integrity**: Speaker tags and chronological flow are maintained without collapsing distinct interlocutors.

### Stage 3: LLM 2 – Structured Extraction & Anti-Hallucination
- **Model**: Gemini 2.5 Flash / GPT-4o-mini.
- **Role**: Transforms the refined transcript into typed, validated meeting minutes, confirmed decisions, and actionable tasks via Pydantic v2 schemas.
- **Strict Guardrails**:
  - **Decisions vs. Proposals**: Items under discussion, preliminary pitches, or rejected suggestions are never marked as decisions. A decision requires explicit group agreement and a mandatory verbatim `supporting_quote`.
  - **Conservative Task Attribution**: Action item owners and deadlines are populated **only** when explicitly stated in dialogue. If a task is mentioned without an explicit volunteer or deadline, the fields strictly default to `"Unspecified"`.

---

## 3. Quickstart & Setup Guide

### Option A: 1-Click Automated Setup (Recommended)

#### Linux / macOS:
```bash
chmod +x setup.sh
./setup.sh
```

#### Windows:
```cmd
setup.bat
```

The script verifies Python 3.10+, provisions a local virtual environment (`venv`), installs all requirements, sets up the credential template at `keys/api_keys.json`, and displays execution commands.

---

### Option B: Manual Setup (4 Steps)

1. **Clone and enter the directory**:
   ```bash
   git clone <repo-url>
   cd meeting-assistant
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure API Credentials**:
   Copy the example keys configuration:
   ```bash
   # Linux / macOS
   cp keys/api_keys.example.json keys/api_keys.json

   # Windows
   copy keys\api_keys.example.json keys\api_keys.json
   ```
   Add your Gemini or OpenAI API keys inside `keys/api_keys.json`:
   ```json
   {
     "STT_PROVIDER": "gemini",
     "LLM_PROVIDER": "gemini",
     "GEMINI_API_KEYS": [
       "AIzaSy-your-primary-key",
       "AIzaSy-your-backup-key"
     ],
     "GEMINI_API_KEY": "AIzaSy-your-primary-key"
   }
   ```
   *(Note: The system also respects root `.env` and system environment variables).*

---

### Running the System

#### 1. Interactive Web Interface
Launch the Streamlit web app:
```bash
streamlit run app.py
```
Open `http://localhost:8501` to upload audio files, monitor live processing progress, review side-by-side transcripts, and export records.

#### 2. Headless CLI Runner
For automated evaluation, batch scoring, or headless server execution:
```bash
# Run single file evaluation:
python run_evaluation.py --audio samples/test_meeting.wav

# Run with custom domain glossary hint:
python run_evaluation.py --audio samples/test_meeting.wav --glossary "Kubernetes, ArgoCD, Prometheus, Grafana"

# Batch process all audio files in samples/:
python run_evaluation.py
```
Outputs are written directly to `samples/<audio_stem>_record.json` and `samples/<audio_stem>_summary.md`.

#### 3. Test Suite
Execute the test suite to verify configuration loading, audio validation, schema constraints, and edge case handling:
```bash
pytest tests/ -v
```

---

## 4. Interface & Output Deliverables

### Streamlit Application Features
- **File Validation & Audio Player**: Immediate file format and size checks with an embedded audio player for quick spot-checks.
- **Dynamic Key Pool Manager**: View active, standby, and exhausted API keys with live rotation indicators. Add fallback keys on the fly.
- **Side-by-Side Transcript Comparison**: Raw transcript (with highlighted `[unclear: word?]` tags) alongside the refined transcript to inspect domain corrections instantly.
- **Executive Minutes & Decisions**: Overview section featuring meeting agenda topics, context, and confirmed decisions accompanied by verbatim quotes.
- **Action Items Table**: Clear matrix showing task description, assignee, and target deadline with prominent `"Unspecified"` fallback badges.
- **Dual Export Buttons**: Instant one-click downloads for machine-readable JSON (`meeting_record.json`) and formatted report Markdown (`meeting_summary.md`).

### Output Parity Guarantee
Both `meeting_record.json` and `meeting_summary.md` are derived from the exact same validated Pydantic model (`MeetingRecord`). The structured data guarantees 1:1 parity between downstream machine ingestion and human-readable documentation:

```text
exports/
├── <meeting>_record.json    # Strict schema: metadata, transcripts, decisions, tasks
└── <meeting>_summary.md     # Formatted markdown: tables, quotes, summaries, audit logs
```

---

## 5. Evaluation Rubric Compliance Checklist

| Rubric Component | Weight | Implementation Details in This Repository | Verified By |
| :--- | :---: | :--- | :--- |
| **Speech Transcription** | **20 pts** | - Verbatim audio capture without omissions.<br>- Multi-speaker acoustic turn tracking with chronological timestamps.<br>- Preservation of proper nouns, numbers, currencies, and negations.<br>- Token-level confidence filtering tagging low-certainty audio as `[unclear: word?]`. | `core/transcriber.py`<br>`tests/test_pipeline.py` |
| **Transcript Refinement** | **20 pts** | - Dedicated Stage 2 LLM acting as domain-aware proofreader.<br>- Contextual correction of technical jargon (DevOps, databases, APIs).<br>- Strict preservation of invariant truths (names, numbers, negations, speaker tags).<br>- Zero fabrication or unprompted summarization in this stage. | `core/refiner.py`<br>`tests/test_pipeline.py` |
| **Minutes & Decisions** | **25 pts** | - Structured executive summary and topic breakdown.<br>- Strict consensus rule: distinguishes confirmed decisions from open proposals.<br>- Every confirmed decision requires a verbatim `supporting_quote` from the transcript. | `core/extractor.py`<br>`core/schemas.py` |
| **Action Items** | **15 pts** | - Discrete, actionable task identification.<br>- Conservative attribution: assigns owner and deadline ONLY if explicitly stated.<br>- Strict fallback: assigns `"Unspecified"` when owner or deadline is omitted. | `core/extractor.py`<br>`tests/test_pipeline.py` |
| **End-to-End Application** | **15 pts** | - Single-click execution via Streamlit UI or headless CLI (`run_evaluation.py`).<br>- Automatic Gemini API key pooling with transparent failover on 429/quota limits.<br>- Robust error handling: rejects empty, unsupported, or corrupt audio files with clean errors. | `app.py`<br>`run_evaluation.py`<br>`core/audio_processor.py` |
| **Submission Quality** | **5 pts** | - Modular repository layout without GPU bloat.<br>- Cross-platform 1-click setup scripts (`setup.sh`, `setup.bat`).<br>- Bundled test sample (`samples/test_meeting.wav`) with pre-computed outputs.<br>- 100% test pass rate across unit and integration tests. | Root repository<br>`setup.sh` / `setup.bat`<br>`samples/` |
| **Total Score** | **100 pts** | **Fully compliant with all Inter IIT Tech Meet 15.0 requirements.** | `pytest tests/ -v` |

---

## Repository Structure

```text
meeting-assistant/
├── README.md                      # Complete project documentation and rubric guide
├── requirements.txt               # Pinned lightweight dependencies
├── setup.sh                       # 1-click automated setup for macOS and Linux
├── setup.bat                      # 1-click automated setup for Windows
├── app.py                         # Streamlit interactive web interface
├── run_evaluation.py              # Headless CLI benchmark evaluation script
├── .gitignore                     # Git exclusion rules preventing credential leaks
├── keys/                          # Credential management directory
│   ├── README.md                  # Credential setup & 429 rotation guide
│   ├── api_keys.example.json      # Safe configuration template
│   └── api_keys.json              # Local active credentials (git-ignored)
├── core/                          # Modular core engine
│   ├── __init__.py
│   ├── config.py                  # Pydantic settings & credential loader
│   ├── key_manager.py             # Dynamic Gemini key pooling & 429 failover
│   ├── audio_processor.py         # Audio validation, normalization & compression
│   ├── transcriber.py             # Stage 1: Cloud STT with uncertainty tagging
│   ├── refiner.py                 # Stage 2: Domain-aware transcript proofreader
│   ├── extractor.py               # Stage 3: Structured minutes & action item extractor
│   ├── schemas.py                 # Pydantic v2 data models with consensus rules
│   └── exporter.py                # Dual JSON and Markdown serialization with strict parity
├── samples/                       # Evaluation audio samples and generated outputs
│   ├── README.md                  # Audio testing guide
│   ├── test_meeting.wav           # Bundled test meeting recording
│   ├── test_meeting_record.json   # Machine-readable evaluation deliverable
│   └── test_meeting_summary.md    # Human-readable evaluation deliverable
└── tests/                         # Pytest test suite
    └── test_pipeline.py           # Unit and integration tests covering the complete pipeline
```
