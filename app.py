"""
Streamlit Web Frontend for AI Meeting Assistant (Inter IIT Tech Meet 15.0).
Phase 4: Responsive, domain-aware meeting transcription and structured documentation UI.
Updated: Audio pipeline with bundled imageio-ffmpeg & Gemini 3.5 Flash Lite support.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path for robust imports regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import os
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore", message=".*Couldn't find ffprobe or avprobe.*")

# Configure ffmpeg and patch pydub inside active process memory
try:
    import imageio_ffmpeg
    from pydub import AudioSegment
    import pydub.utils
    import pydub.audio_segment

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
        AudioSegment.converter = _ffmpeg_exe
    elif shutil.which("ffmpeg"):
        AudioSegment.converter = shutil.which("ffmpeg")

    # Safely bypass missing ffprobe to decode audio via ffmpeg
    pydub.utils.mediainfo_json = lambda *args, **kwargs: None
    pydub.audio_segment.mediainfo_json = lambda *args, **kwargs: None
except Exception:
    pass

import core.key_manager
importlib.reload(core.key_manager)
import core.config
importlib.reload(core.config)
import core.audio_processor
importlib.reload(core.audio_processor)

from core.config import Config, ConfigurationError
from core.key_manager import GeminiKeyPool, AllKeysExhaustedError
from core.audio_processor import (
    AudioProcessor,
    AudioProcessingError,
    EmptyAudioError,
    UnsupportedAudioFormatError,
    CorruptAudioError,
    get_audio_mime_type,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from core.transcriber import CloudTranscriber, TranscriptionError
from core.refiner import TranscriptRefiner
from core.extractor import MeetingDocumenter
from core.exporter import export_to_json, export_to_markdown
from core.schemas import MeetingRecord

# ==========================================
# Streamlit Page Setup & Custom Styling
# ==========================================

st.set_page_config(
    page_title="AI Meeting Assistant | Inter IIT Tech Meet 15.0",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive aesthetic and responsive typography
st.markdown(
    """
    <style>
    /* Global Typography & Font Enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Badges & Pills */
    .status-badge-connected {
        display: inline-flex;
        align-items: center;
        background-color: rgba(34, 197, 94, 0.16);
        color: #22c55e !important;
        border: 1px solid rgba(34, 197, 94, 0.38);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-badge-missing {
        display: inline-flex;
        align-items: center;
        background-color: rgba(239, 68, 68, 0.16);
        color: #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.38);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Transcripts Box: Theme Adaptive */
    .transcript-box {
        background-color: var(--secondary-background-color, #1e293b);
        color: var(--text-color, #f8fafc) !important;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        padding: 20px;
        height: 480px;
        overflow-y: auto;
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Uncertainty Token Highlight */
    .unclear-tag {
        background-color: rgba(245, 158, 11, 0.18);
        color: #f59e0b !important;
        border: 1px solid rgba(245, 158, 11, 0.45);
        padding: 2px 7px;
        border-radius: 5px;
        font-weight: 600;
        font-family: monospace;
    }

    /* Speaker Diarization Badges */
    .speaker-tag {
        display: inline-block;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 2px 9px;
        border-radius: 6px;
        margin-right: 6px;
        letter-spacing: 0.02em;
        vertical-align: middle;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.18);
    }
    .timestamp-tag {
        display: inline-block;
        background-color: rgba(148, 163, 184, 0.16);
        color: var(--text-color, #94a3b8) !important;
        font-family: monospace;
        font-size: 0.80rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 5px;
        margin-right: 8px;
        border: 1px solid rgba(148, 163, 184, 0.28);
        vertical-align: middle;
    }

    /* Cards: Theme Adaptive */
    .decision-card {
        background-color: var(--secondary-background-color, #1e293b);
        color: var(--text-color, #f8fafc) !important;
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-left: 5px solid #3b82f6;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .decision-card-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--text-color, #f8fafc) !important;
        margin-bottom: 10px;
    }
    .decision-card-quote {
        font-style: italic;
        color: var(--text-color, #e2e8f0) !important;
        background-color: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.22);
        padding: 10px 14px;
        border-radius: 6px;
        line-height: 1.6;
    }

    /* Neutral Tag for Unspecified values */
    .badge-unspecified {
        background-color: rgba(148, 163, 184, 0.16);
        color: var(--text-color, #94a3b8) !important;
        border: 1px solid rgba(148, 163, 184, 0.28);
        padding: 3px 9px;
        border-radius: 5px;
        font-weight: 500;
        font-family: monospace;
        font-size: 0.82rem;
    }

    /* Executive summary box: Theme Adaptive */
    .summary-card {
        background-color: var(--secondary-background-color, #1e293b);
        color: var(--text-color, #f8fafc) !important;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 5px solid #10b981;
        border-radius: 10px;
        padding: 20px 24px;
        font-size: 1.05rem;
        line-height: 1.75;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    /* Action Items Table: Theme Adaptive */
    .action-table {
        width: 100%;
        border-collapse: collapse;
        background-color: var(--secondary-background-color, #1e293b);
        color: var(--text-color, #f8fafc) !important;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        overflow: hidden;
    }
    .action-table th {
        background-color: rgba(148, 163, 184, 0.12);
        color: var(--text-color, #e2e8f0) !important;
        padding: 12px 16px;
        font-weight: 600;
        border-bottom: 2px solid rgba(148, 163, 184, 0.25);
        text-align: left;
    }
    .action-table td {
        padding: 12px 16px;
        color: var(--text-color, #f8fafc) !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .action-table tr:hover {
        background-color: rgba(148, 163, 184, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 1. Header & API Status Banner
# ==========================================

col_header_left, col_header_right = st.columns([0.72, 0.28])

with col_header_left:
    st.title("🎙️ AI Meeting Assistant")
    st.markdown("**Domain-Aware Audio Transcription & Structured Documentation Pipeline**")

if "gemini_key_pool" not in st.session_state:
    st.session_state["gemini_key_pool"] = []

# Check configuration & API credentials
config: Config | None = None
config_error_msg: str | None = None

try:
    config = Config.load()
    if config:
        pool = getattr(config, "key_pool", None)
        if pool is None:
            pool = GeminiKeyPool(getattr(config, "gemini_api_keys", []))
            try:
                config.key_pool = pool
            except Exception:
                pass
        for k in st.session_state["gemini_key_pool"]:
            pool.add_key(k)
except ConfigurationError as err:
    if st.session_state["gemini_key_pool"]:
        try:
            config = Config(
                gemini_api_keys=list(st.session_state["gemini_key_pool"]),
                gemini_api_key=st.session_state["gemini_key_pool"][0],
                stt_provider="gemini",
                llm_provider="gemini",
            )
        except Exception:
            config_error_msg = str(err)
    else:
        config_error_msg = str(err)
except Exception as ex:
    config_error_msg = f"Unexpected configuration error: {ex}"

with col_header_right:
    st.write("")
    st.write("")
    if config:
        st.markdown(
            '<div style="text-align: right;"><span class="status-badge-connected">● Cloud APIs Connected</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"STT: `{config.stt_provider}` ({config.stt_model}) | LLM: `{config.llm_provider}` ({config.llm_model})"
        )
    else:
        st.markdown(
            '<div style="text-align: right;"><span class="status-badge-missing">● API Keys Missing</span></div>',
            unsafe_allow_html=True,
        )

# Display instructional warning if credentials are not configured
if not config:
    st.warning(
        "⚠️ **Missing API Credentials**: Please populate `keys/api_keys.json` with your active API keys or configure environment variables.",
        icon="🔑",
    )
    with st.expander("Setup Instructions for Credentials", expanded=False):
        st.markdown(
            """
            1. Copy the example configuration template:
               ```bash
               cp keys/api_keys.example.json keys/api_keys.json
               ```
            2. Populate `keys/api_keys.json` with your real keys:
               ```json
               {
                 "STT_PROVIDER": "openai",
                 "OPENAI_API_KEY": "sk-proj-...",
                 "LLM_PROVIDER": "gemini",
                 "GEMINI_API_KEY": "AIzaSy..."
               }
               ```
            """
        )

st.markdown("---")

# Sidebar Configuration & Telemetry
with st.sidebar:
    st.header("⚙️ Pipeline Controls")

    domain_glossary = st.text_area(
        "Domain Terminology Glossary (Optional)",
        placeholder="e.g. Kubernetes, Kubeflow, PostgreSQL, PyTorch, Inter IIT, ArgoCD",
        help="Injected into Language Model 1 for contextual speech recognition error correction.",
        height=130,
    )

    st.markdown("---")
    st.subheader("🔑 Gemini Key Pool")
    st.caption("Automatic failover & rotation on token exhaustion or 429 rate limits.")

    new_key = st.text_input("Append Gemini API Key", type="password", placeholder="AIzaSy...")
    if st.button("Add Key to Pool", use_container_width=True):
        clean_new_key = new_key.strip() if new_key else ""
        if clean_new_key and len(clean_new_key) > 8:
            if clean_new_key not in st.session_state["gemini_key_pool"]:
                st.session_state["gemini_key_pool"].append(clean_new_key)
            if config is not None:
                pool = getattr(config, "key_pool", None)
                if pool is None:
                    pool = GeminiKeyPool(getattr(config, "gemini_api_keys", []))
                    try:
                        config.key_pool = pool
                    except Exception:
                        pass
                pool.add_key(clean_new_key)
                if not getattr(config, "gemini_api_key", None):
                    config.gemini_api_key = clean_new_key
            st.success("API Key successfully added to pool!")
            st.rerun()
        else:
            st.warning("Please enter a valid, non-empty Gemini API key.")

    # Key Pool Status Display
    pool = getattr(config, "key_pool", None) if config else None
    if pool is not None and pool.total_count > 0:
        avail_count = pool.available_count
        if avail_count > 0:
            st.markdown(f"**🟢 {avail_count} Keys Available**")
        else:
            st.markdown(f"**🔴 0 Keys Available**")

        masked_keys = pool.get_masked_keys()
        for item in masked_keys:
            m_key = item["masked"]
            if item["exhausted"]:
                st.markdown(f"`{m_key}` :red[[Exhausted]]")
            elif item["is_active"]:
                st.markdown(f"`{m_key}` :green[[Active]]")
            else:
                st.markdown(f"`{m_key}` :gray[[Standby]]")

        if st.button("Clear Pool / Reset Keys", use_container_width=True):
            st.session_state["gemini_key_pool"] = []
            if config:
                pool = getattr(config, "key_pool", None)
                if pool:
                    pool.clear()
                try:
                    fresh_cfg = Config.load()
                    config.key_pool = getattr(fresh_cfg, "key_pool", None)
                    config.gemini_api_keys = getattr(fresh_cfg, "gemini_api_keys", [])
                    config.gemini_api_key = getattr(fresh_cfg, "gemini_api_key", None)
                except Exception:
                    pass
            st.rerun()
    else:
        st.caption("No Gemini keys active in pool.")

    st.markdown("---")
    st.markdown("### 🧩 Multi-Model Architecture")
    st.markdown(
        """
        - **Audio Conditioning**: Loudness leveling to -20 dBFS
        - **Stage 1 (Cloud STT)**: Word confidence uncertainty tagging
        - **Stage 2 (LLM 1)**: Transcript proofreading & jargon alignment
        - **Stage 3 (LLM 2)**: Anti-hallucination minutes & tasks extraction
        """
    )

    if st.button("Clear Cached Analysis", use_container_width=True):
        if "meeting_record" in st.session_state:
            del st.session_state["meeting_record"]
        st.rerun()

# ==========================================
# 2. File Upload & Immediate Validation
# ==========================================

st.subheader("1. Meeting Audio Ingestion")

uploaded_file = st.file_uploader(
    "Upload Meeting Recording",
    type=["wav", "mp3", "m4a", "ogg"],
    help="Supported formats: WAV, MP3, M4A, OGG (Max 25MB recommended)",
)

file_is_valid = False
input_scratch_path: Path | None = None

if uploaded_file is not None:
    try:
        # Extract extension and dynamic MIME type
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        mime_type = get_audio_mime_type(ext)

        # Immediate validation via AudioProcessor
        file_bytes = uploaded_file.getvalue()
        AudioProcessor.validate_file(file_bytes, uploaded_file.name)
        file_is_valid = True

        # Safe Temporary Disk Write with preserved container suffix
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_f:
            tmp_f.write(file_bytes)
            input_scratch_path = Path(tmp_f.name)

        col_audio, col_meta = st.columns([0.7, 0.3])
        with col_audio:
            st.audio(uploaded_file, format=mime_type)
        with col_meta:
            size_mb = len(file_bytes) / (1024 * 1024)
            st.info(f"**Filename:** `{uploaded_file.name}`\n\n**Format:** `{ext.upper().lstrip('.')}` (`{mime_type}`)\n\n**Payload Size:** `{size_mb:.2f} MB`")

    except EmptyAudioError as err:
        st.error(f"❌ **Empty File Error:** {err}\n*Action:* Please upload a recording that contains valid audio data.")
    except UnsupportedAudioFormatError as err:
        st.error(f"❌ **Format Error:** {err}\n*Action:* Please supply a WAV, MP3, M4A, or OGG recording.")
    except CorruptAudioError as err:
        st.error(f"❌ **Corrupt Stream:** {err}\n*Action:* Verify that your audio contains audible speech.")
    except AudioProcessingError as err:
        st.error(f"❌ **Audio Processing Error:** {err}")
    except Exception as ex:
        st.error(f"❌ **Unexpected Error:** {ex}")

# Primary Run Action Button
run_analysis_btn = st.button(
    "🚀 Run End-to-End Meeting Analysis",
    type="primary",
    disabled=(not file_is_valid or config is None),
    use_container_width=True,
)

# ==========================================
# 3. Multi-Step Progress Tracker
# ==========================================

if run_analysis_btn and input_scratch_path and config:
    with st.status("Processing Meeting Recording...", expanded=True) as status:
        try:
            # Step 1: Conditioning
            st.write("Step 1: Validating & Normalizing Audio Stream...")
            processed_audio_path = Path("scratch/processed") / f"norm_{uploaded_file.name}.mp3"
            norm_audio_str = AudioProcessor.preprocess_audio(
                str(input_scratch_path),
                str(processed_audio_path),
            )

            # Step 2: Cloud STT
            st.write("Step 2: Transcribing Audio via Cloud STT API (Stage 1)...")
            transcriber = CloudTranscriber(config)
            raw_transcript_text, segments = transcriber.transcribe(norm_audio_str)

            # Step 3: LLM 1 Refinement
            st.write("Step 3: Refining Domain Terminology & Correcting Jargon (Stage 2 - LLM 1)...")
            refiner = TranscriptRefiner(config)
            refined_transcript_text = refiner.refine(
                raw_transcript_text,
                domain_context=domain_glossary.strip() if domain_glossary else None,
            )

            # Step 4: LLM 2 Extraction
            st.write("Step 4: Extracting Minutes, Decisions & Action Items (Stage 3 - LLM 2)...")
            documenter = MeetingDocumenter(config)
            minutes_model = documenter.document(refined_transcript_text)

            # Step 5: Deliverables Compilation
            st.write("Step 5: Compiling Structured Deliverables...")
            record = MeetingRecord(
                raw_transcript=raw_transcript_text,
                refined_transcript=refined_transcript_text,
                minutes=minutes_model,
            )

            # Save in Streamlit session state for persistent rendering across re-renders
            st.session_state["meeting_record"] = record
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        except TranscriptionError as err:
            status.update(label="Speech-to-Text Failed", state="error", expanded=True)
            st.error(f"STT Error: {err}")
        except ConfigurationError as err:
            status.update(label="Configuration Error", state="error", expanded=True)
            st.error(f"Credentials Error: {err}")
        except AllKeysExhaustedError as err:
            status.update(label="Gemini Key Pool Exhausted", state="error", expanded=True)
            st.error(f"Key Pool Exhausted: {err}")
        except Exception as err:
            status.update(label="Pipeline Failure", state="error", expanded=True)
            st.error(f"Pipeline Execution Failed: {err}")

# ==========================================
# 4. Interactive Inspection Tabs & Results
# ==========================================

meeting_record: MeetingRecord | None = st.session_state.get("meeting_record")

if meeting_record:
    st.markdown("---")
    st.subheader("2. Meeting Intelligence & Deliverables")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Transcripts Review",
        "📋 Summary & Minutes",
        "⚖️ Key Decisions",
        "✅ Action Items",
    ])

    # ----------------------------------------------------
    # Tab 1: Transcripts Review & Speaker Diarization
    # ----------------------------------------------------
    with tab1:
        st.markdown("#### Transcripts Review & Acoustic Speaker Diarization")
        st.caption("Acoustic speaker turns, diarization badges, and uncertainty tags:")

        def format_diarized_transcript_html(raw_input: str) -> str:
            escaped = html.escape(raw_input)
            # Highlight uncertainty tokens
            formatted = re.sub(
                r"\[unclear:\s*(.*?)\?\]",
                r'<span class="unclear-tag">[unclear: \1?]</span>',
                escaped,
            )
            # Highlight speaker tags with timestamps: e.g. [Alex] (00:15):
            formatted = re.sub(
                r"\[([A-Za-z0-9 _\-]+)\]\s*\(([0-9]{1,2}:[0-9]{2})\):?",
                r'<span class="speaker-tag">[\1]</span><span class="timestamp-tag">(\2)</span>',
                formatted,
            )
            # Highlight speaker tags without timestamps: e.g. [Alex]:
            formatted = re.sub(
                r"\[([A-Za-z0-9 _\-]+)\]:",
                r'<span class="speaker-tag">[\1]</span>',
                formatted,
            )
            return formatted

        col_raw, col_refined = st.columns(2)

        with col_raw:
            st.markdown("**Raw Speech-to-Text Transcript (Stage 1 - Diarized)**")
            st.caption("Acoustic speaker turns and low-confidence tokens highlighted below:")
            highlighted_raw = format_diarized_transcript_html(meeting_record.raw_transcript)
            st.markdown(
                f'<div class="transcript-box">{highlighted_raw}</div>',
                unsafe_allow_html=True,
            )

        with col_refined:
            st.markdown("**Domain-Refined Transcript (Stage 2 - LLM 1)**")
            st.caption("Contextually proofread dialogue with speaker labels preserved:")
            highlighted_refined = format_diarized_transcript_html(meeting_record.refined_transcript)
            st.markdown(
                f'<div class="transcript-box">{highlighted_refined}</div>',
                unsafe_allow_html=True,
            )

    # ----------------------------------------------------
    # Tab 2: Summary & Minutes
    # ----------------------------------------------------
    with tab2:
        st.markdown("#### Executive Summary")
        st.markdown(
            f'<div class="summary-card">{html.escape(meeting_record.minutes.concise_summary)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Key Discussion Points")
        if meeting_record.minutes.discussion_points:
            for idx, pt in enumerate(meeting_record.minutes.discussion_points, start=1):
                st.markdown(f"**{idx}.** {pt}")
        else:
            st.info("No specific discussion points recorded.")

    # ----------------------------------------------------
    # Tab 3: Key Decisions
    # ----------------------------------------------------
    with tab3:
        st.markdown("#### Finalized Consensus & Binding Decisions")
        decisions = meeting_record.minutes.key_decisions

        if not decisions:
            st.info("No explicit decisions were reached in this meeting.")
        else:
            for idx, dec in enumerate(decisions, start=1):
                clean_title = html.escape(dec.decision_text)
                clean_quote = html.escape(dec.supporting_quote)
                st.markdown(
                    f"""
                    <div class="decision-card">
                        <div class="decision-card-title">{idx}. {clean_title}</div>
                        <div class="decision-card-quote">
                            <strong>Supporting Verbatim Quote:</strong> "{clean_quote}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ----------------------------------------------------
    # Tab 4: Action Items
    # ----------------------------------------------------
    with tab4:
        st.markdown("#### Deliverables & Task Ownership")
        actions = meeting_record.minutes.action_items

        if not actions:
            st.info("No action items were assigned.")
        else:
            # Build clean HTML table displaying 'Unspecified' with distinct neutral badge
            table_rows = []
            for item in actions:
                esc_task = html.escape(item.task_description)

                if item.owner.strip().lower() == "unspecified":
                    owner_html = '<span class="badge-unspecified">Unspecified</span>'
                else:
                    owner_html = f"<strong>{html.escape(item.owner)}</strong>"

                if item.deadline.strip().lower() == "unspecified":
                    deadline_html = '<span class="badge-unspecified">Unspecified</span>'
                else:
                    deadline_html = f"<code>{html.escape(item.deadline)}</code>"

                table_rows.append(
                    f"<tr>"
                    f"<td>{esc_task}</td>"
                    f"<td>{owner_html}</td>"
                    f"<td>{deadline_html}</td>"
                    f"</tr>"
                )

            table_html = f"""
            <table class="action-table">
                <thead>
                    <tr>
                        <th>Task Description</th>
                        <th style="width: 22%;">Owner</th>
                        <th style="width: 22%;">Deadline</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

    # ==========================================
    # 5. Dual Download Section
    # ==========================================
    st.markdown("---")
    st.subheader("3. Export Verified Deliverables")

    json_payload = export_to_json(meeting_record)
    markdown_payload = export_to_markdown(meeting_record)

    col_btn_json, col_btn_md = st.columns(2)

    with col_btn_json:
        st.download_button(
            label="💾 Download Machine-Readable JSON",
            data=json_payload,
            file_name="meeting_record.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_btn_md:
        st.download_button(
            label="📄 Download Human-Readable Markdown",
            data=markdown_payload,
            file_name="meeting_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )
