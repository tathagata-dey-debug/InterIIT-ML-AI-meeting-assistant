"""
Pydantic v2 schemas with anti-hallucination defaults and strict typing.
Defines ActionItem, KeyDecision, MeetingMinutes, and MeetingRecord.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. Speech-to-Text & Uncertainty Tagging
# ==========================================

class TranscriptWord(BaseModel):
    """Word-level speech timestamp and confidence evaluation."""
    word: str = Field(description="Transcribed word token.")
    start: float = Field(ge=0.0, description="Start timestamp in seconds.")
    end: float = Field(ge=0.0, description="End timestamp in seconds.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from STT engine."
    )
    is_uncertain: bool = Field(
        default=False,
        description="Flagged true when confidence falls below the uncertainty threshold."
    )


class TranscriptSegment(BaseModel):
    """Chunk or utterance-level transcript block."""
    id: int = Field(description="Sequential segment index.")
    speaker: Optional[str] = Field(default=None, description="Speaker identifier if diarized.")
    start: float = Field(ge=0.0, description="Start offset in seconds.")
    end: float = Field(ge=0.0, description="End offset in seconds.")
    text: str = Field(description="Segment text.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score across segment tokens."
    )
    words: List[TranscriptWord] = Field(
        default_factory=list,
        description="Word-level token breakdown with uncertainties."
    )


class RawTranscript(BaseModel):
    """Full raw transcription output from Stage 1."""
    audio_filename: str = Field(description="Source audio file name.")
    duration_seconds: float = Field(ge=0.0, description="Total audio duration in seconds.")
    segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="Chronological transcript segments."
    )
    full_text: str = Field(description="Joined plain transcript.")
    uncertain_token_count: int = Field(
        default=0,
        ge=0,
        description="Count of tokens flagged with low confidence."
    )


class TerminologyCorrection(BaseModel):
    """Specific acronym, domain term, or technical correction made by Refiner."""
    original: str = Field(description="Phonetically misrecognized term in raw audio.")
    corrected: str = Field(description="Domain-corrected technical term or entity.")
    context: str = Field(description="Surrounding context sentence.")


class RefinedTranscript(BaseModel):
    """Stage 2 normalized and terminology-corrected transcript."""
    cleaned_text: str = Field(description="Grammatically smoothed and terminology-refined transcript.")
    corrections_applied: List[TerminologyCorrection] = Field(
        default_factory=list,
        description="List of domain-specific phonetic corrections applied."
    )
    speakers_identified: List[str] = Field(
        default_factory=list,
        description="Distinct speaker tags identified or normalized."
    )


# ==========================================
# 2. Phase 3 LLM Schemas
# ==========================================

class ActionItem(BaseModel):
    task_description: str = Field(
        description="Clear, actionable description of the task to be performed."
    )
    owner: str = Field(
        default="Unspecified",
        description="The exact individual or role explicitly assigned to this task. If not stated in the recording, MUST strictly be 'Unspecified'."
    )
    deadline: str = Field(
        default="Unspecified",
        description="The explicit due date, time, or timeframe mentioned. If not stated in the recording, MUST strictly be 'Unspecified'."
    )

    # Backwards-compatibility properties
    @property
    def task(self) -> str:
        return self.task_description

    @property
    def assignee(self) -> str:
        return self.owner

    @property
    def evidence_quote(self) -> str:
        return f"Task assigned to {self.owner}: {self.task_description}"


class KeyDecision(BaseModel):
    decision_text: str = Field(
        description="Explicit decision agreed upon by meeting participants. Never include unagreed proposals or open debates."
    )
    supporting_quote: str = Field(
        description="Verbatim snippet from the transcript proving that consensus was finalized."
    )

    # Backwards-compatibility properties
    @property
    def decision(self) -> str:
        return self.decision_text

    @property
    def evidence_quote(self) -> str:
        return self.supporting_quote


# Decision alias for backward compatibility
Decision = KeyDecision


class MeetingMinutes(BaseModel):
    concise_summary: str = Field(
        description="High-level executive summary of the meeting's primary objectives and outcomes."
    )
    discussion_points: List[str] = Field(
        description="Chronological or thematic key topics discussed during the meeting."
    )
    key_decisions: List[KeyDecision] = Field(
        default_factory=list,
        description="List of finalized decisions. Must be an empty list if no decisions were agreed upon."
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="List of actionable tasks. Must be an empty list if no tasks were assigned."
    )

    # Backward-compatibility property
    @property
    def decisions(self) -> List[KeyDecision]:
        return self.key_decisions


class MeetingRecord(BaseModel):
    raw_transcript: str = Field(
        description="Raw verbatim transcript from Stage 1."
    )
    refined_transcript: str = Field(
        description="Domain-refined transcript from Language Model 1."
    )
    minutes: MeetingMinutes = Field(
        description="Structured minutes extracted by Language Model 2."
    )
