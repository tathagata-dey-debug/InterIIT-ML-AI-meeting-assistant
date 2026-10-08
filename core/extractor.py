"""
Stage 3: Language Model 2 - Documentation & Structured Extraction.
Implements MeetingDocumenter extracting structured MeetingMinutes with strict anti-hallucination rules.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from .config import Config, ConfigurationError
from .key_manager import GeminiKeyPool
from .schemas import MeetingMinutes, RefinedTranscript

logger = logging.getLogger(__name__)

DOCUMENTER_SYSTEM_PROMPT = """You are a meticulous, objective executive secretary. Your task is to extract structured meeting minutes, key decisions, and action items from the provided refined transcript.

STRICT ANTI-HALLUCINATION RULES:
1. Grounding: All extracted content must be 100% grounded in the text.
2. Decisions vs. Proposals: A proposal, suggestion, idea, or ongoing debate is NOT a decision. Extract a decision ONLY if participants explicitly agreed or confirmed it. If no explicit decisions were reached, return an empty list.
3. Action Items:
   - Extract actionable, concrete tasks.
   - SPEAKER ATTRIBUTION: Use the speaker tags in the refined transcript to identify who committed to an action item. If a specific speaker explicitly volunteers or is directly assigned, record their name as 'owner'. If no explicit owner is named or assigned, keep 'owner' strictly as 'Unspecified'. DO NOT GUESS OR INFER.
   - 'deadline': Extract the deadline ONLY if a specific timeline or date was mentioned. If no deadline was stated, set deadline strictly to 'Unspecified'. DO NOT GUESS.
   - If no tasks were assigned, return an empty list.
4. Summary & Points: Provide a clean, comprehensive executive summary and a list of key discussion points."""


class MeetingDocumenter:
    """Language Model 2: Populates MeetingMinutes schema via instructor or native structured outputs."""

    def __init__(self, config: Config):
        self.config = config

    def extract(
        self,
        refined_transcript: Union[str, RefinedTranscript],
        meeting_context: Optional[str] = None,
    ) -> MeetingMinutes:
        """Alias for document() to support both calling conventions."""
        return self.document(refined_transcript, meeting_context=meeting_context)

    def document(
        self,
        refined_transcript: Union[str, RefinedTranscript],
        meeting_context: Optional[str] = None,
    ) -> MeetingMinutes:
        """
        Takes refined_transcript: str as input and extracts structured MeetingMinutes.

        Args:
            refined_transcript: Refined transcript string.
            meeting_context: Optional additional context.

        Returns:
            Validated MeetingMinutes Pydantic v2 model.
        """
        text = (
            refined_transcript.cleaned_text
            if hasattr(refined_transcript, "cleaned_text")
            else str(refined_transcript)
        )

        user_content = f"Refined Transcript:\n\n{text}"
        if meeting_context:
            user_content = f"Context:\n{meeting_context}\n\n" + user_content

        if self.config.llm_provider == "openai":
            return self._document_openai(user_content)
        elif self.config.llm_provider == "gemini":
            return self._document_gemini(user_content)
        else:
            raise ConfigurationError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def _document_openai(self, user_content: str) -> MeetingMinutes:
        """Extract structured minutes using instructor and OpenAI."""
        try:
            import instructor
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "openai and instructor packages are required. Install via requirements.txt"
            ) from err

        if not self.config.openai_api_key:
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )

        client = instructor.from_openai(OpenAI(api_key=self.config.openai_api_key))
        model = self.config.llm_model if "gpt" in self.config.llm_model else "gpt-4o-mini"

        response: MeetingMinutes = client.chat.completions.create(
            model=model,
            response_model=MeetingMinutes,
            temperature=self.config.llm_temperature,
            messages=[
                {"role": "system", "content": DOCUMENTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response

    def _document_gemini(self, user_content: str) -> MeetingMinutes:
        """Extract structured minutes using Google Gemini SDK with dynamic key pool failover."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as err:
            raise ImportError("google-genai is required. Install via requirements.txt") from err

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

        def _do_extract(api_key: str) -> MeetingMinutes:
            client = genai.Client(api_key=api_key)
            model = self.config.llm_model if "gemini" in self.config.llm_model else "gemini-3.5-flash-lite"
            if "1.5" in model or "2.5" in model:
                model = "gemini-3.5-flash-lite"

            response = client.models.generate_content(
                model=model,
                contents=[DOCUMENTER_SYSTEM_PROMPT, user_content],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MeetingMinutes,
                    temperature=self.config.llm_temperature,
                ),
            )
            return MeetingMinutes.model_validate_json(response.text)

        return key_pool.execute_with_fallback(_do_extract)


# Backwards compatibility alias
MinutesExtractor = MeetingDocumenter
