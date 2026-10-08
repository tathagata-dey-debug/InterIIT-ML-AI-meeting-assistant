"""
Stage 2: Language Model 1 - Transcript Refinement.
Implements TranscriptRefiner correcting domain terminology, acronyms, and speech recognition errors.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from .config import Config, ConfigurationError
from .schemas import RawTranscript
from .key_manager import AllKeysExhaustedError, GeminiKeyPool

logger = logging.getLogger(__name__)

REFINER_SYSTEM_PROMPT = """You are an expert audio transcript proofreader and domain terminology specialist.
Your sole job is to review a raw, machine-generated transcript and correct plausible speech recognition errors.

OPERATING RULES:
1. Contextual Terminology: Identify the technical, industry, or domain context of the meeting (e.g., Software Engineering, Data Science, Finance, Medical, Operations). Use this domain context to correct misheard jargon, technical acronyms, and specialized nomenclature.
2. Address Uncertainty: Pay special attention to tokens marked with '[unclear: word?]'. If context makes the intended term evident, replace it with the corrected term. If context cannot determine the intended word, retain the original word and remove the tag brackets.
3. PRESERVE SPEAKER LABELS: You must preserve all speaker tags (e.g., [Speaker 1]:, [Alex]:) and timestamps exactly as formatted at the beginning of each line. Never merge distinct speaker turns or remove speaker identifiers.
4. INVARIANTS - DO NOT ALTER:
   - Proper names of people, companies, or products.
   - Numbers, financial figures, dates, and deadlines.
   - Negations (e.g., 'not', 'cannot', 'did not', 'won't'). Reversing a negation is a catastrophic failure.
   - Commitments and speaker intent.
5. SEMANTIC SPEAKER RESOLUTION & TURN SPLITTING:
   - Check speaker labels in the raw transcript. If all turns are collapsed under '[Speaker 1]', or if multiple dialogue exchanges are merged into a single paragraph:
     a) Analyze conversational context, speaker mentions, and dialogue cues to identify distinct participants (e.g., meeting lead, team members like Priya, Dev).
     b) Split merged turns into separate lines with correct speaker tags. For example, if one sentence assigns a task ("Dev, please...") and the next sentence accepts it ("I will deliver..."), split them into distinct speaker turns ([Alex] and [Dev]).
     c) Replace generic tags ([Speaker 1], [Speaker 2]) with confirmed names ([Alex], [Priya], [Dev]) when supported by conversational evidence.
   - Preserve timestamps in format (MM:SS) where available.
6. Output ONLY the refined transcript text with zero introductory remarks, explanations, or conversational filler."""


class TranscriptRefiner:
    """Language Model 1: Reviews raw machine-generated transcript and corrects speech recognition errors."""

    def __init__(self, config: Config):
        self.config = config

    def refine(
        self,
        raw_transcript: Union[str, RawTranscript],
        domain_context: Optional[str] = None,
    ) -> str:
        """
        Takes raw_transcript: str as input and outputs the domain-refined transcript text.

        Args:
            raw_transcript: Raw transcript text string or RawTranscript object.
            domain_context: Optional domain terminology hints or glossary.

        Returns:
            Refined transcript text with zero conversational filler.
        """
        text_to_refine = (
            raw_transcript.full_text
            if isinstance(raw_transcript, RawTranscript)
            else str(raw_transcript)
        )

        user_content = text_to_refine
        if domain_context:
            user_content = f"Domain Context / Key Terminology Glossary:\n{domain_context}\n\nTranscript:\n{text_to_refine}"

        if self.config.llm_provider == "openai":
            return self._refine_openai(user_content)
        elif self.config.llm_provider == "gemini":
            return self._refine_gemini(user_content)
        else:
            raise ConfigurationError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def _refine_openai(self, user_content: str) -> str:
        """Refine using OpenAI chat completion."""
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("openai package is required. Install via requirements.txt") from err

        if not self.config.openai_api_key:
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )

        client = OpenAI(api_key=self.config.openai_api_key)
        model = self.config.llm_model if "gpt" in self.config.llm_model else "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model,
            temperature=self.config.llm_temperature,
            messages=[
                {"role": "system", "content": REFINER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )

        output = response.choices[0].message.content or ""
        return output.strip()

    def _refine_gemini(self, user_content: str) -> str:
        """Refine using Google Gemini SDK with dynamic key pool failover."""
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

        def _do_refine(api_key: str) -> str:
            client = genai.Client(api_key=api_key)
            model = self.config.llm_model if "gemini" in self.config.llm_model else "gemini-3.5-flash-lite"
            if "1.5" in model or "2.5" in model:
                model = "gemini-3.5-flash-lite"

            response = client.models.generate_content(
                model=model,
                contents=[REFINER_SYSTEM_PROMPT, user_content],
                config=types.GenerateContentConfig(
                    temperature=self.config.llm_temperature,
                ),
            )
            output = response.text or ""
            return output.strip()

        return key_pool.execute_with_fallback(_do_refine)
