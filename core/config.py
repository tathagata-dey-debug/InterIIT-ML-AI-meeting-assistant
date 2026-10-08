"""
Typed configuration and credential loader for AI Meeting Assistant.
Utilizes Pydantic v2 schemas and enforces multi-provider validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .key_manager import GeminiKeyPool

try:
    from dotenv import load_dotenv

    # Proactively search and load .env from project root if it exists
    load_dotenv(override=False)
except ImportError:
    pass


class ConfigurationError(Exception):
    """Raised when configuration validation or API credential loading fails."""

    pass


class Config(BaseModel):
    """Centralized configuration for audio processing, STT, and LLM stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __getattr__(self, item: str) -> Any:
        if item == "key_pool":
            pool_keys = list(getattr(self, "gemini_api_keys", []))
            single_key = getattr(self, "gemini_api_key", None)
            if single_key and single_key not in pool_keys:
                pool_keys.insert(0, single_key)
            pool = GeminiKeyPool(pool_keys) if pool_keys else None
            try:
                object.__setattr__(self, "key_pool", pool)
            except Exception:
                pass
            return pool
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")

    # Provider Choices
    stt_provider: Literal["openai", "gemini", "groq"] = Field(
        default="openai",
        description="Speech-to-text cloud provider."
    )
    llm_provider: Literal["gemini", "openai"] = Field(
        default="gemini",
        description="LLM provider for refinement and extraction stages."
    )

    # API Keys
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI models (Whisper / GPT)."
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Primary or fallback API key for Google Gemini models."
    )
    gemini_api_keys: List[str] = Field(
        default_factory=list,
        description="Pool of Gemini API keys for dynamic failover on rate-limiting / quota limits."
    )
    key_pool: Optional[GeminiKeyPool] = Field(
        default=None,
        exclude=True,
        description="Active GeminiKeyPool instance managing key failover and rotation."
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="API key for Groq Whisper models."
    )

    # Model Parameters
    stt_model: str = Field(
        default="whisper-1",
        description="Speech-to-text model identifier."
    )
    llm_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="LLM model identifier for refiner and extractor."
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for deterministic extraction."
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum output token budget for LLM stages."
    )

    # Audio Conditioning Defaults
    target_dbfs: float = Field(
        default=-20.0,
        description="Target loudness level in dBFS for normalization."
    )
    chunk_length_seconds: int = Field(
        default=600,
        gt=0,
        description="Max segment duration in seconds for audio chunking (10 mins)."
    )

    # Paths
    keys_file_path: Path = Field(
        default=Path("keys/api_keys.json"),
        description="Path to local credentials JSON file."
    )
    samples_dir: Path = Field(
        default=Path("samples"),
        description="Directory for audio input recordings."
    )
    export_dir: Path = Field(
        default=Path("exports"),
        description="Directory for generated JSON and Markdown outputs."
    )

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Config":
        """
        Validate that the chosen STT and LLM providers have valid, non-placeholder API keys.
        """
        placeholder_tokens = {
            "sk-your-openai-key-here",
            "aiza-your-gemini-key-here",
            "your-openai-api-key-here",
            "your-gemini-api-key-here",
            "",
        }

        def is_valid_key(key: Optional[str]) -> bool:
            if not key or not isinstance(key, str):
                return False
            clean = key.strip().lower()
            return clean not in placeholder_tokens and len(clean) > 8

        # Collect and sync Gemini key pool
        pool_keys: List[str] = []
        if isinstance(self.gemini_api_keys, list):
            for k in self.gemini_api_keys:
                if is_valid_key(k) and k.strip() not in pool_keys:
                    pool_keys.append(k.strip())

        if is_valid_key(self.gemini_api_key):
            clean_single = self.gemini_api_key.strip()
            if clean_single not in pool_keys:
                pool_keys.insert(0, clean_single)

        if pool_keys and not self.gemini_api_key:
            self.gemini_api_key = pool_keys[0]

        self.gemini_api_keys = pool_keys

        if self.key_pool is None:
            self.key_pool = GeminiKeyPool(pool_keys)
        else:
            for k in pool_keys:
                self.key_pool.add_key(k)

        # Check STT provider requirement
        if self.stt_provider == "openai" and not is_valid_key(self.openai_api_key):
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )
        elif self.stt_provider == "gemini" and not (is_valid_key(self.gemini_api_key) or pool_keys):
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )
        elif self.stt_provider == "groq" and not is_valid_key(self.groq_api_key):
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )

        # Check LLM provider requirement
        if self.llm_provider == "openai" and not is_valid_key(self.openai_api_key):
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )
        elif self.llm_provider == "gemini" and not (is_valid_key(self.gemini_api_key) or pool_keys):
            raise ConfigurationError(
                "Missing API credentials. Please populate keys/api_keys.json with your API keys."
            )

        return self

    @classmethod
    def load(
        cls,
        keys_path: Optional[Path | str] = None,
        base_dir: Optional[Path | str] = None,
        **overrides: Any,
    ) -> "Config":
        """
        Load configuration hierarchically:
        1. Explicit overrides
        2. `keys/api_keys.json` (if exists)
        3. `.env` file or environment variables
        4. Class defaults
        """
        root = Path(base_dir).resolve() if base_dir else Path.cwd()
        resolved_keys_path = (
            Path(keys_path)
            if keys_path
            else root / "keys" / "api_keys.json"
        )

        data: Dict[str, Any] = {}

        # 1. Attempt reading keys/api_keys.json
        if resolved_keys_path.exists():
            try:
                with open(resolved_keys_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    # Normalize keys from UPPERCASE (JSON style) to lowercase
                    for k, v in raw_data.items():
                        norm_key = k.lower()
                        data[norm_key] = v
            except json.JSONDecodeError as err:
                raise ConfigurationError(
                    f"Invalid JSON format in {resolved_keys_path}: {err}"
                ) from err

        # Parse GEMINI_API_KEYS (list or string)
        gemini_keys_list: List[str] = []
        raw_keys = data.get("gemini_api_keys")
        if isinstance(raw_keys, list):
            gemini_keys_list.extend(raw_keys)
        elif isinstance(raw_keys, str):
            try:
                parsed = json.loads(raw_keys)
                if isinstance(parsed, list):
                    gemini_keys_list.extend(parsed)
                else:
                    gemini_keys_list.append(raw_keys)
            except Exception:
                gemini_keys_list.extend([k.strip() for k in raw_keys.split(",") if k.strip()])

        # 2. Environment variables fallback
        env_openai = os.getenv("OPENAI_API_KEY")
        if env_openai and not data.get("openai_api_key"):
            data["openai_api_key"] = env_openai

        env_gemini = os.getenv("GEMINI_API_KEY")
        if env_gemini and not data.get("gemini_api_key"):
            data["gemini_api_key"] = env_gemini

        env_gemini_keys = os.getenv("GEMINI_API_KEYS")
        if env_gemini_keys:
            try:
                parsed = json.loads(env_gemini_keys)
                if isinstance(parsed, list):
                    gemini_keys_list.extend(parsed)
                else:
                    gemini_keys_list.append(env_gemini_keys)
            except Exception:
                gemini_keys_list.extend([k.strip() for k in env_gemini_keys.split(",") if k.strip()])

        if gemini_keys_list:
            data["gemini_api_keys"] = gemini_keys_list

        env_groq = os.getenv("GROQ_API_KEY")
        if env_groq and not data.get("groq_api_key"):
            data["groq_api_key"] = env_groq

        if "STT_PROVIDER" in os.environ and "stt_provider" not in data:
            data["stt_provider"] = os.environ["STT_PROVIDER"].lower()

        if "LLM_PROVIDER" in os.environ and "llm_provider" not in data:
            data["llm_provider"] = os.environ["LLM_PROVIDER"].lower()

        # Adjust default STT model according to provider
        if "stt_model" not in data:
            if data.get("stt_provider") == "groq":
                data["stt_model"] = "whisper-large-v3"
            elif data.get("stt_provider") == "gemini":
                data["stt_model"] = "gemini-3.5-flash-lite"
            else:
                data["stt_model"] = "whisper-1"

        # Adjust default LLM model according to provider if not explicitly passed
        if "llm_model" not in data:
            chosen_llm_provider = data.get("llm_provider", "gemini")
            if chosen_llm_provider == "openai":
                data["llm_model"] = "gpt-4o-mini"
            else:
                data["llm_model"] = "gemini-3.5-flash-lite"

        # Normalize deprecated Gemini model identifiers to active gemini-3.5-flash-lite
        deprecated_gemini_models = {
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        }
        if data.get("stt_model") in deprecated_gemini_models:
            data["stt_model"] = "gemini-3.5-flash-lite"
        if data.get("llm_model") in deprecated_gemini_models:
            data["llm_model"] = "gemini-3.5-flash-lite"

        # 3. Apply overrides
        data.update(overrides)
        data["keys_file_path"] = resolved_keys_path

        return cls(**data)
