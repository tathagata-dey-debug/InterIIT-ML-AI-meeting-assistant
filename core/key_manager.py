"""
Dynamic Gemini API Key Pooling with automatic failover and rotation.
Handles token limits, quota limits, and HTTP 429 / ResourceExhausted errors transparently.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AllKeysExhaustedError(Exception):
    """Raised when all Gemini API keys in the pool have exceeded their quota or token limits."""
    pass


def is_exhaustion_error(exc: Exception) -> bool:
    """
    Determine if an exception represents rate limiting, quota exhaustion, or 429.
    """
    # 1. Inspect HTTP status codes
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in (429, 403):
        return True

    # 2. Inspect exception class name
    cls_name = exc.__class__.__name__.lower()
    exhaustion_classes = (
        "ratelimit",
        "resourceexhausted",
        "quotaexceeded",
        "toomanyrequests",
    )
    if any(ec in cls_name for ec in exhaustion_classes):
        return True

    # 3. Inspect string representation and details
    msg = str(exc).lower()
    exhaustion_keywords = (
        "429",
        "quota",
        "resource_exhausted",
        "resourceexhausted",
        "rate limit",
        "rate_limit",
        "exhausted",
        "tokens per minute",
        "requests per minute",
        "too many requests",
    )
    return any(keyword in msg for keyword in exhaustion_keywords)


def mask_api_key(key: str) -> str:
    """Mask an API key for safe UI display (e.g. AIzaSy...4Q9X)."""
    if not key or not isinstance(key, str):
        return "None"
    clean = key.strip()
    if len(clean) > 10:
        return f"{clean[:6]}...{clean[-4:]}"
    elif len(clean) > 4:
        return f"{clean[:3]}...*"
    return "***"


class GeminiKeyPool:
    """
    Centralized pool managing multiple Gemini API keys with automatic failover
    and rotation on rate-limiting or quota exhaustion.
    """

    def __init__(self, keys: Optional[List[str]] = None) -> None:
        self._keys: List[str] = []
        self._exhausted_keys: Set[str] = set()
        self._current_index: int = 0

        if keys:
            for k in keys:
                self.add_key(k)

    def add_key(self, key: str) -> bool:
        """
        Append a non-empty key to the pool (stripping whitespace and deduplicating).

        Returns:
            True if the key was added, False if duplicate or invalid.
        """
        if not key or not isinstance(key, str):
            return False
        clean = key.strip()
        if not clean or len(clean) < 8 or clean in self._keys:
            return False

        self._keys.append(clean)
        logger.info("Added key %s to GeminiKeyPool (total keys: %d)", mask_api_key(clean), len(self._keys))
        return True

    def remove_key(self, key: str) -> bool:
        """Remove a key from the pool."""
        clean = key.strip() if key else ""
        if clean in self._keys:
            self._keys.remove(clean)
            self._exhausted_keys.discard(clean)
            if self._current_index >= len(self._keys) and self._keys:
                self._current_index = 0
            return True
        return False

    def clear(self) -> None:
        """Clear all keys from the pool."""
        self._keys.clear()
        self._exhausted_keys.clear()
        self._current_index = 0

    def reset_exhaustion(self) -> None:
        """Reset the exhausted state of all keys in the pool."""
        self._exhausted_keys.clear()
        self._current_index = 0
        logger.info("Reset exhaustion status for all keys in GeminiKeyPool.")

    @property
    def total_count(self) -> int:
        """Total number of keys in the pool."""
        return len(self._keys)

    @property
    def available_count(self) -> int:
        """Number of non-exhausted keys in the pool."""
        return len([k for k in self._keys if k not in self._exhausted_keys])

    def has_unexhausted_keys(self) -> bool:
        """Check if any unexhausted key remains in the pool."""
        return self.available_count > 0

    def mark_exhausted(self, key: Optional[str] = None) -> None:
        """Mark a specific key (or current active key) as exhausted."""
        target = key or self.get_active_key_or_none()
        if target:
            self._exhausted_keys.add(target)
            logger.warning("Marked key %s as exhausted in GeminiKeyPool.", mask_api_key(target))

    def get_active_key_or_none(self) -> Optional[str]:
        """Return the active key without raising an exception if none are available."""
        if not self._keys:
            return None
        # Start searching from _current_index
        n = len(self._keys)
        for offset in range(n):
            idx = (self._current_index + offset) % n
            candidate = self._keys[idx]
            if candidate not in self._exhausted_keys:
                self._current_index = idx
                return candidate
        return None

    def get_active_key(self) -> str:
        """
        Return the currently active Gemini API key.

        Raises:
            AllKeysExhaustedError: If no valid keys exist or all keys have been exhausted.
        """
        active = self.get_active_key_or_none()
        if not active:
            raise AllKeysExhaustedError(
                "All Gemini API keys in the pool have exceeded their quota or token limits. Please add a valid key."
            )
        return active

    def rotate_key(self) -> str:
        """
        Advance to the next unexhausted key in the pool.

        Raises:
            AllKeysExhaustedError: If all keys in the pool have been exhausted.
        """
        if not self._keys:
            raise AllKeysExhaustedError(
                "All Gemini API keys in the pool have exceeded their quota or token limits. Please add a valid key."
            )

        n = len(self._keys)
        # Advance at least once
        for offset in range(1, n + 1):
            idx = (self._current_index + offset) % n
            candidate = self._keys[idx]
            if candidate not in self._exhausted_keys:
                self._current_index = idx
                logger.info("Rotated to next key %s in GeminiKeyPool.", mask_api_key(candidate))
                return candidate

        raise AllKeysExhaustedError(
            "All Gemini API keys in the pool have exceeded their quota or token limits. Please add a valid key."
        )

    def get_masked_keys(self) -> List[Dict[str, Any]]:
        """
        Return a list of keys masked for UI display.
        Example: [{'masked': 'AIzaSy...4Q9X', 'is_active': True, 'exhausted': False}]
        """
        active_key = self.get_active_key_or_none()
        results: List[Dict[str, Any]] = []

        for k in self._keys:
            results.append({
                "masked": mask_api_key(k),
                "is_active": (k == active_key),
                "exhausted": (k in self._exhausted_keys),
            })
        return results

    def execute_with_fallback(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Executes the target function with automatic failover across keys in the pool.
        The function `func` is called as `func(api_key, *args, **kwargs)` where `api_key`
        is the current active key.

        If a call raises an exception indicating rate limits or token exhaustion
        (429, ResourceExhausted, QuotaExceeded), marks current key as exhausted,
        cycles to the next key, and retries. Non-quota errors re-raise immediately.
        """
        if not self._keys:
            raise AllKeysExhaustedError(
                "All Gemini API keys in the pool have exceeded their quota or token limits. Please add a valid key."
            )

        max_attempts = len(self._keys)
        last_exception: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                current_key = self.get_active_key()
            except AllKeysExhaustedError as err:
                if last_exception is not None:
                    raise AllKeysExhaustedError(
                        f"All Gemini API keys in the pool have exceeded their quota or token limits. Last error: {last_exception}"
                    ) from last_exception
                raise err

            try:
                return func(current_key, *args, **kwargs)
            except Exception as exc:
                if is_exhaustion_error(exc):
                    last_exception = exc
                    logger.warning(
                        "Quota / Rate limit reached on key %s (attempt %d/%d). Details: %s",
                        mask_api_key(current_key),
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    self.mark_exhausted(current_key)
                    # If any keys remain, rotate and retry
                    if self.has_unexhausted_keys():
                        try:
                            self.rotate_key()
                            continue
                        except AllKeysExhaustedError:
                            pass
                    raise AllKeysExhaustedError(
                        "All Gemini API keys in the pool have exceeded their quota or token limits. Please add a valid key."
                    ) from exc
                else:
                    # Non-quota error (syntax, network failure, invalid model) -> re-raise immediately
                    raise exc

        raise AllKeysExhaustedError(
            "All Gemini API keys in the pool have exceeded their quota or token limits. Please add a valid key."
        ) from last_exception
