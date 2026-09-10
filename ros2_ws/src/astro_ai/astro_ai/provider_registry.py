#!/usr/bin/env python3
"""ASTRO V1 — Dynamic Provider & Model Capability Registry (Production Grade).

Features:
  - Strict Runtime REST discovery of active models for Groq & Gemini (No blind seed fallbacks)
  - Clear separation: DISCOVERED vs. ROUTEABLE vs. REJECTED vs. BLACKLISTED
  - Capability filtering (generateContent, chat, streaming, vision, tool calling)
  - Clear separation of Provider Health vs. Model Availability
  - Strict 8-class error classification
  - Immediate blacklisting on 400/404/unsupported with zero retry storm
  - Dynamic fallback model routing with cooldowns
  - CLI discovery command: `python3 provider_registry.py --discover`
"""

import argparse
import logging

_LOG = logging.getLogger(__name__)

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Set, Tuple


try:
    from astro_ai.circuit_breaker import (
        GlobalProviderCircuitBreaker,
        ProviderState,
        RequestErrorClass,
        get_global_circuit_breaker,
    )
except ImportError:
    from circuit_breaker import (
        GlobalProviderCircuitBreaker,
        ProviderState,
        RequestErrorClass,
        get_global_circuit_breaker,
    )


class ErrorClass(str, Enum):
    NONE = "none"
    QUOTA_EXHAUSTED = "quota_exhausted"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    MODEL_NOT_FOUND = "model_not_found"
    UNSUPPORTED_MODEL = "unsupported_model"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"


class ProviderHealth(str, Enum):
    UNINITIALIZED = "uninitialized"
    HEALTHY = "healthy"
    DISCOVERY_UNAVAILABLE = "discovery_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_FAILED = "authentication_failed"
    DISABLED = "disabled"


class ProviderError(Exception):
    """Exception raised when an LLM provider request fails."""

    def __init__(self, provider: str, model_id: str, error_class: ErrorClass, message: str, status_code: int = 0):
        super().__init__(f"[{provider}:{model_id}] ({error_class.value}): {message}")
        self.provider = provider
        self.model_id = model_id
        self.error_class = error_class
        self.message = message
        self.status_code = status_code


@dataclass
class ModelCapability:
    provider: str
    model_id: str
    chat_supported: bool = True
    streaming_supported: bool = True
    tool_calling_supported: bool = False
    vision_supported: bool = False
    realtime_audio_supported: bool = False
    available: bool = True
    is_blacklisted: bool = False
    cooldown_until: float = 0.0
    consecutive_errors: int = 0
    last_error: str = ""
    last_error_class: ErrorClass = ErrorClass.NONE
    last_latency_ms: float = 0.0


# Approved Production Chat & Vision LLM Models (Strict Whitelist)
GROQ_PRODUCTION_MODELS: Set[str] = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
}

GROQ_PREFERENCE_ORDER: List[str] = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

GEMINI_PRODUCTION_MODELS: Set[str] = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-lite",
}

GEMINI_PREFERENCE_ORDER: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
]



OPENAI_PRODUCTION_MODELS: Set[str] = {
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-realtime-2.1-mini",
    "gpt-realtime-2.1",
    "gpt-realtime-2",
    "gpt-realtime",
    "gpt-4o-realtime-preview",
}


class ProviderRegistry:
    """Registry maintaining real-time model capabilities, availability, and error states."""

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self.circuit_breaker = get_global_circuit_breaker()
        self._models: Dict[str, ModelCapability] = {}
        self._provider_health: Dict[str, ProviderHealth] = {
            "groq": ProviderHealth.UNINITIALIZED,
            "gemini": ProviderHealth.UNINITIALIZED,
            "openai": ProviderHealth.HEALTHY,
            "local": ProviderHealth.HEALTHY,
        }
        self._discovered_raw: Dict[str, List[str]] = {
            "groq": [],
            "gemini": [],
            "openai": list(OPENAI_PRODUCTION_MODELS),
        }
        self._routeable_models: Dict[str, List[str]] = {
            "groq": list(GROQ_PREFERENCE_ORDER),
            "gemini": list(GEMINI_PREFERENCE_ORDER),
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-realtime-2.1-mini", "gpt-realtime-2.1", "gpt-realtime"],
        }
        self._rejected_models: Dict[str, Dict[str, str]] = {
            "groq": {},
            "gemini": {},
            "openai": {},
        }

        # Initialize base model capabilities
        self._register_default_models()

    def _register_default_models(self):
        """Pre-populates verified capability records for production models."""
        # Groq
        for m in GROQ_PRODUCTION_MODELS:
            is_vis = "vision" in m.lower()
            self.register_model(
                ModelCapability(
                    provider="groq",
                    model_id=m,
                    chat_supported=True,
                    streaming_supported=True,
                    tool_calling_supported="llama" in m.lower(),
                    vision_supported=is_vis,
                    realtime_audio_supported=False,
                )
            )
        # Gemini
        for m in GEMINI_PRODUCTION_MODELS:
            self.register_model(
                ModelCapability(
                    provider="gemini",
                    model_id=m,
                    chat_supported=True,
                    streaming_supported=True,
                    tool_calling_supported=True,
                    vision_supported=True,
                    realtime_audio_supported=False,
                )
            )
        # OpenAI
        for m in OPENAI_PRODUCTION_MODELS:
            is_rt = "realtime" in m.lower()
            self.register_model(
                ModelCapability(
                    provider="openai",
                    model_id=m,
                    chat_supported=True,
                    streaming_supported=True,
                    tool_calling_supported=True,
                    vision_supported=True,
                    realtime_audio_supported=is_rt,
                )
            )

    def _log(self, level: str, msg: str) -> None:
        if not self.logger:
            return
        lvl = str(level).lower()
        if lvl == "debug":
            fn = getattr(self.logger, "debug", getattr(self.logger, "info", print))
        elif lvl in ("warn", "warning"):
            fn = getattr(self.logger, "warn", getattr(self.logger, "warning", print))
        elif lvl == "error":
            fn = getattr(self.logger, "error", print)
        else:
            fn = getattr(self.logger, "info", print)
        try:
            fn(str(msg))
        except Exception:
            pass

    def get_provider_health(self, provider: str) -> ProviderHealth:
        return self._provider_health.get(provider, ProviderHealth.UNINITIALIZED)

    def set_provider_health(self, provider: str, health: ProviderHealth) -> None:
        self._provider_health[provider] = health

    def register_model(self, model: ModelCapability) -> None:
        key = f"{model.provider}:{model.model_id}"
        self._models[key] = model

    def get_model(self, provider: str, model_id: str) -> Optional[ModelCapability]:
        return self._models.get(f"{provider}:{model_id}")

    def get_all_models(self) -> List[ModelCapability]:
        return list(self._models.values())

    def is_routeable(self, provider: str, model_id: str) -> bool:
        """Returns True if the provider is healthy and the model is discovered, routeable, not blacklisted, and not under cooldown."""
        # 1. Check Global Circuit Breaker
        if not self.circuit_breaker.is_available(provider, model_id=model_id):
            return False

        health = self.get_provider_health(provider)
        if health in (ProviderHealth.DISCOVERY_UNAVAILABLE, ProviderHealth.AUTHENTICATION_FAILED, ProviderHealth.DISABLED):
            return False

        if model_id not in self._routeable_models.get(provider, []):
            return False

        model = self.get_model(provider, model_id)
        if not model:
            return False
        if model.is_blacklisted or not model.available:
            return False
        if model.cooldown_until > time.monotonic():
            return False

        return True

    def find_routeable_model(
        self,
        capability: str = "chat",
        preferred_providers: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Finds the best available (provider, model_id) matching the capability with circuit breaker awareness."""
        providers = preferred_providers or ["openai", "groq", "gemini"]
        for p in providers:
            if not self.circuit_breaker.is_available(p):
                continue
            candidates = self.get_available_models(p)
            for m_id in candidates:
                model = self.get_model(p, m_id)
                if not model:
                    continue
                if capability == "realtime_audio" and not getattr(model, "realtime_audio_supported", False):
                    continue
                if capability == "vision" and not model.vision_supported:
                    continue
                if capability == "streaming" and not model.streaming_supported:
                    continue
                if capability == "tool_calling" and not model.tool_calling_supported:
                    continue
                if self.circuit_breaker.is_available(p, model_id=m_id):
                    return (p, m_id)
        return None

    def get_discovery_stats(self, provider: str) -> Dict[str, int]:
        """Returns structured statistics: discovered, routeable, rejected, blacklisted."""
        discovered = len(self._discovered_raw.get(provider, []))
        routeable = len(self.get_available_models(provider))
        rejected = len(self._rejected_models.get(provider, {}))
        blacklisted = sum(1 for m in self._models.values() if m.provider == provider and m.is_blacklisted)
        return {
            "discovered": discovered,
            "routeable": routeable,
            "rejected": rejected,
            "blacklisted": blacklisted,
        }

    def discover_models(self, provider: str, api_key: str) -> List[str]:
        """Dispatches discovery to the specific provider and updates registry without blind fallbacks."""
        if not api_key:
            self._provider_health[provider] = ProviderHealth.AUTHENTICATION_FAILED
            self._log("warn", f"⚠️ ProviderRegistry: {provider.capitalize()} API key is missing. Status set to AUTHENTICATION_FAILED.")
            return []

        if provider == "groq":
            return self._discover_groq_models(api_key)
        elif provider == "gemini":
            return self._discover_gemini_models(api_key)
        return []

    def _discover_groq_models(self, api_key: str) -> List[str]:
        """Queries Groq /models API at runtime and registers active capability-filtered models."""
        self._discovered_raw["groq"] = []
        self._routeable_models["groq"] = []
        self._rejected_models["groq"] = {}

        if not api_key:
            self._provider_health["groq"] = ProviderHealth.AUTHENTICATION_FAILED
            return []

        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Astro-V1-SocialRobot/2.0",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_models = data.get("data", [])

            active_ids = []

            for item in raw_models:
                m_id = item.get("id", "")
                if not m_id:
                    continue
                self._discovered_raw["groq"].append(m_id)

                if item.get("active") is False:
                    self._rejected_models["groq"][m_id] = "inactive_model"
                    continue
                if any(x in m_id.lower() for x in ("canopylabs", "orpheus", "tts", "audio")):
                    self._rejected_models["groq"][m_id] = "tts_or_audio_model"
                    continue
                if any(x in m_id.lower() for x in ("qwen3.6", "1b", "3b", "qwen/qwen")):
                    self._rejected_models["groq"][m_id] = "preview_or_unverified_model"
                    continue
                if any(x in m_id.lower() for x in ("whisper", "embed")):
                    self._rejected_models["groq"][m_id] = "non_chat_modality"
                    continue
                if any(x in m_id.lower() for x in ("guard", "safety")):
                    self._rejected_models["groq"][m_id] = "safety_guard_model"
                    continue
                if any(x in m_id.lower() for x in ("specdec", "allam", "r1", "deepseek", "compound")):
                    self._rejected_models["groq"][m_id] = "unsupported_architecture"
                    continue
                if m_id not in GROQ_PRODUCTION_MODELS:
                    self._rejected_models["groq"][m_id] = "non_production_chat_model"
                    continue

                active_ids.append(m_id)
                is_vis = ("vision" in m_id.lower() or "vl" in m_id.lower())
                self.register_model(
                    ModelCapability(
                        provider="groq",
                        model_id=m_id,
                        chat_supported=True,
                        streaming_supported=True,
                        tool_calling_supported="llama" in m_id.lower(),
                        vision_supported=is_vis,
                    )
                )

            # Order by preferred priority among actually discovered models
            ordered = [m for m in GROQ_PREFERENCE_ORDER if m in active_ids]
            for m in active_ids:
                if m not in ordered:
                    ordered.append(m)

            self._routeable_models["groq"] = ordered

            if ordered:
                self._provider_health["groq"] = ProviderHealth.HEALTHY
                stats = self.get_discovery_stats("groq")
                self._log(
                    "info",
                    f"✅ ProviderRegistry: Groq discovered={stats['discovered']} routeable={stats['routeable']} rejected={stats['rejected']} (top: {ordered[:3]})"
                )
            else:
                self._provider_health["groq"] = ProviderHealth.DISCOVERY_UNAVAILABLE
                self._log("warn", "⚠️ ProviderRegistry: Groq discovery returned 0 routeable chat models.")

            return ordered

        except urllib.error.HTTPError as http_e:
            err_body = http_e.read().decode("utf-8", errors="ignore")
            if http_e.code in (401, 403):
                self._provider_health["groq"] = ProviderHealth.AUTHENTICATION_FAILED
                self._log("error", f"⛔ ProviderRegistry: Groq authentication failed (HTTP {http_e.code}).")
            else:
                self._provider_health["groq"] = ProviderHealth.DISCOVERY_UNAVAILABLE
                self._log("warn", f"⚠️ ProviderRegistry: Groq discovery HTTP error {http_e.code}: {err_body[:100]}")
            return []
        except Exception as e:
            self._provider_health["groq"] = ProviderHealth.DISCOVERY_UNAVAILABLE
            self._log("warn", f"⚠️ ProviderRegistry: Groq discovery failed ({e}). Provider marked DISCOVERY_UNAVAILABLE.")
            return []

    def _discover_gemini_models(self, api_key: str) -> List[str]:
        """Queries Google Gemini /models API at runtime and registers active capability-filtered models."""
        self._discovered_raw["gemini"] = []
        self._routeable_models["gemini"] = []
        self._rejected_models["gemini"] = {}

        if not api_key:
            self._provider_health["gemini"] = ProviderHealth.AUTHENTICATION_FAILED
            return []

        try:
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                headers={"User-Agent": "Astro-V1-SocialRobot/2.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_models = data.get("models", [])

            has_25_family = any("2.5" in item.get("name", "") for item in raw_models)
            active_ids = []

            for item in raw_models:
                raw_name = item.get("name", "")
                m_id = raw_name.replace("models/", "")
                if not m_id:
                    continue
                self._discovered_raw["gemini"].append(m_id)

                methods = item.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    self._rejected_models["gemini"][m_id] = "no_generate_content"
                    continue
                if any(x in m_id.lower() for x in ("-image", "image-", "imagen")):
                    self._rejected_models["gemini"][m_id] = "image_generation_model"
                    continue
                if any(x in m_id.lower() for x in ("gemini-1.5", "1.5")) and has_25_family:
                    self._rejected_models["gemini"][m_id] = "deprecated_or_legacy_family"
                    continue
                if any(x in m_id.lower() for x in ("embedding", "aqa", "imagen", "tts", "stt")):
                    self._rejected_models["gemini"][m_id] = "non_chat_modality"
                    continue


                if any(x in m_id.lower() for x in ("bison", "chat-bison", "learnlm", "gemini-1.0")):
                    self._rejected_models["gemini"][m_id] = "legacy_deprecated_family"
                    continue
                if any(x in m_id.lower() for x in ("experimental", "preview-")):
                    self._rejected_models["gemini"][m_id] = "experimental_preview"
                    continue
                if m_id not in GEMINI_PRODUCTION_MODELS:
                    self._rejected_models["gemini"][m_id] = "non_production_llm_model"
                    continue

                active_ids.append(m_id)
                self.register_model(
                    ModelCapability(
                        provider="gemini",
                        model_id=m_id,
                        chat_supported=True,
                        streaming_supported=True,
                        tool_calling_supported=True,
                        vision_supported="flash" in m_id.lower() or "pro" in m_id.lower(),
                    )
                )

            ordered = [m for m in GEMINI_PREFERENCE_ORDER if m in active_ids]
            for m in active_ids:
                if m not in ordered:
                    ordered.append(m)

            self._routeable_models["gemini"] = ordered

            if ordered:
                self._provider_health["gemini"] = ProviderHealth.HEALTHY
                stats = self.get_discovery_stats("gemini")
                self._log(
                    "info",
                    f"✅ ProviderRegistry: Gemini discovered={stats['discovered']} routeable={stats['routeable']} rejected={stats['rejected']} (top: {ordered[:3]})"
                )
            else:
                self._provider_health["gemini"] = ProviderHealth.DISCOVERY_UNAVAILABLE
                self._log("warn", "⚠️ ProviderRegistry: Gemini discovery returned 0 routeable chat models.")

            return ordered

        except urllib.error.HTTPError as http_e:
            err_body = http_e.read().decode("utf-8", errors="ignore")
            if http_e.code in (401, 403):
                self._provider_health["gemini"] = ProviderHealth.AUTHENTICATION_FAILED
                self._log("error", f"⛔ ProviderRegistry: Gemini authentication failed (HTTP {http_e.code}).")
            else:
                self._provider_health["gemini"] = ProviderHealth.DISCOVERY_UNAVAILABLE
                self._log("warn", f"⚠️ ProviderRegistry: Gemini discovery HTTP error {http_e.code}: {err_body[:100]}")
            return []
        except Exception as e:
            self._provider_health["gemini"] = ProviderHealth.DISCOVERY_UNAVAILABLE
            self._log("warn", f"⚠️ ProviderRegistry: Gemini discovery failed ({e}). Provider marked DISCOVERY_UNAVAILABLE.")
            return []

    def classify_error(self, http_code: int, error_body: str, exc: Optional[Exception] = None) -> ErrorClass:
        """Classifies HTTP status codes and error payloads into strict ErrorClass."""
        body_lower = (error_body or "").lower()

        if http_code in (401, 403):
            return ErrorClass.AUTHENTICATION_ERROR

        if http_code == 404:
            return ErrorClass.MODEL_NOT_FOUND

        if http_code == 429:
            if any(q in body_lower for q in ("quota", "insufficient_quota", "exceeded", "credit", "billing")):
                return ErrorClass.QUOTA_EXHAUSTED
            return ErrorClass.RATE_LIMITED

        if http_code == 400:
            if any(u in body_lower for u in ("not found", "model", "unsupported", "does not exist", "deprecated", "unknown", "invalid model")):
                return ErrorClass.UNSUPPORTED_MODEL
            return ErrorClass.UNSUPPORTED_MODEL

        if http_code in (500, 502, 503, 504):
            return ErrorClass.SERVER_ERROR

        if isinstance(exc, (TimeoutError, urllib.error.URLError)):
            if "timed out" in str(exc).lower():
                return ErrorClass.TIMEOUT
            return ErrorClass.NETWORK_ERROR

        return ErrorClass.SERVER_ERROR

    def mark_unavailable(
        self,
        provider: str,
        model_id: str,
        error_class: ErrorClass,
        cooldown: Optional[float] = None
    ) -> None:
        """Marks a model as blacklisted or in cooldown and updates provider state."""
        key = f"{provider}:{model_id}"
        model = self._models.get(key)
        if not model:
            model = ModelCapability(provider=provider, model_id=model_id)
            self._models[key] = model

        model.last_error_class = error_class
        model.consecutive_errors += 1
        now = time.monotonic()

        if error_class in (ErrorClass.UNSUPPORTED_MODEL, ErrorClass.MODEL_NOT_FOUND, ErrorClass.AUTHENTICATION_ERROR):
            model.is_blacklisted = True
            model.available = False
            self.circuit_breaker.record_error(provider, error_class=RequestErrorClass.MODEL_UNAVAILABLE, error_msg=f"{provider}/{model_id} unavailable", model_id=model_id)
            self._log("error", f"⛔ [ProviderRegistry] Model permanently blacklisted ({error_class.value}): {provider}/{model_id}")
        elif error_class == ErrorClass.QUOTA_EXHAUSTED:
            model.cooldown_until = now + (cooldown or 300.0)
            self.set_provider_health(provider, ProviderHealth.RATE_LIMITED)
            self.circuit_breaker.record_error(provider, error_class=RequestErrorClass.QUOTA_EXHAUSTED, error_msg=f"{provider} quota exhausted", model_id=model_id)
            self._log("warn", f"⏳ [ProviderRegistry] Quota exhausted ({provider}/{model_id}): {cooldown or 300.0}s cooldown.")
        elif error_class == ErrorClass.RATE_LIMITED:
            model.cooldown_until = now + (cooldown or 45.0)
            self.circuit_breaker.record_error(provider, error_class=RequestErrorClass.RATE_LIMITED, error_msg=f"{provider} rate limit", model_id=model_id)
            self._log("warn", f"⏳ [ProviderRegistry] Rate limit ({provider}/{model_id}): {cooldown or 45.0}s cooldown.")
        elif error_class == ErrorClass.SERVER_ERROR:
            model.cooldown_until = now + (cooldown or 20.0)
            self.circuit_breaker.record_error(provider, error_class=RequestErrorClass.SERVER_ERROR, error_msg=f"{provider} server error", model_id=model_id)
        elif error_class in (ErrorClass.TIMEOUT, ErrorClass.NETWORK_ERROR):
            model.cooldown_until = now + (cooldown or 10.0)
            self.circuit_breaker.record_error(provider, error_class=RequestErrorClass.NETWORK_ERROR, error_msg=f"{provider} network error", model_id=model_id)

    def record_error(self, provider: str, model_id: str, error_class: ErrorClass, error_msg: str) -> None:
        """Legacy helper recording error message and marking unavailable."""
        key = f"{provider}:{model_id}"
        model = self._models.get(key)
        if model:
            model.last_error = error_msg
        self.mark_unavailable(provider, model_id, error_class)

    def record_success(self, provider: str, model_id: str, latency_ms: float) -> None:
        """Records a successful turn completion for a model."""
        key = f"{provider}:{model_id}"
        model = self._models.get(key)
        if model:
            model.consecutive_errors = 0
            model.last_error = ""
            model.last_error_class = ErrorClass.NONE
            model.last_latency_ms = latency_ms
            model.cooldown_until = 0.0
        self.set_provider_health(provider, ProviderHealth.HEALTHY)
        self.circuit_breaker.record_success(provider, model_id=model_id)

    def get_available_models(self, provider: str) -> List[str]:
        """Returns list of currently available, non-blacklisted routeable models for provider."""
        health = self.get_provider_health(provider)
        if health in (ProviderHealth.DISCOVERY_UNAVAILABLE, ProviderHealth.AUTHENTICATION_FAILED, ProviderHealth.DISABLED):
            return []

        candidates = []
        routeable = self._routeable_models.get(provider, [])
        for m_id in routeable:
            if self.is_routeable(provider, m_id):
                candidates.append(m_id)

        return candidates

    def get_candidate_models(self, provider: str) -> List[str]:
        """Alias for get_available_models."""
        return self.get_available_models(provider)

    def select_best_model(self, provider: str, capabilities: Optional[List[str]] = None) -> Optional[str]:
        """Selects the highest-priority routeable model matching the required capabilities."""
        available = self.get_available_models(provider)
        if not available:
            return None

        if not capabilities:
            return available[0]

        for m_id in available:
            model = self.get_model(provider, m_id)
            if not model:
                continue
            matches = True
            for cap in capabilities:
                if not getattr(model, f"{cap}_supported", False):
                    matches = False
                    break
            if matches:
                return m_id

        return available[0]

    def stream_groq_completion(
        self,
        api_key: str,
        model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.65,
        timeout: float = 2.5,
    ) -> Generator[str, None, None]:
        """Streams tokens from Groq Chat Completions API with strict error classification and immediate blacklisting."""
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "presence_penalty": 0.2,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if any(x in model_id.lower() for x in ("gpt-oss", "reasoning", "r1")):
            payload["reasoning_effort"] = "low"
            payload["include_reasoning"] = False

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Astro-V1-SocialRobot/2.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except Exception as _exc:
                        self._log("debug", f"stream_groq_completion: yok sayılan hata ({_exc})")
        except urllib.error.HTTPError as http_e:
            err_body = http_e.read().decode("utf-8", errors="ignore")
            err_class = self.classify_error(http_e.code, err_body, http_e)
            self.record_error("groq", model_id, err_class, err_body)
            raise ProviderError("groq", model_id, err_class, err_body, http_e.code)
        except Exception as ge:
            err_class = self.classify_error(0, str(ge), ge)
            self.record_error("groq", model_id, err_class, str(ge))
            raise ProviderError("groq", model_id, err_class, str(ge))

    def generate_gemini_content(
        self,
        api_key: str,
        model_id: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.65,
        timeout: float = 4.0,
    ) -> str:
        """Generates text from Google Gemini REST API with strict error classification and immediate blacklisting."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

        conv_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        full_text = f"SİSTEM YÖNERGESİ:\n{system_prompt}\n\nKONUŞMA GEÇMİŞİ:\n{conv_text}"

        gem_payload = {
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(gem_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Astro-V1-SocialRobot/2.0"},
            method="POST",
        )

        t_start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                latency_ms = (time.monotonic() - t_start) * 1000.0
                res_json = json.loads(resp.read().decode("utf-8"))
                candidates = res_json.get("candidates", [])
                text_out = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text_out = parts[0].get("text", "").strip()
                self._log(
                    "info",
                    f"⚡ [LLM CALL] provider=gemini model={model_id} mode=generateContent endpoint=generativelanguage.googleapis.com latency_ms={latency_ms:.1f} status=200 error=none"
                )
                return text_out
        except urllib.error.HTTPError as http_e:
            latency_ms = (time.monotonic() - t_start) * 1000.0
            err_body = http_e.read().decode("utf-8", errors="ignore")
            err_class = self.classify_error(http_e.code, err_body, http_e)
            self.record_error("gemini", model_id, err_class, err_body)
            self._log(
                "warn" if err_class in (ErrorClass.QUOTA_EXHAUSTED, ErrorClass.RATE_LIMITED) else "error",
                f"⚠️ [LLM CALL ERROR] provider=gemini model={model_id} mode=generateContent endpoint=generativelanguage.googleapis.com latency_ms={latency_ms:.1f} status={http_e.code} error_class={err_class.value} reason={err_body[:100]}"
            )
            raise ProviderError("gemini", model_id, err_class, err_body, http_e.code)
        except Exception as ge:
            latency_ms = (time.monotonic() - t_start) * 1000.0
            err_class = self.classify_error(0, str(ge), ge)
            self.record_error("gemini", model_id, err_class, str(ge))
            self._log(
                "error",
                f"⛔ [LLM CALL ERROR] provider=gemini model={model_id} mode=generateContent endpoint=generativelanguage.googleapis.com latency_ms={latency_ms:.1f} status=0 error_class={err_class.value} reason={str(ge)[:100]}"
            )
            raise ProviderError("gemini", model_id, err_class, str(ge))



def _cli_discover():
    parser = argparse.ArgumentParser(description="ASTRO V1 Runtime Provider & Model Discovery Tool")
    parser.add_argument("--discover", action="store_true", help="Queries and prints routeable models for Groq & Gemini")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            load_dotenv()
    except ImportError as _exc:
        _LOG.debug("_cli_discover: yok sayılan hata (%s)", _exc)

    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    registry = ProviderRegistry()
    groq_models = registry.discover_models("groq", groq_key)
    gemini_models = registry.discover_models("gemini", gemini_key)

    print("\n" + "=" * 65)
    print(" [*] ASTRO V1 Runtime Provider Discovery Report")
    print("=" * 65)

    # Groq Report
    g_stats = registry.get_discovery_stats("groq")
    print(f"\n[Groq] Status: {registry.get_provider_health('groq').value.upper()} | discovered={g_stats['discovered']} routeable={g_stats['routeable']} rejected={g_stats['rejected']} blacklisted={g_stats['blacklisted']}")
    if groq_models:
        print(f"  Routeable Production Models ({len(groq_models)}):")
        for m in groq_models:
            print(f"    - {m}")
    else:
        print("  [!] No routeable Groq models discovered.")
    if registry._rejected_models.get("groq"):
        print(f"  Rejected Models ({len(registry._rejected_models['groq'])}):")
        for r_id, r_reason in list(registry._rejected_models["groq"].items())[:6]:
            print(f"    - {r_id} ({r_reason})")

    # Gemini Report
    gem_stats = registry.get_discovery_stats("gemini")
    print(f"\n[Gemini] Status: {registry.get_provider_health('gemini').value.upper()} | discovered={gem_stats['discovered']} routeable={gem_stats['routeable']} rejected={gem_stats['rejected']} blacklisted={gem_stats['blacklisted']}")
    if gemini_models:
        print(f"  Routeable Production Models ({len(gemini_models)}):")
        for m in gemini_models:
            print(f"    - {m}")
    else:
        print("  [!] No routeable Gemini models discovered.")
    if registry._rejected_models.get("gemini"):
        print(f"  Rejected Models ({len(registry._rejected_models['gemini'])}):")
        for r_id, r_reason in list(registry._rejected_models["gemini"].items())[:6]:
            print(f"    - {r_id} ({r_reason})")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    _cli_discover()
