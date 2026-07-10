"""
CARE-Agent — LLM Client
========================
Unified client supporting 12 providers via OpenAI-compatible API.

Provider groups:
  🏠 Local / Self-hosted  — Ollama, LM Studio  (no API key needed)
  🌐 International APIs   — OpenAI, OpenRouter, Groq, Mistral, Together AI
  🇨🇳 Chinese APIs        — Qwen (Alibaba), Kimi (Moonshot), DeepSeek,
                             Zhipu AI (GLM), Baichuan

All providers use the OpenAI chat-completions interface:
    POST {base_url}/chat/completions
    Authorization: Bearer {api_key}

Local providers ignore the API key entirely.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional


# ── Provider registry ──────────────────────────────────────────────────────────
# Each entry:
#   base_url        — endpoint root (no trailing slash)
#   label           — display name
#   group           — UI grouping
#   api_key_label   — hint shown next to the key field
#   api_key_url     — where to get a key (empty for local)
#   needs_key       — False for local providers
#   default_model   — pre-selected model
#   models          — dropdown list  (empty list → show free-text input)
#   custom_base_url — True if user can override the base URL
#   key_placeholder — placeholder text in the API key field

PROVIDERS: dict[str, dict] = {

    # ── 🏠 Local / Self-hosted ────────────────────────────────────────────────
    "ollama": {
        "base_url":        "http://localhost:11434/v1",
        "label":           "Ollama (local)",
        "group":           "🏠 Local / Self-hosted",
        "api_key_label":   "No key needed",
        "api_key_url":     "",
        "needs_key":       False,
        "default_model":   "llama3.3",
        "models":          [],   # free-text: run `ollama list` to see installed
        "custom_base_url": True,
        "key_placeholder": "(not required)",
    },
    "lmstudio": {
        "base_url":        "http://localhost:1234/v1",
        "label":           "LM Studio (local)",
        "group":           "🏠 Local / Self-hosted",
        "api_key_label":   "No key needed",
        "api_key_url":     "",
        "needs_key":       False,
        "default_model":   "",
        "models":          [],   # free-text: use model ID shown in LM Studio
        "custom_base_url": True,
        "key_placeholder": "(not required)",
    },

    # ── 🌐 International APIs ─────────────────────────────────────────────────
    "openai": {
        "base_url":        "https://api.openai.com/v1",
        "label":           "OpenAI",
        "group":           "🌐 International APIs",
        "api_key_label":   "API Key",
        "api_key_url":     "https://platform.openai.com/api-keys",
        "needs_key":       True,
        "default_model":   "gpt-5.5",
        "models": [
            # GPT-5.5 family (Apr 2026) — newest frontier
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.5-2026-04-23",
            # GPT-5.4 family
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-2026-03-05",
            # GPT-5.3 / Codex
            "gpt-5.3",
            "gpt-5.3-codex",
            # GPT-5.1 / GPT-5
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            # GPT-4.1 family (legacy)
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            # GPT-4o family (legacy)
            "gpt-4o",
            "gpt-4o-mini",
            # o-series reasoning (legacy)
            "o4-mini",
            "o3",
            "o3-mini",
        ],
        "custom_base_url": False,
        "key_placeholder": "sk-...",
    },
    "openrouter": {
        "base_url":        "https://openrouter.ai/api/v1",
        "label":           "OpenRouter (300+ models)",
        "group":           "🌐 International APIs",
        "api_key_label":   "API Key",
        "api_key_url":     "https://openrouter.ai/keys",
        "needs_key":       True,
        "default_model":   "anthropic/claude-haiku-4.5",
        "models": [
            # Anthropic Claude (4.6/4.7 generation, 2026) — OpenRouter uses dotted versions
            "anthropic/claude-opus-4.7",
            "anthropic/claude-opus-4.6",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-haiku-4.5",
            # OpenAI (GPT-5.5 / 5.4 / 5.3-Codex, 2026)
            "openai/gpt-5.5",
            "openai/gpt-5.5-pro",
            "openai/gpt-5.4",
            "openai/gpt-5.3-codex",
            "openai/gpt-5.1",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            # Google Gemini 3.x (2026)
            "google/gemini-3.1-pro-preview",
            "google/gemini-3.1-flash-lite-preview",
            "google/gemini-3-flash-preview",
            "google/gemini-2.5-flash",
            # Meta Llama 4 (2026)
            "meta-llama/llama-4-maverick",
            "meta-llama/llama-4-scout",
            "meta-llama/llama-3.3-70b-instruct",
            # DeepSeek V4 (Apr 2026)
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
            "deepseek/deepseek-v3.2",
            "deepseek/deepseek-chat",
            "deepseek/deepseek-r1",
            # Qwen3.x (2026)
            "qwen/qwen3.5-flash-02-23",
            "qwen/qwen3-max",
            "qwen/qwen3-coder-next",
            "qwen/qwen3-235b-a22b-2507",
            "qwen/qwen3-235b-a22b-thinking-2507",
            "qwen/qwen3-next-80b-a3b-instruct",
            "qwen/qwen3-next-80b-a3b-thinking",
            # Zhipu GLM (2026) — OpenRouter namespaces these under z-ai/
            "z-ai/glm-5",
            "z-ai/glm-4.6",
            # Mistral (2026)
            "mistralai/mistral-large-2512",
            "mistralai/mistral-small-2603",
            "mistralai/mistral-medium-3.1",
            "mistralai/ministral-14b-2512",
            # Kimi / Moonshot (2026)
            "moonshotai/kimi-k2.6",
            "moonshotai/kimi-k2.5",
            # MiniMax / Xiaomi (2026)
            "minimax/minimax-m2.7",
            "xiaomi/mimo-v2-pro",
            # NVIDIA
            "nvidia/nemotron-3-super-120b-a12b",
            # Free auto-router
            "openrouter/auto",
        ],
        "custom_base_url": False,
        "key_placeholder": "sk-or-...",
    },
    "groq": {
        "base_url":        "https://api.groq.com/openai/v1",
        "label":           "Groq (fast inference)",
        "group":           "🌐 International APIs",
        "api_key_label":   "API Key",
        "api_key_url":     "https://console.groq.com/keys",
        "needs_key":       True,
        "default_model":   "llama-3.3-70b-versatile",
        "models": [
            # Llama
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            # OpenAI open-weight (gpt-oss)
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
            # DeepSeek distilled reasoners
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
            # Kimi
            "moonshotai/kimi-k2-instruct",
            # Qwen
            "qwen/qwen3-32b",
            # Google
            "gemma2-9b-it",
            # Compound (built-in tools)
            "compound-beta",
            "compound-beta-mini",
        ],
        "custom_base_url": False,
        "key_placeholder": "gsk_...",
    },
    "mistral": {
        "base_url":        "https://api.mistral.ai/v1",
        "label":           "Mistral AI",
        "group":           "🌐 International APIs",
        "api_key_label":   "API Key",
        "api_key_url":     "https://console.mistral.ai/api-keys",
        "needs_key":       True,
        "default_model":   "mistral-large-latest",
        "models": [
            # Frontier (Dec 2025 → 2026)
            "mistral-large-latest",          # → Mistral Large 3 (675B / 41B active, 256k ctx)
            "mistral-small-latest",          # → Mistral Small 4 (unified flagship)
            # Reasoning
            "magistral-medium-latest",
            "magistral-small-latest",
            # Ministral 3 family (Dec 2025, Apache 2.0)
            "ministral-3-14b-latest",
            "ministral-3-8b-latest",
            "ministral-3-3b-latest",
            # Specialist
            "pixtral-large-latest",          # multimodal
            "codestral-latest",              # coding
            "devstral-latest",               # agentic coding
            "voxtral-tts-latest",            # text-to-speech (Mar 2026)
            # Open / legacy
            "open-mistral-nemo",
            "open-mixtral-8x22b",
        ],
        "custom_base_url": False,
        "key_placeholder": "...",
    },
    "together": {
        "base_url":        "https://api.together.xyz/v1",
        "label":           "Together AI (200+ models)",
        "group":           "🌐 International APIs",
        "api_key_label":   "API Key",
        "api_key_url":     "https://api.together.ai/settings/api-keys",
        "needs_key":       True,
        "default_model":   "deepseek-ai/DeepSeek-V4-Pro",
        "models": [
            # DeepSeek V4 (Apr 2026)
            "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek-ai/DeepSeek-V4-Flash",
            "deepseek-ai/DeepSeek-V3.1",
            "deepseek-ai/DeepSeek-R1-0528",
            "deepseek-ai/DeepSeek-V3-0324",
            # Kimi (Moonshot)
            "moonshotai/Kimi-K2.6-Instruct",
            "moonshotai/Kimi-K2.5-Instruct",
            # Zhipu GLM (2026)
            "zhipu-ai/GLM-5",
            "zhipu-ai/GLM-5.1",
            # Qwen3.x
            "Qwen/Qwen3.5-397B-A17B",
            "Qwen/Qwen3-Coder-Next",
            "Qwen/Qwen3-Coder-480B",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "Qwen/Qwen3.5-9B",
            # MiniMax / Xiaomi
            "MiniMaxAI/MiniMax-M2.7",
            # Llama 5 / 4
            "meta-llama/Llama-5",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            # NVIDIA Nemotron
            "nvidia/Nemotron-3-Super",
            # Google
            "google/gemma-4-31b-it",
            "google/gemma-4-26b-a4b-it",
            # Mistral (legacy)
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ],
        "custom_base_url": False,
        "key_placeholder": "...",
    },

    # ── 🇨🇳 Chinese APIs ─────────────────────────────────────────────────────
    "qwen": {
        "base_url":        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "label":           "Qwen / 通义千问 (Alibaba)",
        "group":           "🇨🇳 Chinese APIs",
        "api_key_label":   "DashScope API Key",
        "api_key_url":     "https://dashscope.console.aliyun.com/apiKey",
        "needs_key":       True,
        "default_model":   "qwen-plus",
        "models": [
            # Flagship aliases (always latest snapshot)
            "qwen3-max",
            "qwen-plus",
            "qwen3.5-flash",
            "qwen-turbo",
            "qwen-long",
            # Qwen3-Next (MoE, 2025-2026)
            "qwen3-next-80b-a3b-instruct",
            "qwen3-next-80b-a3b-thinking",
            # Qwen3 2507 snapshots (instruct + thinking)
            "qwen3-235b-a22b-instruct-2507",
            "qwen3-235b-a22b-thinking-2507",
            "qwen3-30b-a3b-instruct-2507",
            "qwen3-30b-a3b-thinking-2507",
            # Qwen3 dense
            "qwen3-235b-a22b",
            "qwen3-32b",
            "qwen3-14b",
            "qwen3-8b",
            "qwen3-4b",
            # Multimodal
            "qwen3-omni-flash",
            "qwen3-vl-plus",
            # Coder
            "qwen3-coder-next",
            "qwen3-coder-480b",
            # Embedding
            "text-embedding-v4",
        ],
        "custom_base_url": False,
        "key_placeholder": "sk-...",
    },
    "kimi": {
        "base_url":        "https://api.moonshot.cn/v1",
        "label":           "Kimi / 月之暗面 (Moonshot AI)",
        "group":           "🇨🇳 Chinese APIs",
        "api_key_label":   "Moonshot API Key",
        "api_key_url":     "https://platform.moonshot.cn/console/api-keys",
        "needs_key":       True,
        "default_model":   "kimi-k2.6",
        "models": [
            # K2 generation (2026)
            "kimi-k2.6",                # Apr 2026 flagship, multimodal + agentic
            "kimi-k2.5",
            "kimi-k2",
            # Legacy moonshot-v1 series
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
            "moonshot-v1-8k-vision-preview",
            "moonshot-v1-32k-vision-preview",
        ],
        "custom_base_url": False,
        "key_placeholder": "sk-...",
    },
    "deepseek": {
        "base_url":        "https://api.deepseek.com/v1",
        "label":           "DeepSeek",
        "group":           "🇨🇳 Chinese APIs",
        "api_key_label":   "API Key",
        "api_key_url":     "https://platform.deepseek.com/api_keys",
        "needs_key":       True,
        "default_model":   "deepseek-v4-flash",
        "models": [
            # DeepSeek V4 (Apr 2026, 1M context, 32T-token training)
            "deepseek-v4-pro",       # frontier — full-power V4
            "deepseek-v4-flash",     # fast, default — replaces deepseek-chat / deepseek-reasoner
            # V3.2 (current behind deepseek-chat / deepseek-reasoner)
            "deepseek-v3.2",
            # Legacy aliases (deprecated 2026-07-24, map to v4-flash modes)
            "deepseek-chat",         # non-thinking mode
            "deepseek-reasoner",     # thinking mode
        ],
        "custom_base_url": False,
        "key_placeholder": "sk-...",
    },
    "zhipu": {
        "base_url":        "https://open.bigmodel.cn/api/paas/v4",
        "label":           "Zhipu AI / 智谱 GLM",
        "group":           "🇨🇳 Chinese APIs",
        "api_key_label":   "API Key",
        "api_key_url":     "https://open.bigmodel.cn/usercenter/apikeys",
        "needs_key":       True,
        "default_model":   "glm-5",
        "models": [
            # GLM-5 (Feb 2026, 745B MoE / 44B active)
            "glm-5",
            "glm-5-air",
            # GLM-4.6 (late-2025 / 2026)
            "glm-4.6",
            "glm-4.6-air",
            # GLM-4.5
            "glm-4.5",
            "glm-4.5-air",
            # GLM-4 line
            "glm-4-plus",
            "glm-4-flash",
            "glm-4-air",
            "glm-4-long",
            # Reasoning (Z1)
            "glm-z1-plus",
            "glm-z1-air",
            "glm-z1-flash",
        ],
        "custom_base_url": False,
        "key_placeholder": "...",
    },
    "baichuan": {
        "base_url":        "https://api.baichuan-ai.com/v1",
        "label":           "Baichuan AI / 百川",
        "group":           "🇨🇳 Chinese APIs",
        "api_key_label":   "API Key",
        "api_key_url":     "https://platform.baichuan-ai.com/console/apikey",
        "needs_key":       True,
        "default_model":   "Baichuan4-Turbo",
        "models": [
            # Baichuan-M3 (2026, 235B, medical-enhanced)
            "Baichuan-M3-235B",
            "Baichuan-M2",
            # Baichuan 4 generation
            "Baichuan4",
            "Baichuan4-Turbo",
            "Baichuan4-Air",
            # Baichuan 2 / 3 (legacy long-context)
            "Baichuan2-Turbo",
            "Baichuan2-Turbo-192k",
            "Baichuan3-Turbo",
            "Baichuan3-Turbo-128k",
        ],
        "custom_base_url": False,
        "key_placeholder": "...",
    },
}

# Ordered provider keys for display (grouped)
PROVIDER_ORDER = [
    "ollama", "lmstudio",
    "openai", "openrouter", "groq", "mistral", "together",
    "qwen", "kimi", "deepseek", "zhipu", "baichuan",
]

# Group labels for UI section headers
PROVIDER_GROUPS = [
    "🏠 Local / Self-hosted",
    "🌐 International APIs",
    "🇨🇳 Chinese APIs",
]


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    success: bool
    error: str = ""
    tokens_used: int = 0
    finish_reason: str = ""     # 'stop', 'length' (truncated), etc.
    truncated: bool = False     # True if the model hit the output-token ceiling


class LLMClient:
    """
    Unified LLM client. All providers use the OpenAI-compatible chat API.
    Local providers (Ollama, LM Studio) require no API key.
    """

    def __init__(
        self,
        provider: str = "openrouter",
        api_key: str = "",
        model: Optional[str] = None,
        base_url_override: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,    # generous: reasoning models spend output budget on hidden thinking
        timeout: float = 120.0,    # slow reasoning models (e.g. DeepSeek) can take >60s
        max_retries: int = 2,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")

        info = PROVIDERS[provider]
        self.provider    = provider
        # Local providers (Ollama / LM Studio) ignore the key, so a placeholder is
        # fine. Providers that need a key keep "" so the missing-key guard fires.
        self.api_key     = api_key or ("local" if not info["needs_key"] else "")
        self.model       = model or info["default_model"]
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout        # seconds before a hung request gives up
        self.max_retries = max_retries    # SDK-level retries for transient network errors
        self._base_url   = base_url_override or info["base_url"]
        self._needs_key  = info["needs_key"]

    # ── Request helpers ────────────────────────────────────────────────────────
    def _create(self, client, messages, use_completion_tokens: bool, use_temperature: bool,
                force_json: bool = False, max_tokens: Optional[int] = None):
        """Build kwargs and call the chat endpoint.

        Newer reasoning models (OpenAI o-series / GPT-5 family and some others)
        reject ``max_tokens`` (they want ``max_completion_tokens``) and only allow
        the default temperature. We try the standard parameters first and fall
        back to the reasoning-model variant on a parameter error.

        ``force_json`` asks the provider for a JSON object via response_format;
        ``max_tokens`` lets the caller raise the ceiling on a truncation retry.
        """
        mt = max_tokens or self.max_tokens
        kwargs = {"model": self.model, "messages": messages}
        if use_completion_tokens:
            kwargs["max_completion_tokens"] = mt
        else:
            kwargs["max_tokens"] = mt
        if use_temperature:
            kwargs["temperature"] = self.temperature
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    def _safe_create(self, client, messages, force_json: bool, max_tokens: int):
        """Call the endpoint, peeling off parameters the model rejects.

        Order of fallbacks: (1) standard params; (2) reasoning-model params
        (max_completion_tokens, no temperature); (3) drop response_format if the
        provider doesn't support JSON mode; (4) reasoning params + no JSON mode.
        Non-parameter errors (auth, 404, network) propagate immediately.
        """
        combos = [
            dict(uct=False, ut=True,  fj=force_json),
            dict(uct=True,  ut=False, fj=force_json),
        ]
        if force_json:   # add JSON-disabled variants as a further fallback
            combos += [dict(uct=False, ut=True, fj=False),
                       dict(uct=True,  ut=False, fj=False)]
        last = None
        for c in combos:
            try:
                return self._create(client, messages, c["uct"], c["ut"],
                                    force_json=c["fj"], max_tokens=max_tokens)
            except Exception as ex:
                last = ex
                if self._is_param_error(ex) or self._is_response_format_error(ex):
                    continue
                raise
        raise last

    @staticmethod
    def _is_response_format_error(e: Exception) -> bool:
        """True if the error is about an unsupported response_format / JSON mode."""
        msg = str(e).lower()
        status = getattr(e, "status_code", None)
        if status not in (None, 400, 404, 422):
            return False
        return "response_format" in msg or "json_object" in msg or "json mode" in msg

    @staticmethod
    def _is_rate_limit(e: Exception) -> bool:
        """True if the error is a 429 / rate-limit / quota throttle."""
        status = getattr(e, "status_code", None)
        if status == 429:
            return True
        msg = str(e).lower()
        return ("rate limit" in msg or "rate_limit" in msg or "429" in msg
                or "too many requests" in msg)

    @staticmethod
    def _retry_after(e: Exception) -> Optional[float]:
        """Seconds to wait per the provider's Retry-After / rate-limit header.

        Handles three flavours we've seen in the wild:
          1. Integer "seconds to wait"           e.g. "30"
          2. HTTP-date string                    e.g. "Wed, 21 Oct 2026 07:28:00 GMT"
          3. Absolute Unix timestamp             e.g. "1748286123"     (seconds)
                                                 or  "1748286123000"  (milliseconds)
             This is what Qwen-via-OpenRouter returns in `x-ratelimit-reset`,
             and the cause of the "timestamp out of range for platform time_t"
             error if we naively pass it to time.sleep().

        Returns the wait in seconds, capped at 60 s (no sane provider should
        force us to wait longer than that — better to fail fast).
        """
        import time as _time
        from email.utils import parsedate_to_datetime

        WAIT_CEILING = 60.0

        resp = getattr(e, "response", None)
        headers = getattr(resp, "headers", None)
        if not headers:
            return None

        for k in ("retry-after", "Retry-After",
                  "x-ratelimit-reset", "X-RateLimit-Reset",
                  "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            v = headers.get(k) if hasattr(headers, "get") else None
            if v is None:
                continue
            v = str(v).strip()
            if not v:
                continue

            # (1) Pure number — could be "seconds to wait" or an absolute timestamp.
            try:
                num = float(v)
            except (TypeError, ValueError):
                num = None

            if num is not None:
                now = _time.time()
                # Heuristic: anything bigger than 1e9 is plausibly a unix timestamp
                # (1e9 s ≈ year 2001). Convert ms→s if it looks like milliseconds.
                if num > 1e12:           # milliseconds since epoch
                    num = num / 1000.0
                if num > now * 0.9:       # absolute future timestamp
                    wait = max(0.0, num - now)
                else:                     # plain "seconds to wait"
                    wait = max(0.0, num)
                return min(WAIT_CEILING, wait)

            # (2) HTTP-date string per RFC 7231.
            try:
                dt = parsedate_to_datetime(v)
                if dt is not None:
                    wait = max(0.0, dt.timestamp() - _time.time())
                    return min(WAIT_CEILING, wait)
            except (TypeError, ValueError, OverflowError):
                pass

            return None
        return None

    @staticmethod
    def _extract(response):
        """Pull (content, finish_reason, tokens) from a chat response, with a
        fallback to reasoning_content for providers that leave content empty."""
        choice = response.choices[0]
        msg = choice.message
        content = (getattr(msg, "content", None) or "")
        if not content:
            # Some OpenRouter routes return the body in a reasoning field instead
            content = (getattr(msg, "reasoning_content", None)
                       or getattr(msg, "reasoning", None) or "")
        content = content.strip()
        finish = getattr(choice, "finish_reason", None) or ""
        tokens = response.usage.total_tokens if getattr(response, "usage", None) else 0
        return content, finish, tokens

    @staticmethod
    def _is_param_error(e: Exception) -> bool:
        """True if the error looks like an unsupported-parameter 400 (max_tokens /
        temperature) that a reasoning-model retry can fix."""
        msg = str(e).lower()
        status = getattr(e, "status_code", None)
        if status not in (None, 400):
            return False
        markers = (
            "max_tokens", "max_completion_tokens", "unsupported_parameter",
            "unsupported value", "unsupported_value",
        )
        if any(m in msg for m in markers):
            return True
        # "temperature ... unsupported / does not support / only the default"
        if "temperature" in msg and (
            "unsupported" in msg or "does not support" in msg or "default" in msg
        ):
            return True
        return False

    def _friendly_error(self, e: Exception) -> str:
        """Turn a raw SDK/HTTP exception into an actionable, human-readable message."""
        raw   = str(e)
        low   = raw.lower()
        etype = type(e).__name__.lower()
        status = getattr(e, "status_code", None)

        if status == 401 or "authenticationerror" in etype or "invalid api key" in low or "incorrect api key" in low:
            return f"Authentication failed — check your {self.provider} API key."
        if status == 404 or "not found" in low or "does not exist" in low or "no such model" in low or "model_not_found" in low:
            return (f"Model '{self.model}' was not found at {self.provider}. "
                    "Check the exact model ID or whether your account has access to it.")
        if status == 429 or "rate limit" in low or "rate_limit" in low or "quota" in low or "insufficient" in low:
            return f"Rate limit / quota exceeded at {self.provider} — wait a moment or check your plan."
        if status == 400 or "bad request" in low or "invalid_request" in low:
            return (f"Request rejected by {self.provider} (400). Model '{self.model}' may not "
                    "exist or may not accept these parameters. Try a different model.")
        if "timeout" in low or "timeout" in etype or "timed out" in low:
            return f"Request timed out after {self.timeout:.0f}s — {self.provider} may be slow or unreachable."
        if ("connection" in low or "connect" in etype or "getaddrinfo" in low
                or "name or service not known" in low or "failed to establish" in low
                or "refused" in low):
            local_hint = (" For local providers, make sure the server is running."
                          if self.provider in ("ollama", "lmstudio") else "")
            return (f"Could not reach {self.provider} at {self._base_url}. "
                    f"Check your network or the server URL.{local_hint}")
        if "json" in low or "expecting value" in low or "jsondecode" in etype:
            return (f"{self.provider} returned a malformed (non-JSON) response. "
                    "This usually means a wrong base URL or an upstream provider error.")
        # Unknown — surface a trimmed raw message for debugging
        trimmed = raw if len(raw) <= 280 else raw[:280] + "…"
        return f"{type(e).__name__}: {trimmed}"

    def complete(self, system_prompt: str, user_prompt: str,
                 force_json: bool = False) -> LLMResponse:
        """
        Send a chat completion request. Never raises — errors captured in response.

        Robustness built in for reasoning/thinking models:
          • force_json   — request a JSON object (response_format), with graceful
                           fallback if the provider rejects it.
          • empty-retry  — retry blank completions a few times (transient provider
                           hiccups / mis-routes return empty content).
          • truncation   — if the model hits the output ceiling (finish_reason
                           'length'), double the budget and retry, up to a cap.
        """
        if self._needs_key and not self.api_key:
            return LLMResponse(
                content="", model=self.model, provider=self.provider,
                success=False, error="No API key provided. Enter one in the sidebar.",
            )

        try:
            from openai import OpenAI
        except Exception as e:
            return LLMResponse(
                content="", model=self.model, provider=self.provider,
                success=False, error=f"The 'openai' package is not installed: {e}",
            )

        # OpenRouter requires extra headers for attribution
        extra_headers = {}
        if self.provider == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://care-agent.research",
                "X-Title": "CARE-Agent HFpEF Risk Pipeline",
            }

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self._base_url,
                default_headers=extra_headers or None,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        except Exception as e:
            return LLMResponse(
                content="", model=self.model, provider=self.provider,
                success=False, error=self._friendly_error(e),
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        EMPTY_RETRIES = 2                       # retry blank completions this many times
        MAX_TRUNC_BUMPS = 2                     # times we double the budget on truncation
        RATE_RETRIES = 5                        # retry 429s with backoff this many times
        CAP = max(self.max_tokens, 8000)        # never grow the budget past this
        mt = self.max_tokens
        empty_tries = 0
        trunc_bumps = 0
        rate_tries = 0

        while True:
            try:
                response = self._safe_create(client, messages, force_json, mt)
            except Exception as e:
                # Rate-limited (429): wait (honoring Retry-After) and retry, so a
                # high-concurrency burst self-heals instead of failing the cell.
                if self._is_rate_limit(e) and rate_tries < RATE_RETRIES:
                    rate_tries += 1
                    wait = self._retry_after(e) or min(30.0, 2.0 ** rate_tries)
                    time.sleep(wait)
                    continue
                return LLMResponse(
                    content="", model=self.model, provider=self.provider,
                    success=False, error=self._friendly_error(e),
                )

            # Parse defensively
            try:
                content, finish, tokens = self._extract(response)
            except Exception as e:
                return LLMResponse(
                    content="", model=self.model, provider=self.provider,
                    success=False, error=f"Malformed response from {self.provider}: {e}",
                )

            # Empty completion — retry a few times before giving up
            if not content:
                if empty_tries < EMPTY_RETRIES:
                    empty_tries += 1
                    time.sleep(0.8 * empty_tries)
                    continue
                return LLMResponse(
                    content="", model=self.model, provider=self.provider, success=False,
                    finish_reason=finish,
                    error=("The model returned an empty response after retries. It may not "
                           "support chat completions on this route, returns content in a "
                           "non-standard field, or hit a content filter."),
                )

            # Truncated by the output ceiling — grow the budget and retry
            if finish == "length" and trunc_bumps < MAX_TRUNC_BUMPS and mt < CAP:
                trunc_bumps += 1
                mt = min(CAP, mt * 2)
                continue

            return LLMResponse(
                content=content, model=self.model, provider=self.provider,
                success=True, tokens_used=tokens,
                finish_reason=finish, truncated=(finish == "length"),
            )
