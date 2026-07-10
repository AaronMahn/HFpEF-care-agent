"""
CARE-Agent — Settings Manager
Persists API keys and UI preferences to a local JSON file so they
survive page refreshes and app restarts.

File location: <project folder>/.care_agent_settings.json
The file is git-ignored by convention (add to .gitignore if needed).
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(_HERE, ".care_agent_settings.json")


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _load() -> dict:
    """Read the settings file. Returns {} on any error."""
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: dict) -> None:
    """Write settings to disk. Silently ignores write errors."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Settings] Warning: could not save settings: {e}")


# ── API key management ────────────────────────────────────────────────────────

def get_api_key(provider: str) -> str:
    """Return the saved API key for a provider, or '' if none."""
    return _load().get("api_keys", {}).get(provider, "")


def set_api_key(provider: str, key: str) -> None:
    """Save an API key for a provider."""
    data = _load()
    data.setdefault("api_keys", {})[provider] = key
    _save(data)


def clear_api_key(provider: str) -> None:
    """Remove the saved API key for a provider."""
    data = _load()
    data.get("api_keys", {}).pop(provider, None)
    _save(data)


def list_saved_providers() -> list[str]:
    """Return a list of providers that have a saved key."""
    return [p for p, k in _load().get("api_keys", {}).items() if k]


# ── General preferences ───────────────────────────────────────────────────────

def get_preference(key: str, default=None):
    """Get a saved UI preference."""
    return _load().get("preferences", {}).get(key, default)


def set_preference(key: str, value) -> None:
    """Persist a UI preference."""
    data = _load()
    data.setdefault("preferences", {})[key] = value
    _save(data)
