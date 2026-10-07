import os
from pathlib import Path


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing values."""
    path = path or Path(".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _require(var: str) -> str:
    load_env()
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"{var} is not set (put it in .env)")
    return value


def get_elevenlabs_key() -> str:
    return _require("ELEVENLABS_API_KEY")


def get_anthropic_key() -> str:
    return _require("ANTHROPIC_API_KEY")


def get_gemini_key() -> str:
    return _require("GEMINI_API_KEY")
