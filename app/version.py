"""Claude Code version for User-Agent headers.

The version is used to mimic Claude Code's exact User-Agent when calling
Anthropic's OAuth endpoints. It can be set via CLAUDE_CODE_VERSION env var,
or falls back to querying the installed CLI (if available), or a hardcoded default.
"""

import os
import subprocess

_cached_version: str | None = None

# Fallback if CLI is not installed (e.g. inside Docker)
_DEFAULT_VERSION = "2.1.76"


def get_claude_code_version() -> str:
    """Get Claude Code version, cached after first call."""
    global _cached_version
    if _cached_version is not None:
        return _cached_version

    # 1. Explicit env var (set in docker-compose or by operator)
    env_ver = os.environ.get("CLAUDE_CODE_VERSION", "").strip()
    if env_ver:
        _cached_version = env_ver
        return _cached_version

    # 2. Try the installed CLI
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        ver = result.stdout.strip().split()[0] if result.stdout else ""
        if ver and ver[0].isdigit():
            _cached_version = ver
            return _cached_version
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    _cached_version = _DEFAULT_VERSION
    return _cached_version
