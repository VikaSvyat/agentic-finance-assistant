import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECRET_ENV_KEYS = {
    "LLM_PROVIDER",
    "MODEL",
    "GROQ_API_KEY",
    "DATA_DIR",
    "DB_PATH",
    "LOG_FILE",
    "FAST_DEV_MODE",
    "FAST_DEV_SKIP_SQLITE",
    "MAX_AGENT_STEPS",
    "MAX_TOOL_OUTPUT_LENGTH",
    "MAX_HISTORY_MESSAGES",
    "MAX_TRACE_OUTPUT_LENGTH",
    "FINANCE_CACHE_ENABLED",
    "FINANCE_CACHE_MAX_ENTRIES",
}

PATH_DEFAULTS = {
    "DATA_DIR": "data",
    "DB_PATH": "database/finance.db",
    "LOG_FILE": "logs/agent.log",
}


def _stringify_secret(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def apply_streamlit_secrets_to_env() -> None:
    """Copy Streamlit secrets into environment variables when env is missing."""

    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return

    for key in SECRET_ENV_KEYS:
        if os.getenv(key) is not None:
            continue

        try:
            if key in secrets:
                os.environ[key] = _stringify_secret(secrets[key])
        except Exception:
            continue


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def normalize_runtime_paths() -> None:
    """Resolve app storage paths from the project root for local and cloud runs."""

    for key, default_value in PATH_DEFAULTS.items():
        raw_value = os.getenv(key, default_value)
        resolved = resolve_project_path(raw_value)
        os.environ[key] = str(resolved)

    Path(os.environ["DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(os.environ["LOG_FILE"]).parent.mkdir(parents=True, exist_ok=True)


def load_runtime_config() -> None:
    """Load .env, Streamlit secrets, and project-root-relative runtime paths."""

    load_dotenv(PROJECT_ROOT / ".env")
    apply_streamlit_secrets_to_env()
    normalize_runtime_paths()
