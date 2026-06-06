import os
from dotenv import load_dotenv


load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default

    return parsed if parsed > 0 else default


FAST_DEV_MODE = env_bool("FAST_DEV_MODE", False)
FAST_DEV_SKIP_SQLITE = env_bool("FAST_DEV_SKIP_SQLITE", FAST_DEV_MODE)

DEFAULT_MAX_AGENT_STEPS = 8 if FAST_DEV_MODE else 14
DEFAULT_MAX_TOOL_OUTPUT_LENGTH = 500 if FAST_DEV_MODE else 1200
DEFAULT_MAX_HISTORY_MESSAGES = 4 if FAST_DEV_MODE else 6
DEFAULT_MAX_TRACE_OUTPUT_LENGTH = 800 if FAST_DEV_MODE else 3000

MAX_AGENT_STEPS = env_int("MAX_AGENT_STEPS", DEFAULT_MAX_AGENT_STEPS)
MAX_TOOL_OUTPUT_LENGTH = env_int(
    "MAX_TOOL_OUTPUT_LENGTH",
    DEFAULT_MAX_TOOL_OUTPUT_LENGTH,
)
MAX_HISTORY_MESSAGES = env_int(
    "MAX_HISTORY_MESSAGES",
    DEFAULT_MAX_HISTORY_MESSAGES,
)
MAX_TRACE_OUTPUT_LENGTH = env_int(
    "MAX_TRACE_OUTPUT_LENGTH",
    DEFAULT_MAX_TRACE_OUTPUT_LENGTH,
)
