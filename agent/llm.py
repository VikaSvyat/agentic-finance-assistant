import os
import logging
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_groq_client = None
_groq_api_key = None
_ollama_session = None


def get_groq_client():
    global _groq_client, _groq_api_key

    api_key = os.getenv("GROQ_API_KEY")

    if _groq_client is None or _groq_api_key != api_key:
        _groq_client = Groq(api_key=api_key)
        _groq_api_key = api_key

    return _groq_client


def get_ollama_session():
    global _ollama_session

    if _ollama_session is None:
        _ollama_session = requests.Session()

    return _ollama_session


def log_llm_usage(model: str, prompt_tokens: object = "n/a", completion_tokens: object = "n/a") -> None:
    """Log token usage for observability during demos and debugging."""

    try:
        total_tokens = (
            int(prompt_tokens) + int(completion_tokens)
            if prompt_tokens != "n/a" and completion_tokens != "n/a"
            else "n/a"
        )
    except Exception:
        total_tokens = "n/a"

    logger.info(
        "LLM usage\nModel: %s\nPrompt tokens: %s\nCompletion tokens: %s\nTotal tokens: %s",
        model,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


def call_llm(messages):
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("MODEL")

    if provider == "groq":
        client = get_groq_client()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", "n/a")
        completion_tokens = getattr(usage, "completion_tokens", "n/a")
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is not None:
            logger.info(
                "LLM usage\nModel: %s\nPrompt tokens: %s\nCompletion tokens: %s\nTotal tokens: %s",
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )
        else:
            log_llm_usage(model, prompt_tokens, completion_tokens)

        return response.choices[0].message.content

    if provider == "ollama":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        session = get_ollama_session()

        response = session.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()
        log_llm_usage(
            model,
            data.get("prompt_eval_count", "n/a"),
            data.get("eval_count", "n/a"),
        )
        return data["message"]["content"]

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
