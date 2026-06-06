import os
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

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
        return response.json()["message"]["content"]

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
