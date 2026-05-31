import os
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def call_llm(messages):
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("MODEL")

    if provider == "groq":
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        return response.choices[0].message.content

    if provider == "ollama":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        response = requests.post(
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