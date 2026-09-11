from hiver.paths import ROOT
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv(ROOT / ".env")

PROMPT = "Reply with exactly: Connection successful"


def get_key(name):
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your_"):
        raise ValueError(f"Add your real {name} to .env")
    return value


def safe_error(error):
    message = str(error)

    # Remove keys if an error happens to include them.
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        key = os.getenv(name, "")
        if key:
            message = message.replace(key, "[HIDDEN]")

    return message[:800]


def test_gemini():
    client = genai.Client(
        api_key=get_key("GEMINI_API_KEY"),
        http_options={"timeout": 60_000},
    )

    try:
        response = client.interactions.create(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            input=PROMPT,
        )
        text = response.output_text

        if not text or not text.strip():
            raise RuntimeError("Gemini returned no text.")

        print("Gemini:", text.strip())
    finally:
        client.close()


def test_groq():
    with Groq(
        api_key=get_key("GROQ_API_KEY"),
        timeout=60.0,
        max_retries=0,
    ) as client:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            messages=[{"role": "user", "content": PROMPT}],
            max_completion_tokens=256,
        )

        if not response.choices:
            raise RuntimeError("Groq returned no choices.")

        text = response.choices[0].message.content
        if not text or not text.strip():
            raise RuntimeError("Groq returned no text.")

        print("Groq:", text.strip())


if __name__ == "__main__":
    failures = 0

    for name, test in (("Gemini", test_gemini), ("Groq", test_groq)):
        print(f"\nTesting {name}...")
        try:
            test()
        except Exception as error:
            failures += 1
            print(f"{name} FAILED: {safe_error(error)}")

    print(f"\nPassed: {2 - failures}/2")
    raise SystemExit(1 if failures else 0)