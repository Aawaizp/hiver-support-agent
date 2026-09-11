from hiver.paths import ROOT
import hashlib
import json
from pathlib import Path
import random
import re
import time

CACHE = ROOT / "data/api_cache"


def generate_cached(client, model, instructions, payload, schema, validate):
    request = {
        "model": model,
        "system_instruction": instructions,
        "input": json.dumps(payload, ensure_ascii=False),
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        },
    }

    fingerprint = json.dumps(
        request, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    cache_id = hashlib.sha256(fingerprint).hexdigest()

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cache_id}.json"

    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        text = saved["text"]
        validate(text)
        print("[Using cached response]")
        return text

    # At most three attempts. Never retry indefinitely.
    for attempt in range(3):
        try:
            response = client.interactions.create(**request)
            text = response.output_text

            if not text:
                raise RuntimeError("The model returned no text.")

            # Cache only responses that pass schema validation.
            validate(text)

            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"text": text}, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(path)
            return text

        except Exception as error:
            code = getattr(error, "status_code", None)
            if code is None:
                code = getattr(error, "code", None)

            message = str(error)
            rate_limited = (
                str(code) == "429"
                or "RESOURCE_EXHAUSTED" in message
                or "too_many_requests" in message.lower()
            )

            if not rate_limited:
                raise

            if attempt == 2:
                raise RuntimeError(
                    "Quota is still exhausted after retries. "
                    "Check AI Studio for the reset time. "
                    "Previously cached responses remain available."
                ) from None

            match = re.search(
                r"retry in\s+([\d.]+)\s*s", message, re.IGNORECASE
            )
            delay = float(match.group(1)) if match else 15 * (2 ** attempt)

            if delay > 60:
                raise RuntimeError(
                    "The provider requests a longer wait. "
                    "Check your quota reset time before retrying."
                ) from None

            delay += random.uniform(1, 2)
            print(f"[Quota reached. Retrying in {delay:.0f} seconds...]")
            time.sleep(delay)