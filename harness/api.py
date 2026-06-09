"""OpenRouter API client for RP-Bench."""
import json
import os
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .config import (
    OPENROUTER_BASE_URL,
    JUDGE_CONFIG,
    GENERATION_CONFIG,
    REQUEST_DELAY_SECONDS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    PROJECT_ROOT,
)

load_dotenv(PROJECT_ROOT / ".env")

_last_request_time = 0.0
# Thread-safe rate gate. Under concurrency the benchmark lowers _min_interval
# so aggregate request starts scale with worker count; 429s are still caught
# and retried below as the safety net.
_rate_lock = threading.Lock()
_min_interval = REQUEST_DELAY_SECONDS


def set_min_interval(seconds: float):
    """Set the minimum spacing between request starts (global, thread-safe).

    Typical use: set to REQUEST_DELAY_SECONDS / concurrency before a parallel
    run so N workers can keep ~N requests in flight.
    """
    global _min_interval
    _min_interval = max(0.0, seconds)


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key


def _rate_limit():
    # Hold the lock only to claim a start slot (spacing the bursts); the slow
    # HTTP call happens outside the lock so workers run concurrently.
    global _last_request_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < _min_interval:
            time.sleep(_min_interval - elapsed)
        _last_request_time = time.time()


def chat_completion(
    model: str,
    system_prompt: str,
    user_content: str,
    config: dict | None = None,
) -> dict:
    """Send a chat completion request to OpenRouter.

    Returns the parsed response dict with keys: content, model, usage, raw.
    """
    if config is None:
        config = GENERATION_CONFIG

    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rp-bench",
        "X-Title": "RP-Bench",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        **config,
    }

    for attempt in range(MAX_RETRIES):
        _rate_limit()
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if resp.status_code == 429:
                wait = RETRY_DELAY_SECONDS * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            # Under load OpenRouter sometimes returns a non-JSON body (HTML
            # error page, truncated stream). resp.json() then raises
            # JSONDecodeError — treat it like a transient error and retry
            # rather than letting it bubble up and drop the whole session.
            data = resp.json()

            # Reasoning models can leave content=None when their internal
            # thinking exhausts max_tokens. Treat that as an empty response
            # so callers can decide what to do (skip / retry with higher
            # budget) instead of crashing on .strip() downstream.
            content = data["choices"][0]["message"].get("content") or ""
            return {
                "content": content,
                "model": data.get("model", model),
                "usage": data.get("usage", {}),
                "raw": data,
            }

        except (httpx.HTTPStatusError, httpx.RequestError, KeyError,
                json.JSONDecodeError, IndexError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY_SECONDS * (attempt + 1)
                print(f"  Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"API request failed after {MAX_RETRIES} attempts: {e}")

    raise RuntimeError("Exhausted retries")


def generate_rp_response(
    model: str,
    system_prompt: str,
    conversation_context: str,
) -> dict:
    """Generate an RP response from a test model."""
    return chat_completion(model, system_prompt, conversation_context, GENERATION_CONFIG)


def judge_response(
    judge_model: str,
    judge_system_prompt: str,
    eval_payload: str,
) -> dict:
    """Send an evaluation payload to a judge model and parse the JSON scores."""
    result = chat_completion(
        judge_model, judge_system_prompt, eval_payload, JUDGE_CONFIG
    )

    # Try to parse JSON from the response
    content = result["content"].strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        # Strip ```json ... ```
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    try:
        scores = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                scores = json.loads(content[start:end])
            except json.JSONDecodeError:
                scores = {"parse_error": True, "raw_content": result["content"]}
        else:
            scores = {"parse_error": True, "raw_content": result["content"]}

    result["scores"] = scores
    return result
