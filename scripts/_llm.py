"""Shared LLM API helpers for generate_dataset.py and refine_dataset.py."""

import json
import time

import requests
from dotenv import load_dotenv

from training.config import ApiConfigBase

load_dotenv()

RETRY_BASE_DELAY = 2
RETRY_MAX_DELAY = 120


def retry_delay(attempt: int) -> float:
    """Exponential backoff: base * 2^attempt, capped at max."""
    return min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)


class SafeDict(dict):
    """Dict that returns {key} for missing keys, for safe str.format_map."""

    def __missing__(self, key):
        return "{" + key + "}"


def call_llm(llm_cfg: ApiConfigBase, messages: list[dict]) -> dict | list | None:
    """Make an LLM API call with automatic retries. Returns parsed JSON or None on failure."""
    api_key = llm_cfg.get_api_key()
    if not api_key:
        raise ValueError(f"API key not found: set {llm_cfg.api_key_env} environment variable")

    for attempt in range(llm_cfg.max_retries):
        try:
            response = requests.post(
                llm_cfg.endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_cfg.model,
                    "messages": messages,
                    "temperature": llm_cfg.temperature,
                    "max_tokens": llm_cfg.max_tokens,
                    "reasoning": {"effort": "none"},
                },
                timeout=llm_cfg.batch_timeout,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(retry_delay(attempt + 1))
            elif e.response.status_code >= 500:
                time.sleep(retry_delay(attempt))
            else:
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, json.JSONDecodeError):
            time.sleep(retry_delay(attempt))
        except Exception:
            return None

    return None
