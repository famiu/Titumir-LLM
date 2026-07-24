"""Shared LLM API helpers for generate_dataset.py and refine_dataset.py."""

import json
import time

import requests
from dotenv import load_dotenv

from scripts._data import strip_outer_json_fence
from training.config import ApiConfigBase

load_dotenv()

RETRY_BASE_DELAY = 2
RETRY_MAX_DELAY = 120


def retry_delay(attempt: int) -> float:
    """Exponential backoff: base * 2^attempt, capped at max."""
    return min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)


def call_llm(
    llm_cfg: ApiConfigBase,
    messages: list[dict],
    expected_type: type[dict] | type[list] | None = None,
) -> dict | list | None:
    """Make an LLM API call with automatic retries. Returns parsed JSON or None on failure."""
    api_key = llm_cfg.get_api_key()
    if not api_key:
        raise ValueError(f"API key not found: set {llm_cfg.api_key_env} environment variable")

    for attempt in range(llm_cfg.max_retries):
        try:
            payload = {
                "model": llm_cfg.model,
                "messages": messages,
                "temperature": llm_cfg.temperature,
                "max_tokens": llm_cfg.max_tokens,
            }
            if llm_cfg.reasoning_effort is not None:
                payload["reasoning"] = {"effort": llm_cfg.reasoning_effort}

            response = requests.post(
                llm_cfg.endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=llm_cfg.batch_timeout,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            result = json.loads(strip_outer_json_fence(raw))
            if expected_type is not None and not isinstance(result, expected_type):
                raise ValueError(f"Expected {expected_type.__name__} response, got {type(result).__name__}")
            return result

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(retry_delay(attempt + 1))
            elif e.response.status_code >= 500:
                time.sleep(retry_delay(attempt))
            else:
                detail = e.response.text[:500].strip()
                print(f"LLM request failed with HTTP {e.response.status_code}: {detail}")
                return None
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as e:
            if attempt == llm_cfg.max_retries - 1:
                print(f"LLM response failed validation after {llm_cfg.max_retries} attempts: {e}")
            time.sleep(retry_delay(attempt))
        except requests.RequestException as e:
            print(f"LLM request failed: {e}")
            return None

    return None
