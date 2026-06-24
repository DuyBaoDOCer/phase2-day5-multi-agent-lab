"""LLM client abstraction — Mistral AI implementation.

Production note: agents should depend on this interface instead of importing an SDK directly.
Retry, timeout, and token tracking live here so agents stay clean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# Mistral pricing per 1M tokens (as of mid-2024, mistral-small-latest)
_COST_PER_1M_INPUT = 0.20   # USD
_COST_PER_1M_OUTPUT = 0.60  # USD


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Mistral AI LLM client with retry, timeout, and token logging."""

    def __init__(self, api_key: str, model: str = "mistral-small-latest") -> None:
        from mistralai import Mistral  # imported lazily to avoid hard dep at import time

        self._client = Mistral(api_key=api_key)
        self._model = model

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Internal call with retry decorator applied."""
        response = self._client.chat.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if response is None or not response.choices:
            raise AgentExecutionError("Mistral returned empty response")

        content = response.choices[0].message.content or ""
        in_tok = response.usage.prompt_tokens if response.usage else None
        out_tok = response.usage.completion_tokens if response.usage else None

        cost: float | None = None
        if in_tok is not None and out_tok is not None:
            cost = (in_tok / 1_000_000) * _COST_PER_1M_INPUT + (out_tok / 1_000_000) * _COST_PER_1M_OUTPUT

        logger.debug("LLM call: model=%s in=%s out=%s cost=$%.6f", self._model, in_tok, out_tok, cost or 0)
        return LLMResponse(content=content, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry and token tracking.

        Raises AgentExecutionError after all retry attempts are exhausted.
        """
        try:
            return self._call(system_prompt, user_prompt)
        except AgentExecutionError:
            raise
        except Exception as exc:
            logger.error("LLMClient.complete failed after retries: %s", exc)
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc
