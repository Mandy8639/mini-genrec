"""OpenAI calls, content-addressed caching, and fail-closed catalog validation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import unicodedata

from openai import OpenAI, RateLimitError

from .context import Prompt, response_format


def canonical(title: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


@dataclass
class Validated:
    ranking: list[int]
    suggestions: int
    outside_candidates: int
    outside_catalog: int
    duplicates: int


class RunPaused(Exception):
    """A request budget or provider quota stopped work; cached calls can be resumed."""

    def __init__(self, message, category="paused", http_status=None):
        super().__init__(message)
        self.category = category
        self.http_status = http_status


@dataclass
class RequestBudget:
    limit: int = 40
    interval: float = 3.1
    used: int = 0
    last_started: float = 0.0

    def acquire(self):
        if self.used >= self.limit:
            raise RunPaused(f"Reached this invocation's {self.limit}-request budget.", category="request_budget")
        delay = self.interval - (time.monotonic() - self.last_started)
        if self.used and delay > 0:
            time.sleep(delay)
        self.used += 1
        self.last_started = time.monotonic()


def validate(titles: list[str], movies, candidates: list[int],
             unconstrained: bool = False) -> Validated:
    if not isinstance(titles, list) or any(not isinstance(t, str) for t in titles):
        raise ValueError("ranking must be a list of title strings")
    # An ambiguous normalized title is rejected, never assigned to an arbitrary movie.
    groups = {}
    for mid, title in movies.title.items():
        groups.setdefault(canonical(str(title)), []).append(int(mid))
    lookup = {title: ids[0] for title, ids in groups.items() if len(ids) == 1}
    candidate_set = set(candidates)
    ranking, seen = [], set()
    outside_candidates = outside_catalog = duplicates = 0
    for title in titles:
        mid = lookup.get(canonical(title))
        outside_catalog += int(mid is None)
        outside_candidates += int(mid not in candidate_set)
        if mid is None or (not unconstrained and mid not in candidate_set):
            continue
        if mid in seen:
            duplicates += 1
            continue
        seen.add(mid)
        ranking.append(mid)
    return Validated(ranking, len(titles), outside_candidates, outside_catalog, duplicates)


class LLM:
    def __init__(self, model: str, cache: Path, backend: str = "openai",
                 budget: RequestBudget | None = None, reasoning: str = "default",
                 max_output_tokens: int = 2400):
        self.backend = backend
        self.reasoning = reasoning
        self.max_output_tokens = max_output_tokens
        if backend == "openrouter":
            if not model.endswith(":free") or model.startswith("openrouter/"):
                raise ValueError("OpenRouter runs require an explicit model ID ending in :free.")
            # An explicit key and base URL isolate this backend from OpenAI credentials/settings.
            self.client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                                 base_url="https://openrouter.ai/api/v1",
                                 timeout=60, max_retries=0)
            self.budget = budget if budget is not None else RequestBudget()
        elif backend == "openai":
            self.client = OpenAI(timeout=60, max_retries=2)
            self.budget = None
        else:
            raise ValueError(f"Unknown backend: {backend}")
        self.model, self.cache = model, cache
        self.cache.mkdir(parents=True, exist_ok=True)

    def call(self, prompt: Prompt, schema=None) -> dict:
        request = dict(model=self.model, messages=prompt.messages,
                       response_format=response_format() if schema is None else schema, temperature=0,
                       max_completion_tokens=self.max_output_tokens)
        if self.backend == "openrouter":
            request.pop("max_completion_tokens")
            request["max_tokens"] = self.max_output_tokens
            request["extra_body"] = {"provider": {"require_parameters": True}}
            if self.reasoning == "off":
                request["extra_body"]["reasoning"] = {"enabled": False}
        identity = {"backend": self.backend, "request": request}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        if self.budget:
            self.budget.acquire()
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**request)
        except RateLimitError:
            if self.backend == "openrouter":
                raise RunPaused("OpenRouter rate/quota limit reached; retry later with the same command.",
                                category="rate_limit", http_status=429) from None
            raise
        choice = response.choices[0]
        result = dict(content=choice.message.content,
                      refusal=bool(choice.message.refusal),
                      finish_reason=choice.finish_reason,
                      prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                      completion_tokens=response.usage.completion_tokens if response.usage else None,
                      model=response.model, request_hash=key, backend=self.backend,
                      response_id=response.id,
                      cost_usd=getattr(response.usage, "cost", None),
                      provider=getattr(response, "provider", None),
                      reasoning_tokens=getattr(getattr(response.usage, "completion_tokens_details", None),
                                               "reasoning_tokens", None),
                      latency_seconds=time.monotonic() - started)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        temporary.replace(path)
        return result


def parse_response(response: dict, item_type=str) -> list:
    if response["refusal"] or response["finish_reason"] != "stop":
        raise ValueError("refused or truncated response")
    parsed = json.loads(response["content"])
    if not isinstance(parsed, dict) or set(parsed) != {"ranking"}:
        raise ValueError("unexpected response structure")
    ranking = parsed["ranking"]
    if not isinstance(ranking, list) or any(type(item) is not item_type for item in ranking):
        raise ValueError("Invalid ranking item type.")
    return ranking
