"""One small wrapper over Anthropic, OpenAI, and Gemini. SDKs are imported lazily."""
from __future__ import annotations

import json
import re

from . import config

PROVIDERS = ("anthropic", "openai", "gemini")
PROVIDER_LABELS = {"anthropic": "Anthropic (Claude)", "openai": "OpenAI", "gemini": "Google Gemini"}

# Claude models that take `output_config.effort` and the server-side refusal fallback
# (`fallbacks="default"`: if a safety classifier declines, the API re-runs the request on a fallback model).
ANTHROPIC_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"

TIMEOUT_SECONDS = 300   # per request; a long memo takes a few minutes at most
MAX_RETRIES = 1         # SDK-level retries on 429 / 5xx / connection errors (worst case ~20 min per role, counting the JSON retry)
JSON_RETRY_NOTE = "Your previous reply was not valid JSON. Return only the JSON object."

_SDK_PACKAGES = {"anthropic": "anthropic", "openai": "openai", "gemini": "google-genai"}


class LLMError(Exception):
    """A user-facing LLM failure (bad key, rate limit, refusal, unparseable output).

    `raw` holds the model's text when the failure was unusable output, for debugging.
    """

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


def available_providers() -> list[str]:
    """Providers whose API key environment variable is set, in PROVIDERS order."""
    return [p for p in PROVIDERS if config.get_api_key(p)]


class LLMClient:
    """A single-provider client: `complete` returns text, `complete_json` returns a dict."""

    def __init__(self, provider: str, model: str | None = None):
        if provider not in PROVIDERS:
            raise LLMError(f"Unknown LLM provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
        self.provider = provider
        self.model = (model or "").strip() or config.DEFAULT_MODELS[provider]
        self.label = f"{provider} / {self.model}"
        api_key = config.get_api_key(provider)
        if not api_key:
            env_names = " or ".join(config.API_KEY_ENVS[provider])
            raise LLMError(f"No API key for {PROVIDER_LABELS[provider]}. Set {env_names} (e.g. in .env).")
        try:
            self._client = _make_sdk_client(provider, api_key)
        except ImportError as e:
            raise LLMError(f"The {_SDK_PACKAGES[provider]} package is not installed. "
                           f"Run: pip install {_SDK_PACKAGES[provider]}") from e

    def complete(self, system: str, user: str, *, max_tokens: int = 16000, json_mode: bool = False) -> str:
        """One model call. Raises LLMError on API errors, refusals, truncation, or empty output."""
        if self.provider == "anthropic":
            text = self._complete_anthropic(system, user, max_tokens)
        elif self.provider == "openai":
            text = self._complete_openai(system, user, max_tokens, json_mode)
        else:
            text = self._complete_gemini(system, user, max_tokens, json_mode)
        if not text.strip():
            raise LLMError(f"{self.label} returned an empty response.")
        return text

    def complete_json(self, system: str, user: str, *, max_tokens: int = 16000) -> dict:
        """Like `complete`, but parses a JSON object. Retries once if the reply is not valid JSON."""
        text = self.complete(system, user, max_tokens=max_tokens, json_mode=True)
        try:
            return parse_json_response(text)
        except ValueError:
            pass
        text = self.complete(system, f"{user}\n\n{JSON_RETRY_NOTE}", max_tokens=max_tokens, json_mode=True)
        try:
            return parse_json_response(text)
        except ValueError:
            raise LLMError(f"{self.label} did not return valid JSON, even after a retry.", raw=text) from None

    # --- providers ---------------------------------------------------------------------------

    def _complete_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        import anthropic

        request = {"model": self.model, "max_tokens": max_tokens, "system": system,
                   "messages": [{"role": "user", "content": user}]}
        try:
            if self.model in ANTHROPIC_FALLBACK_MODELS:
                # Thinking is adaptive by default on these models; effort sets how hard the model works.
                # Sampling params (temperature/top_p) are rejected by these models, so none are sent.
                response = self._client.beta.messages.create(
                    **request,
                    output_config={"effort": config.ANTHROPIC_EFFORT},
                    betas=[ANTHROPIC_FALLBACK_BETA],
                    fallbacks="default",
                )
            else:
                response = self._client.messages.create(**request)
        except anthropic.APIStatusError as e:
            raise LLMError(self._http_error_message(e.status_code, _api_message(e))) from e
        except anthropic.APITimeoutError as e:
            raise LLMError(self._timeout_message()) from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Could not reach the Anthropic API ({e}). Check your connection and retry.") from e

        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            reason = f" (category: {category})" if category else ""
            raise LLMError(f"{self.label} declined this request{reason}. Try again or switch provider.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "max_tokens":
            raise LLMError(self._truncated_message(max_tokens), raw=text)
        return text

    def _complete_openai(self, system: str, user: str, max_tokens: int, json_mode: bool) -> str:
        import openai

        request = {"model": self.model, "instructions": system, "input": user,
                   "max_output_tokens": max_tokens, "store": False}
        if json_mode:
            # JSON mode requires the word "JSON" somewhere in the prompt or the API rejects the request.
            if "json" not in (system + user).lower():
                request["instructions"] = f"{system}\n\nRespond with a JSON object."
            request["text"] = {"format": {"type": "json_object"}}
        try:
            response = self._client.responses.create(**request)
        except openai.APIStatusError as e:
            raise LLMError(self._http_error_message(e.status_code, _api_message(e))) from e
        except openai.APITimeoutError as e:
            raise LLMError(self._timeout_message()) from e
        except openai.APIConnectionError as e:
            raise LLMError(f"Could not reach the OpenAI API ({e}). Check your connection and retry.") from e

        text = response.output_text or ""
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            if reason == "max_output_tokens":
                raise LLMError(self._truncated_message(max_tokens), raw=text)
            raise LLMError(f"{self.label} stopped early (reason: {reason}).", raw=text)
        if not text:
            for item in response.output:
                for part in getattr(item, "content", None) or []:
                    if part.type == "refusal":
                        raise LLMError(f"{self.label} declined this request: {part.refusal}")
        return text

    def _complete_gemini(self, system: str, user: str, max_tokens: int, json_mode: bool) -> str:
        import httpx
        from google.genai import errors, types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        try:
            response = self._client.models.generate_content(model=self.model, contents=user, config=cfg)
        except errors.APIError as e:
            raise LLMError(self._http_error_message(e.code, e.message)) from e
        except httpx.TimeoutException as e:
            raise LLMError(self._timeout_message()) from e
        except httpx.TransportError as e:
            raise LLMError(f"Could not reach the Gemini API ({e}). Check your connection and retry.") from e

        blocked = response.prompt_feedback.block_reason if response.prompt_feedback else None
        if blocked:
            raise LLMError(f"{self.label} blocked the prompt ({getattr(blocked, 'value', blocked)}).")
        text = response.text or ""
        finish = response.candidates[0].finish_reason if response.candidates else None
        finish = getattr(finish, "value", finish)  # enum -> plain string, e.g. "STOP", "SAFETY"
        if finish == "MAX_TOKENS":
            raise LLMError(self._truncated_message(max_tokens), raw=text)
        if not text and finish not in (None, "STOP"):
            raise LLMError(f"{self.label} returned no text (finish reason: {finish}).")
        return text

    # --- error messages ----------------------------------------------------------------------

    def _http_error_message(self, status: int | None, detail: str | None) -> str:
        detail = (detail or "").strip()[:300]
        if "credit balance" in detail.lower():  # Anthropic answers an empty prepaid balance with HTTP 400
            return f"{self.label}: the API account has no credit left (HTTP {status}). Add credits or switch provider."
        if status == 401:
            env_names = " or ".join(config.API_KEY_ENVS[self.provider])
            return f"{self.label}: the API key was rejected (HTTP 401). Check {env_names}."
        if status == 403:
            return (f"{self.label}: permission denied (HTTP 403). "
                    f"The key may not have access to this model. {detail}")
        if status == 404:
            return (f"{self.label}: model '{self.model}' was not found or is not available to this key (HTTP 404). "
                    f"Pick another model under Settings → Model or set {self.provider.upper()}_MODEL.")
        if status == 429:
            return (f"{self.label}: rate limit or quota exceeded (HTTP 429). "
                    "Wait a minute and retry, or check your plan's limits.")
        if status is not None and status >= 500:
            return f"{self.label}: the provider is temporarily unavailable (HTTP {status}). Try again shortly."
        return f"{self.label}: request failed (HTTP {status}). {detail}"

    def _timeout_message(self) -> str:
        return f"{self.label} timed out after {TIMEOUT_SECONDS} s without a reply. Try again, or pick a faster model."

    def _truncated_message(self, max_tokens: int) -> str:
        return f"{self.label} hit the {max_tokens:,}-token output limit before finishing; the reply was cut off."


def _api_message(e) -> str:
    """The API's own error message from an Anthropic/OpenAI status error.

    The SDKs' e.message is "Error code: 400 - {...}", a dump of the whole response body (including a
    per-request id), so read body["error"]["message"] (OpenAI already passes the inner error as body).
    """
    body = getattr(e, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return getattr(e, "message", "") or str(e)


def _make_sdk_client(provider: str, api_key: str):
    """Build the provider SDK client (lazy import: a missing SDK only matters if you pick that provider)."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
    if provider == "openai":
        import openai
        return openai.OpenAI(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
    from google import genai
    from google.genai import types
    # google-genai does not retry by default; attempts includes the first call.
    http_options = types.HttpOptions(timeout=TIMEOUT_SECONDS * 1000,
                                     retry_options=types.HttpRetryOptions(attempts=MAX_RETRIES + 1))
    return genai.Client(api_key=api_key, http_options=http_options)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_response(text: str) -> dict:
    """Parse a JSON object from model output: plain, ```json fenced, or wrapped in prose. Raises ValueError."""
    text = (text or "").strip()
    candidates = [text]
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate, strict=False)  # strict=False tolerates raw newlines inside strings
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output is not a JSON object.")
