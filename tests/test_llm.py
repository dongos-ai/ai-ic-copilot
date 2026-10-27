"""Offline tests for copilot.llm: no network, no real API keys (SDK clients are replaced with fakes)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from copilot import config
from copilot.llm import (ANTHROPIC_FALLBACK_BETA, JSON_RETRY_NOTE, MAX_RETRIES, TIMEOUT_SECONDS, LLMClient, LLMError,
                         available_providers, parse_json_response)

ALL_KEY_ENVS = [name for names in config.API_KEY_ENVS.values() for name in names]


@pytest.fixture
def no_keys(monkeypatch):
    for name in ALL_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class Recorder:
    """Stands in for an SDK method: records kwargs and returns (or raises) a canned value."""

    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_client(no_keys, provider, model=None):
    env = config.API_KEY_ENVS[provider][0]
    no_keys.setenv(env, "test-key-not-real")
    return LLMClient(provider, model)


# --- parse_json_response -----------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', {"a": 1}),
    ('  \n{"a": 1}\n', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```JSON\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": {"b": [1, 2]}}\n```', {"a": {"b": [1, 2]}}),
    ('Here is the memo:\n```json\n{"a": 1}\n```\nLet me know!', {"a": 1}),
    ('Sure! {"a": "x"} Hope this helps.', {"a": "x"}),
    ('{"a": "line one\nline two"}', {"a": "line one\nline two"}),  # raw newline inside a string
    ('{"text": "uses } and { inside"}', {"text": "uses } and { inside"}),
])
def test_parse_json_response_accepts_common_shapes(text, expected):
    assert parse_json_response(text) == expected


@pytest.mark.parametrize("text", ["", "   ", None, "no json here", "[1, 2, 3]", '"just a string"',
                                  '{"a": 1', "```json\n```", "{not: valid}"])
def test_parse_json_response_rejects_non_objects(text):
    with pytest.raises(ValueError):
        parse_json_response(text)


# --- available_providers / constructor ---------------------------------------------------------

def test_available_providers_none_set(no_keys):
    assert available_providers() == []


def test_available_providers_in_fixed_order(no_keys):
    no_keys.setenv("GEMINI_API_KEY", "g")
    no_keys.setenv("ANTHROPIC_API_KEY", "a")
    assert available_providers() == ["anthropic", "gemini"]


def test_available_providers_google_api_key_and_blank_values(no_keys):
    no_keys.setenv("GOOGLE_API_KEY", "g")
    no_keys.setenv("OPENAI_API_KEY", "   ")
    assert available_providers() == ["gemini"]


def test_client_without_key_raises_readable_error(no_keys):
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        LLMClient("anthropic")


def test_unknown_provider_raises(no_keys):
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        LLMClient("llama")


def test_a_stalled_provider_fails_within_minutes():
    # Worst case per role is (1 + MAX_RETRIES) x TIMEOUT_SECONDS: 10 minutes, not half an hour.
    assert MAX_RETRIES <= 1 and TIMEOUT_SECONDS <= 300


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_client_uses_default_model_and_label(no_keys, provider):
    client = make_client(no_keys, provider)
    assert client.model == config.DEFAULT_MODELS[provider]
    assert client.label == f"{provider} / {client.model}"
    assert make_client(no_keys, provider, "  custom-model ").model == "custom-model"


# --- Anthropic -----------------------------------------------------------------------------------

def anthropic_response(text="hello", stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=stop_details,
        content=[SimpleNamespace(type="thinking", thinking="", signature="s"),
                 SimpleNamespace(type="text", text=text[:3]), SimpleNamespace(type="text", text=text[3:])])


def fake_anthropic(client, result):
    beta, plain = Recorder(result), Recorder(result)
    client._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=beta)),
                                     messages=SimpleNamespace(create=plain))
    return beta, plain


def test_anthropic_fallback_models_use_beta_call_shape(no_keys):
    client = make_client(no_keys, "anthropic", "claude-opus-5-5")
    beta, plain = fake_anthropic(client, anthropic_response("hello world"))
    assert client.complete("SYS", "USER", max_tokens=1234) == "hello world"  # only text blocks, joined
    assert plain.kwargs is None
    assert beta.kwargs == {
        "model": "claude-opus-5-5", "max_tokens": 1234, "system": "SYS",
        "messages": [{"role": "user", "content": "USER"}],
        "output_config": {"effort": config.ANTHROPIC_EFFORT},
        "betas": [ANTHROPIC_FALLBACK_BETA], "fallbacks": "default",
    }
    assert "temperature" not in beta.kwargs and "thinking" not in beta.kwargs


def test_anthropic_other_models_use_plain_call(no_keys):
    client = make_client(no_keys, "anthropic", "claude-haiku-4-5")
    beta, plain = fake_anthropic(client, anthropic_response("ok!"))
    assert client.complete("SYS", "USER") == "ok!"
    assert beta.kwargs is None
    assert set(plain.kwargs) == {"model", "max_tokens", "system", "messages"}


def test_anthropic_refusal_raises(no_keys):
    client = make_client(no_keys, "anthropic")
    fake_anthropic(client, anthropic_response("", "refusal", SimpleNamespace(category="cyber")))
    with pytest.raises(LLMError, match="declined.*cyber"):
        client.complete("SYS", "USER")


def test_anthropic_truncation_raises_with_raw(no_keys):
    client = make_client(no_keys, "anthropic")
    fake_anthropic(client, anthropic_response('{"partial": ', "max_tokens"))
    with pytest.raises(LLMError, match="output limit") as info:
        client.complete("SYS", "USER")
    assert info.value.raw == '{"partial": '


def test_anthropic_empty_text_raises(no_keys):
    client = make_client(no_keys, "anthropic")
    fake_anthropic(client, SimpleNamespace(stop_reason="end_turn", stop_details=None, content=[]))
    with pytest.raises(LLMError, match="empty"):
        client.complete("SYS", "USER")


@pytest.mark.parametrize("status, pattern", [(401, "API key was rejected.*ANTHROPIC_API_KEY"),
                                             (429, "rate limit"), (404, "not found.*ANTHROPIC_MODEL"),
                                             (529, "temporarily unavailable")])
def test_anthropic_status_errors_are_readable(no_keys, status, pattern):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")  # anthropic 1.x's HTTP library
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    client = make_client(no_keys, "anthropic")
    fake_anthropic(client, anthropic.APIStatusError("boom", response=response, body=None))
    with pytest.raises(LLMError, match=pattern):
        client.complete("SYS", "USER")


def anthropic_400(message: str):
    """A BadRequestError shaped like the SDK's: e.message dumps the whole body, request_id included."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message},
            "request_id": "req_011CabcDEF"}
    return anthropic.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)


def test_anthropic_empty_credit_balance_says_so(no_keys):
    client = make_client(no_keys, "anthropic", "claude-opus-5-5")
    fake_anthropic(client, anthropic_400("Your credit balance is too low to access the Anthropic API. "
                                         "Please go to Plans & Billing to upgrade or purchase credits."))
    with pytest.raises(LLMError) as info:
        client.complete("SYS", "USER")
    assert str(info.value) == ("anthropic / claude-opus-5-5: the API account has no credit left (HTTP 400). "
                               "Add credits or switch provider.")


def test_anthropic_other_400_shows_only_the_api_message(no_keys):
    client = make_client(no_keys, "anthropic", "claude-opus-5-5")
    fake_anthropic(client, anthropic_400("max_tokens: 99999999 > 128000, the maximum allowed"))
    with pytest.raises(LLMError) as info:
        client.complete("SYS", "USER")
    assert str(info.value) == ("anthropic / claude-opus-5-5: request failed (HTTP 400). "
                               "max_tokens: 99999999 > 128000, the maximum allowed")


def test_anthropic_connection_error_is_readable(no_keys):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    client = make_client(no_keys, "anthropic")
    fake_anthropic(client, error)
    with pytest.raises(LLMError, match="Could not reach the Anthropic API"):
        client.complete("SYS", "USER")


def test_anthropic_timeout_says_how_long_it_waited(no_keys):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    error = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    client = make_client(no_keys, "anthropic")
    fake_anthropic(client, error)
    with pytest.raises(LLMError, match=f"timed out after {TIMEOUT_SECONDS} s"):
        client.complete("SYS", "USER")


# --- OpenAI --------------------------------------------------------------------------------------

def openai_response(text="hello", status="completed", reason=None, output=None):
    return SimpleNamespace(output_text=text, status=status, output=output or [],
                           incomplete_details=SimpleNamespace(reason=reason) if reason else None)


def fake_openai(client, result):
    create = Recorder(result)
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return create


def test_openai_call_shape_json_mode(no_keys):
    client = make_client(no_keys, "openai")
    create = fake_openai(client, openai_response('{"a": 1}'))
    assert client.complete("Reply in JSON.", "USER", max_tokens=99, json_mode=True) == '{"a": 1}'
    assert create.kwargs == {"model": config.DEFAULT_MODELS["openai"], "instructions": "Reply in JSON.",
                             "input": "USER", "max_output_tokens": 99, "store": False,
                             "text": {"format": {"type": "json_object"}}}


def test_openai_json_mode_adds_the_word_json_when_missing(no_keys):
    client = make_client(no_keys, "openai")
    create = fake_openai(client, openai_response("{}x"))
    client.complete("Be terse.", "USER", json_mode=True)
    assert "JSON" in create.kwargs["instructions"]
    client.complete("Be terse.", "USER")
    assert "text" not in create.kwargs and create.kwargs["instructions"] == "Be terse."


def test_openai_incomplete_and_refusal(no_keys):
    client = make_client(no_keys, "openai")
    fake_openai(client, openai_response('{"a"', "incomplete", "max_output_tokens"))
    with pytest.raises(LLMError, match="output limit"):
        client.complete("SYS", "USER")
    refusal = SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="Not allowed.")])
    fake_openai(client, openai_response("", output=[refusal]))
    with pytest.raises(LLMError, match="declined this request: Not allowed"):
        client.complete("SYS", "USER")


def test_openai_status_error_is_readable(no_keys):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    client = make_client(no_keys, "openai")
    fake_openai(client, openai.AuthenticationError("bad key", response=response, body=None))
    with pytest.raises(LLMError, match="API key was rejected.*OPENAI_API_KEY"):
        client.complete("SYS", "USER")


def test_openai_400_shows_only_the_api_message(no_keys):
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
    body = {"message": "Unsupported parameter: 'temperature'.", "type": "invalid_request_error", "code": None}
    client = make_client(no_keys, "openai")
    # The SDK passes the inner "error" object as body but still dumps the whole response into e.message.
    fake_openai(client, openai.BadRequestError(f"Error code: 400 - {{'error': {body}}}", response=response, body=body))
    with pytest.raises(LLMError) as info:
        client.complete("SYS", "USER")
    assert str(info.value) == (f"openai / {client.model}: request failed (HTTP 400). "
                               "Unsupported parameter: 'temperature'.")


def test_openai_timeout_says_how_long_it_waited(no_keys):
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")  # openai 3.x's HTTP library
    client = make_client(no_keys, "openai")
    fake_openai(client, openai.APITimeoutError(request=httpx2.Request("POST", "https://api.openai.com/v1/responses")))
    with pytest.raises(LLMError, match=f"timed out after {TIMEOUT_SECONDS} s"):
        client.complete("SYS", "USER")


# --- Gemini --------------------------------------------------------------------------------------

def gemini_response(text="hello", finish="STOP", block_reason=None):
    candidates = [SimpleNamespace(finish_reason=finish)] if finish else []
    feedback = SimpleNamespace(block_reason=block_reason) if block_reason else None
    return SimpleNamespace(text=text, candidates=candidates, prompt_feedback=feedback)


def fake_gemini(client, result):
    generate = Recorder(result)
    client._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    return generate


def test_gemini_call_shape(no_keys):
    client = make_client(no_keys, "gemini")
    generate = fake_gemini(client, gemini_response('{"a": 1}'))
    assert client.complete("SYS", "USER", max_tokens=77, json_mode=True) == '{"a": 1}'
    assert generate.kwargs["model"] == config.DEFAULT_MODELS["gemini"]
    assert generate.kwargs["contents"] == "USER"
    cfg = generate.kwargs["config"]
    assert (cfg.system_instruction, cfg.max_output_tokens, cfg.response_mime_type) == ("SYS", 77, "application/json")
    client.complete("SYS", "USER")
    assert generate.kwargs["config"].response_mime_type is None


def test_gemini_blocked_truncated_and_safety(no_keys):
    client = make_client(no_keys, "gemini")
    fake_gemini(client, gemini_response(None, finish=None, block_reason="PROHIBITED_CONTENT"))
    with pytest.raises(LLMError, match="blocked the prompt"):
        client.complete("SYS", "USER")
    fake_gemini(client, gemini_response('{"a', finish="MAX_TOKENS"))
    with pytest.raises(LLMError, match="output limit"):
        client.complete("SYS", "USER")
    fake_gemini(client, gemini_response(None, finish="SAFETY"))
    with pytest.raises(LLMError, match="SAFETY"):
        client.complete("SYS", "USER")


def test_gemini_api_error_is_readable(no_keys):
    errors = pytest.importorskip("google.genai.errors")
    client = make_client(no_keys, "gemini")
    fake_gemini(client, errors.ClientError(429, {"error": {"code": 429, "message": "quota",
                                                           "status": "RESOURCE_EXHAUSTED"}}))
    with pytest.raises(LLMError, match="rate limit"):
        client.complete("SYS", "USER")


def test_gemini_timeout_says_how_long_it_waited(no_keys):
    httpx = pytest.importorskip("httpx")
    client = make_client(no_keys, "gemini")
    fake_gemini(client, httpx.ReadTimeout("timed out"))
    with pytest.raises(LLMError, match=f"timed out after {TIMEOUT_SECONDS} s"):
        client.complete("SYS", "USER")


# --- complete_json -------------------------------------------------------------------------------

class ScriptedClient(LLMClient):
    """LLMClient whose `complete` returns scripted replies (exercises the real complete_json)."""

    def __init__(self, replies):
        self.provider, self.model, self.label = "fake", "fake-model", "fake / fake-model"
        self.replies = list(replies)
        self.users = []

    def complete(self, system, user, *, max_tokens=16000, json_mode=False):
        assert json_mode
        self.users.append(user)
        return self.replies.pop(0)


def test_complete_json_parses_fenced_output():
    client = ScriptedClient(['```json\n{"ok": true}\n```'])
    assert client.complete_json("SYS", "USER") == {"ok": True}
    assert len(client.users) == 1


def test_complete_json_retries_once_after_invalid_json():
    client = ScriptedClient(["Sure, here is my analysis in prose.", '{"ok": true}'])
    assert client.complete_json("SYS", "USER") == {"ok": True}
    assert client.users[1].startswith("USER") and client.users[1].endswith(JSON_RETRY_NOTE)


def test_complete_json_gives_up_after_one_retry():
    client = ScriptedClient(["nope", "still nope", '{"never": "reached"}'])
    with pytest.raises(LLMError, match="valid JSON") as info:
        client.complete_json("SYS", "USER")
    assert info.value.raw == "still nope"
    assert len(client.users) == 2
