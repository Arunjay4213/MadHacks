"""Azure AI Foundry clients: chat completions for the slow reader, the Anthropic Messages API for the rewriter.

Credentials come from the environment: FOUNDRY_ENDPOINT, FOUNDRY_KEY and FOUNDRY_DEPLOYMENT (on the engine
machine, ~/.madhacks-foundry.env). The slow reader's deployment defaults to gpt-4.1-nano, which answers in about
0.4 to 0.8 s from the engine machine. The rewriter uses FOUNDRY_REWRITE_DEPLOYMENT (default claude-sonnet-5-5)
on the same resource, through https://<resource>.services.ai.azure.com/anthropic/v1/messages, where <resource>
is the first label of FOUNDRY_ENDPOINT's host. FOUNDRY_ANTHROPIC_URL replaces that whole URL (a local fake
endpoint in tests).
"""

import logging
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

API_VERSION = "2024-10-21"
DEFAULT_DEPLOYMENT = "gpt-4.1-nano"
DEFAULT_REWRITE_DEPLOYMENT = "claude-sonnet-5-5"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class FoundrySettings:
    endpoint: str
    key: str
    deployment: str            # the slow reader's chat completions deployment
    rewrite_deployment: str    # the rewriter's Anthropic model deployment
    anthropic_url: str         # the rewriter's Messages API URL


def anthropic_url(endpoint: str) -> str:
    """https://<resource>.services.ai.azure.com/anthropic/v1/messages for any endpoint host of that resource
    (<resource>.cognitiveservices.azure.com, <resource>.openai.azure.com, <resource>.services.ai.azure.com)."""
    host = urlsplit(endpoint if "//" in endpoint else f"https://{endpoint}").hostname or ""
    resource = host.split(".")[0]
    if not resource:
        raise ValueError(f"cannot find the Foundry resource name in {endpoint!r}")
    return f"https://{resource}.services.ai.azure.com/anthropic/v1/messages"


def settings_from_env(environ=os.environ) -> FoundrySettings | None:
    """The FOUNDRY_* settings, or None when the endpoint or the key is missing (the caller logs what that means)."""
    endpoint, key = environ.get("FOUNDRY_ENDPOINT", "").strip(), environ.get("FOUNDRY_KEY", "").strip()
    if not endpoint or not key:
        return None
    deployment = environ.get("FOUNDRY_DEPLOYMENT", "").strip() or DEFAULT_DEPLOYMENT
    rewrite_deployment = environ.get("FOUNDRY_REWRITE_DEPLOYMENT", "").strip() or DEFAULT_REWRITE_DEPLOYMENT
    url = environ.get("FOUNDRY_ANTHROPIC_URL", "").strip() or anthropic_url(endpoint)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # each caller logs one line per call; no URL per request
    return FoundrySettings(endpoint, key, deployment, rewrite_deployment, url)


class FoundryChat:
    """One chat completions call per `complete(...)`; returns the model's message text.

    Raises httpx.HTTPError on a transport error, a timeout or a non-2xx status, and KeyError, IndexError or
    TypeError on a reply of the wrong shape. One httpx client keeps the TLS connection open between calls,
    which saves a handshake per call. httpx.Client is safe to share between threads.
    """

    def __init__(self, endpoint: str, key: str, deployment: str = DEFAULT_DEPLOYMENT, timeout_s: float = 2.0,
                 client: httpx.Client | None = None) -> None:
        self.deployment = deployment
        self._url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
        self._client = client or httpx.Client(timeout=timeout_s, headers={"api-key": key})

    def complete(self, messages: list[dict], max_tokens: int, json_object: bool = False) -> str:
        body = {"messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if json_object:
            body["response_format"] = {"type": "json_object"}
        r = self._client.post(self._url, params={"api-version": API_VERSION}, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class FoundryMessages:
    """One Anthropic Messages API call per `complete(system, messages)`; returns the reply's text.

    Request body: {"model", "max_tokens", "system", "messages", "thinking": {"type": "between_tools"},
    "output_config": {"effort": "low"}}. No temperature: the model rejects it. "between_tools" is how Claude
    Sonnet 5.5 runs with thinking off (it is accepted at effort high or below and takes no other field).
    The text is the content blocks of type "text", joined. A refusal (stop_reason "refusal") has no text, so it
    returns "" and the caller's validation rejects it.
    Raises httpx.HTTPError on a transport error, a timeout or a non-2xx status, and KeyError or TypeError on a
    reply of the wrong shape.
    """

    MAX_TOKENS = 1024

    def __init__(self, url: str, key: str, model: str = DEFAULT_REWRITE_DEPLOYMENT, timeout_s: float = 5.0,
                 client: httpx.Client | None = None) -> None:
        self.model = model
        self._url = url
        self._client = client or httpx.Client(
            timeout=timeout_s, headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION})

    @classmethod
    def from_settings(cls, settings: FoundrySettings, timeout_s: float) -> "FoundryMessages":
        return cls(settings.anthropic_url, settings.key, settings.rewrite_deployment, timeout_s=timeout_s)

    def body(self, system: str, messages: list[dict]) -> dict:
        return {"model": self.model, "max_tokens": self.MAX_TOKENS, "system": system, "messages": messages,
                "thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}

    def complete(self, system: str, messages: list[dict]) -> str:
        r = self._client.post(self._url, json=self.body(system, messages))
        r.raise_for_status()
        reply = r.json()
        return "".join(b["text"] for b in reply["content"] if b.get("type") == "text")
