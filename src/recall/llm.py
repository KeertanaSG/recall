"""The one place that calls Claude, so speed and fallback settings can't drift apart."""

import json
import logging

import anthropic

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FAST_BETA = "fast-mode-2026-02-01"


def ask_json(
    client: anthropic.Anthropic, model: str, fast: bool, system: str, prompt: str, schema: dict
) -> dict | None:
    """One quick call for structured output. Returns None on failure, so callers can carry on."""
    try:
        response = create_message(
            client,
            model,
            fast,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.APIError:
        log.exception("A structured model call failed, continuing without it")
        return None
    if response.stop_reason != "end_turn":
        log.warning("A structured model call stopped early: %s", response.stop_reason)
        return None
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("A structured model call returned invalid JSON")
        return None


def create_message(client: anthropic.Anthropic, model: str, fast: bool, **kwargs):
    # Haiku has no server-side fallback or fast mode, and the API rejects both for it.
    betas = []
    if "haiku" not in model:
        betas.append(FALLBACK_BETA)
        kwargs["fallbacks"] = "default"
    create = client.beta.messages.create
    if fast and "haiku" not in model:
        try:
            return create(model=model, speed="fast", betas=[*betas, FAST_BETA], **kwargs)
        except anthropic.RateLimitError:
            log.warning("Fast mode is rate limited, using standard speed")
    return create(model=model, betas=betas, **kwargs)
