"""The one place that calls Claude, so speed and fallback settings can't drift apart."""

import logging

import anthropic

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FAST_BETA = "fast-mode-2026-02-01"


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
