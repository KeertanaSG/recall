"""Claude behind the LLM interface. Everything specific to Anthropic's API stays in this file."""

import base64
import json
import logging

import anthropic

from recall.llm.base import Completion, LLMError, Stop, ToolCall, ToolResult

log = logging.getLogger(__name__)

PHOTO_TYPE = "image/jpeg"  # Telegram converts every photo to JPEG.
STOPS = {"tool_use": Stop.TOOL_USE, "max_tokens": Stop.CUT_OFF, "refusal": Stop.REFUSED}

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


class ClaudeLLM:
    def __init__(self, client: anthropic.Anthropic, fast: bool = False):
        self.client = client
        self.fast = fast

    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        effort: str,
        note: str = "",
    ) -> Completion:
        blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if note:
            blocks.append({"type": "text", "text": note})
        try:
            response = create_message(
                self.client,
                model,
                self.fast,
                max_tokens=16000,
                system=blocks,
                tools=tools,
                messages=messages,
                output_config={"effort": effort},
            )
        except anthropic.APIError as exc:
            raise LLMError(str(exc)) from exc
        log.info(
            "model step: stop=%s input=%s output=%s",
            response.stop_reason,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return Completion(
            _text(response),
            STOPS.get(response.stop_reason, Stop.DONE),
            [
                ToolCall(block.id, block.name, block.input)
                for block in response.content
                if block.type == "tool_use"
            ],
            response.content,
        )

    def ask_json(self, model: str, system: str, prompt: str, schema: dict) -> dict:
        try:
            response = create_message(
                self.client,
                model,
                False,
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIError as exc:
            raise LLMError(str(exc)) from exc
        if response.stop_reason != "end_turn":
            raise LLMError(f"Stopped early: {response.stop_reason}")
        try:
            return json.loads(_text(response))
        except json.JSONDecodeError as exc:
            raise LLMError("Returned invalid JSON") from exc

    def user_message(self, text: str, image: bytes | None = None) -> dict:
        if image is None:
            return {"role": "user", "content": text}
        data = base64.standard_b64encode(image).decode()
        source = {"type": "base64", "media_type": PHOTO_TYPE, "data": data}
        content = [{"type": "image", "source": source}, {"type": "text", "text": text}]
        return {"role": "user", "content": content}

    def assistant_message(self, completion: Completion) -> dict:
        return {"role": "assistant", "content": completion.raw}

    def tool_results_message(self, results: list[ToolResult]) -> dict:
        content = []
        for result in results:
            block = {"type": "tool_result", "tool_use_id": result.call_id, "content": result.content}
            if result.is_error:
                block["is_error"] = True
            content.append(block)
        return {"role": "user", "content": content}


def _text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")
