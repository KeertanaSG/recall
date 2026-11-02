"""Claude behind the LLM interface. Everything specific to Anthropic's API stays in this file."""

import base64
import json

import anthropic

from recall.llm.base import Completion, LLMError, Stop, ToolCall, ToolResult
from recall.logs import count, get_logger, measured

log = get_logger(__name__)

PHOTO_TYPE = "image/jpeg"  # Telegram converts every photo to JPEG.
STOPS = {"tool_use": Stop.TOOL_USE, "max_tokens": Stop.CUT_OFF, "refusal": Stop.REFUSED}

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def create_message(client: anthropic.Anthropic, model: str, **kwargs):
    if "haiku" in model:  # Haiku has no server-side fallback, and the API rejects it.
        return client.beta.messages.create(model=model, betas=[], **kwargs)
    return client.beta.messages.create(
        model=model, betas=[FALLBACK_BETA], fallbacks="default", **kwargs
    )


class ClaudeLLM:
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

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
        with measured(log, "llm_call", kind="chat", model=model) as stats:
            try:
                response = create_message(
                    self.client,
                    model,
                    max_tokens=16000,
                    system=blocks,
                    tools=tools,
                    messages=messages,
                    output_config={"effort": effort},
                )
            except anthropic.APIError as exc:
                raise LLMError(str(exc)) from exc
            _record(stats, response)
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
        with measured(log, "llm_call", kind="json", model=model) as stats:
            try:
                response = create_message(
                    self.client,
                    model,
                    max_tokens=4000,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={
                        "effort": "low",
                        "format": {"type": "json_schema", "schema": schema},
                    },
                )
            except anthropic.APIError as exc:
                raise LLMError(str(exc)) from exc
            _record(stats, response)
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
            block = {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": result.content,
            }
            if result.is_error:
                block["is_error"] = True
            content.append(block)
        return {"role": "user", "content": content}


def _record(stats: dict, response) -> None:
    stats["stop"] = response.stop_reason
    usage = getattr(response, "usage", None)
    if usage:
        count(
            llm_calls=1,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )


def _text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")
