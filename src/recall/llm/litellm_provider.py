"""Any model LiteLLM supports, behind the LLM interface. Provider details stay in this file.

LiteLLM's PyPI package was compromised twice in 2026, so pyproject.toml pins one exact version and
uv.lock pins its hash. Check its security advisories before changing that pin.
"""

import base64
import json
import os
from collections.abc import Callable

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # No network call on import.

import litellm  # noqa: E402
import openai  # noqa: E402

from recall.llm.base import Completion, LLMError, Stop, ToolCall, ToolResult  # noqa: E402
from recall.logs import count, get_logger, measured  # noqa: E402

litellm.suppress_debug_info = True  # Its prints go to stdout, which carries the MCP protocol.

log = get_logger(__name__)

PHOTO_TYPE = "image/jpeg"  # Telegram converts every photo to JPEG.
RETRIES = 2
STOPS = {"tool_calls": Stop.TOOL_USE, "length": Stop.CUT_OFF, "content_filter": Stop.REFUSED}


class LiteLLMProvider:
    def __init__(self, complete: Callable = litellm.completion):
        self.complete = complete

    def _call(self, kind: str, model: str, **kwargs):
        """Make one call and record its usage."""
        with measured(log, "llm_call", kind=kind, model=model) as stats:
            try:
                response = self.complete(
                    model=model, num_retries=RETRIES, drop_params=True, **kwargs
                )
            except openai.OpenAIError as exc:  # LiteLLM's errors all extend these.
                raise LLMError(str(exc)) from exc
            stats["stop"] = response.choices[0].finish_reason
            usage = response.usage
            cached = getattr(usage, "cache_read_input_tokens", 0) or 0
            count(
                llm_calls=1,
                input_tokens=usage.prompt_tokens - cached,
                output_tokens=usage.completion_tokens,
                cache_read_tokens=cached,
            )
            return response.choices[0]

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
        choice = self._call(
            "chat",
            model,
            messages=[{"role": "system", "content": blocks}, *messages],
            tools=[_tool(tool) for tool in tools],
            reasoning_effort=effort,
            max_tokens=16000,
        )
        calls = choice.message.tool_calls or []
        return Completion(
            choice.message.content or "",
            STOPS.get(choice.finish_reason, Stop.DONE),
            [
                ToolCall(call.id, call.function.name, json.loads(call.function.arguments))
                for call in calls
            ],
            choice.message,
        )

    def ask_json(self, model: str, system: str, prompt: str, schema: dict) -> dict:
        choice = self._call(
            "json",
            model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
            reasoning_effort="low",
            max_tokens=4000,
        )
        if choice.finish_reason != "stop":
            raise LLMError(f"Stopped early: {choice.finish_reason}")
        try:
            return json.loads(choice.message.content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMError("Returned invalid JSON") from exc

    def user_message(self, text: str, image: bytes | None = None) -> dict:
        if image is None:
            return {"role": "user", "content": text}
        data = base64.standard_b64encode(image).decode()
        url = f"data:{PHOTO_TYPE};base64,{data}"
        content = [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text},
        ]
        return {"role": "user", "content": content}

    def assistant_message(self, completion: Completion) -> dict:
        return completion.raw.model_dump(exclude_none=True)

    def tool_results_message(self, results: list[ToolResult]) -> list[dict]:
        # This format has no error flag, so the wording has to tell the model a tool failed.
        return [
            {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": f"Error: {result.content}" if result.is_error else result.content,
            }
            for result in results
        ]


def _tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }
