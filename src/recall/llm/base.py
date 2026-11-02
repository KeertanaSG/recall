"""What Recall needs from a language model. Nothing outside recall/llm knows which one it is.

Messages are opaque to callers: a provider builds them with user_message, assistant_message and
tool_results_message, so it can keep whatever its API needs in order to replay a conversation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class LLMError(Exception):
    """The model couldn't be reached or gave no usable answer."""


class Stop(Enum):
    DONE = "done"
    TOOL_USE = "tool_use"
    CUT_OFF = "cut_off"
    REFUSED = "refused"


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Completion:
    text: str
    stop: Stop
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # The provider's own reply, for assistant_message.


class LLM(Protocol):
    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        effort: str,
        note: str = "",
    ) -> Completion:
        """One step of a conversation. `note` is extra system text that changes between calls."""
        ...

    def ask_json(self, model: str, system: str, prompt: str, schema: dict) -> dict:
        """One structured answer matching a JSON schema. Raises LLMError on failure."""
        ...

    def user_message(self, text: str, image: bytes | None = None) -> dict: ...

    def assistant_message(self, completion: Completion) -> dict: ...

    def tool_results_message(self, results: list[ToolResult]) -> dict | list[dict]: ...
