"""JSON-compatible integration contract; no model/server-specific parameters."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


RequestStatus = Literal["success", "failed", "timeout", "cancelled"]


class TimingMetadata(TypedDict):
    queue_entry_time: str
    execution_start_time: str | None
    execution_finish_time: str
    queue_time_ms: float
    api_time_ms: float
    total_time_ms: float


class InferenceResponse(TypedDict):
    request_id: str
    task_id: str
    step_id: int
    response: str | None
    status: RequestStatus
    worker_id: int | None
    timing_metadata: TimingMetadata
    token_usage: dict[str, int]
    error: str | None


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    task_id: str
    step_id: int
    prompt: str | None = None
    generation_settings: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "task_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Provide exactly one of prompt or messages")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string")
        if self.messages is not None:
            if not isinstance(self.messages, list) or not self.messages:
                raise ValueError("messages must be a non-empty list")
            for message in self.messages:
                if not isinstance(message, dict):
                    raise ValueError("Each message must be a dictionary")
                role = message.get("role")
                if not isinstance(role, str) or not role.strip():
                    raise ValueError("Each message must have a non-empty string role")
        if not isinstance(self.generation_settings, dict):
            raise ValueError("generation_settings must be a dictionary")
        if any(not isinstance(key, str) for key in self.generation_settings):
            raise ValueError("generation_settings keys must be strings")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "generation_settings": self.generation_settings,
        }
        if self.messages is not None:
            result["messages"] = self.messages
        else:
            result["prompt"] = self.prompt
        return deepcopy(result)

    @classmethod
    def parse(cls, value: "InferenceRequest | dict[str, Any]") -> "InferenceRequest":
        if isinstance(value, cls):
            return cls(**value.to_dict())
        if not isinstance(value, dict):
            raise ValueError("request must be an InferenceRequest or dictionary")
        try:
            return cls(**deepcopy(value))
        except TypeError as exc:
            raise ValueError(f"Invalid request fields: {exc}") from exc
