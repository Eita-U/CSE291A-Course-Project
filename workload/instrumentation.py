"""Module C: request-level and task-level experiment records.

Scheduler timing_metadata is stored as returned. This module adds identifiers,
token totals, task-span timestamps, and tool-wait measurements. Metric
aggregation (throughput, tail latency, peak memory) belongs to measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _int_usage(token_usage: Mapping[str, Any] | None) -> dict[str, int]:
    if not token_usage:
        return {}
    return {
        str(key): int(value)
        for key, value in token_usage.items()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }


@dataclass
class RequestRecord:
    request_id: str
    task_id: str
    step_id: int
    profile_name: str
    status: str
    worker_id: int | None
    error: str | None
    prompt_token_estimate: int
    max_tokens: int
    tool_delay_s: float
    tool_wait_ms: float
    timing_metadata: dict[str, Any]
    token_usage: dict[str, int]
    response_chars: int | None


@dataclass
class TaskRecord:
    task_id: str
    profile_name: str
    status: str
    n_steps_planned: int
    n_steps_completed: int
    task_start_time: str
    task_finish_time: str
    task_duration_ms: float
    tool_wait_ms_total: float
    queue_time_ms_total: float
    api_time_ms_total: float
    prompt_tokens: int
    completion_tokens: int
    request_ids: list[str]
    error: str | None


@dataclass
class InstrumentationRecorder:
    requests: list[RequestRecord] = field(default_factory=list)
    tasks: list[TaskRecord] = field(default_factory=list)

    def record_request(
        self,
        *,
        profile_name: str,
        scheduler_response: Mapping[str, Any],
        prompt_token_estimate: int,
        max_tokens: int,
        tool_delay_s: float,
        tool_wait_ms: float,
    ) -> RequestRecord:
        response = scheduler_response.get("response")
        record = RequestRecord(
            request_id=str(scheduler_response["request_id"]),
            task_id=str(scheduler_response["task_id"]),
            step_id=int(scheduler_response["step_id"]),
            profile_name=profile_name,
            status=str(scheduler_response["status"]),
            worker_id=scheduler_response.get("worker_id"),
            error=scheduler_response.get("error"),
            prompt_token_estimate=prompt_token_estimate,
            max_tokens=max_tokens,
            tool_delay_s=tool_delay_s,
            tool_wait_ms=tool_wait_ms,
            timing_metadata=dict(scheduler_response.get("timing_metadata") or {}),
            token_usage=_int_usage(scheduler_response.get("token_usage")),
            response_chars=None if response is None else len(str(response)),
        )
        self.requests.append(record)
        return record

    def record_task(
        self,
        *,
        spec_task_id: str,
        profile_name: str,
        n_steps_planned: int,
        request_records: Sequence[RequestRecord],
        task_start_time: str,
        task_finish_time: str,
        task_duration_ms: float,
    ) -> TaskRecord:
        completed = list(request_records)
        if not completed:
            status = "failed"
            error = "Task produced no requests"
        elif all(item.status == "success" for item in completed) and len(completed) == n_steps_planned:
            status = "success"
            error = None
        else:
            status = next((item.status for item in completed if item.status != "success"), "failed")
            error = next((item.error for item in reversed(completed) if item.error), None)

        prompt_tokens = sum(item.token_usage.get("prompt_tokens", 0) for item in completed)
        completion_tokens = sum(item.token_usage.get("completion_tokens", 0) for item in completed)
        record = TaskRecord(
            task_id=spec_task_id,
            profile_name=profile_name,
            status=status,
            n_steps_planned=n_steps_planned,
            n_steps_completed=len(completed),
            task_start_time=task_start_time,
            task_finish_time=task_finish_time,
            task_duration_ms=task_duration_ms,
            tool_wait_ms_total=sum(item.tool_wait_ms for item in completed),
            queue_time_ms_total=sum(
                float(item.timing_metadata.get("queue_time_ms") or 0) for item in completed
            ),
            api_time_ms_total=sum(
                float(item.timing_metadata.get("api_time_ms") or 0) for item in completed
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            request_ids=[item.request_id for item in completed],
            error=error,
        )
        self.tasks.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": [asdict(item) for item in self.requests],
            "tasks": [asdict(item) for item in self.tasks],
        }

    def dump_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def dump_jsonl(
        self,
        request_path: str | Path,
        task_path: str | Path,
    ) -> None:
        _write_jsonl(request_path, [asdict(item) for item in self.requests])
        _write_jsonl(task_path, [asdict(item) for item in self.tasks])


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
