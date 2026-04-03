"""Pydantic models for execution state."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class NodeStatusUpdate(BaseModel):
    type: str = "node_status"
    node_id: str
    status: NodeStatus
    progress: float = 0.0
    message: str = ""
    outputs: dict[str, str] | None = None  # port_name -> buffer_id


class ExecutionComplete(BaseModel):
    type: str = "execution_complete"
    execution_id: str
    success: bool
    error: str | None = None


class ExecutionRequest(BaseModel):
    workflow: dict[str, Any]  # serialized Workflow


class PartialExecutionRequest(BaseModel):
    workflow: dict[str, Any]  # serialized Workflow
    from_node_id: str
    cached_outputs: dict[str, dict[str, str]]  # node_id -> {port_name: buffer_id}


class ExecutionResponse(BaseModel):
    execution_id: str
    status: str
