"""Pydantic models for workflow serialization."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Position(BaseModel):
    x: float
    y: float


class WorkflowNode(BaseModel):
    id: str
    type: str
    position: Position
    params: dict[str, Any] = {}


class WorkflowConnection(BaseModel):
    from_node: str
    from_port: str
    to_node: str
    to_port: str


class Workflow(BaseModel):
    version: str = "1.0"
    name: str = "Untitled"
    project_type: str = "workflow"  # "workflow" or "training"
    id: str | None = None
    nodes: list[WorkflowNode] = []
    connections: list[WorkflowConnection] = []


class WorkflowListItem(BaseModel):
    id: str
    name: str
    modified: str
