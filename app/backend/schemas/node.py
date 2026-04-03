"""Pydantic models for node definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class PortType(str, Enum):
    AUDIO = "audio"
    STRING = "string"


class PortDefinition(BaseModel):
    name: str
    port_type: PortType
    label: str
    optional: bool = False


class ParameterType(str, Enum):
    SELECT = "select"
    SLIDER = "slider"
    CHECKBOX = "checkbox"
    TEXT = "text"
    INT = "int"
    FLOAT = "float"


class ParameterDefinition(BaseModel):
    name: str
    label: str
    type: ParameterType
    default: Any = None
    options: list[dict[str, str]] | None = None  # for SELECT: [{value, label}]
    min: float | None = None  # for SLIDER/INT/FLOAT
    max: float | None = None
    step: float | None = None


class NodeDefinition(BaseModel):
    node_type: str
    label: str
    category: str
    description: str = ""
    inputs: list[PortDefinition] = []
    outputs: list[PortDefinition] = []
    parameters: list[ParameterDefinition] = []
    dynamic_outputs: bool = False  # outputs change based on params (e.g., MSST model)
