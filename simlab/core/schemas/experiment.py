"""Pydantic v2 models for SIMLAB unified experiment request/response."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ExperimentRequest(BaseModel):
    """Unified experiment request schema."""

    experiment_id: str = Field(
        default_factory=lambda: f"exp_{uuid4().hex[:8]}",
        description="Unique experiment identifier",
    )
    domain: Literal[
        "math",
        "physics",
        "chemistry",
        "materials",
        "atomic",
        "nuclear",
        "data",
        "control",
        "hybrid",
        "nvidia",
        "qdgeometry",
    ] = Field(description="Scientific domain for the experiment")
    type: str = Field(description="Experiment type within the domain")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Domain-specific parameters"
    )
    environment: dict[str, Any] = Field(
        default_factory=dict, description="Environment overrides (e.g. gravity, temperature)"
    )
    outputs: list[str] = Field(
        default_factory=list,
        description="Requested output types: 'plot', 'report', 'json', 'latex'",
    )
    solver: str = Field(
        default="auto", description="Preferred solver/method (auto selects best)"
    )

    model_config = {"extra": "allow"}


class ExperimentResult(BaseModel):
    """Unified experiment result schema."""

    experiment_id: str
    domain: str
    type: str
    status: Literal["success", "error", "partial"] = "success"
    results: dict[str, Any] = Field(default_factory=dict)
    plots: list[str] = Field(
        default_factory=list,
        description="Base64-encoded PNG strings or file paths",
    )
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class ValidationResult(BaseModel):
    """Result of a pre-flight validation check."""

    valid: bool
    blocked: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    checks_passed: list[str] = Field(default_factory=list)
