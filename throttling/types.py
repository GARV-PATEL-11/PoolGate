"""Enums for the two-layer throttling system."""

from __future__ import annotations

import enum


class CapabilityType(enum.Enum):
    TEXT_GENERATION = "text_generation"
    STRUCTURED_GEN = "structured_gen"
    TOOL_CALLING = "tool_calling"
    MODERATION = "moderation"
    TRANSCRIPTION = "transcription"
    SYNTHESIS = "synthesis"


class ThrottleMode(enum.Enum):
    DISABLED = "disabled"
    RPM_ONLY = "rpm_only"
    CONCURRENCY_ONLY = "concurrency_only"
    RPM_AND_CONCURRENCY = "rpm_and_concurrency"
