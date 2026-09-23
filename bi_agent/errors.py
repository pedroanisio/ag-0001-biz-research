"""Typed errors for the bi_agent package. Every failure path raises one of these."""

from __future__ import annotations


class BiAgentError(Exception):
    """Base class for all package errors."""


class CrawlError(BiAgentError):
    """The start URL could not be fetched or parsed at all."""


class LLMOutputError(BiAgentError):
    """Model output failed schema or semantic validation after bounded retries."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or []


class EvidenceError(BiAgentError):
    """A claim references evidence that does not exist or does not support its classification."""


class StageError(BiAgentError):
    """A pipeline stage was invoked before its inputs exist."""
