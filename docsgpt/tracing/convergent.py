"""Convergent destination for DocsGPT's existing OTel replay (no second spans)."""

from __future__ import annotations

import os

from docsgpt.version import get_version


def configure() -> None:
    """Attach to the existing provider only when a destination is configured.

    ``init`` is process-idempotent and preserves application-owned exporters.
    A release always comes from the deployed package unless explicitly overridden.
    """
    if not any(os.getenv(key) for key in ("CONVERGENT_API_KEY", "CONVERGENT_SPANS_DIR", "CONVERGENT_TRACES_EXPORTER")):
        return
    import convergent

    convergent.init(release=os.getenv("CONVERGENT_RELEASE") or get_version())
