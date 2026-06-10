"""TMG (Temporal Memory Graph) client — the system under evaluation.

Wraps the TMG HTTP API for use in the benchmark.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import requests

from .base import MemorySystem, QueryResult

logger = logging.getLogger(__name__)


class TMGClient(MemorySystem):
    """Client for the TMG (Temporal Memory Graph) system."""

    def __init__(
        self,
        name: str = "TMG (ours)",
        api_base: str = "http://localhost:8732",
        storage_path: Optional[str] = None,
        timeout: int = 30,
        top_k: int = 5,
    ):
        super().__init__(name)
        self.api_base = api_base.rstrip("/")
        self.storage_path = storage_path or str(Path("/tmp/tmg_benchmark_storage"))
        self.timeout = timeout
        self.top_k = top_k

    def remember(
        self,
        text: str,
    ) -> str:
        # POST /api/remember
        resp = requests.post(
            f"{self.api_base}/api/remember",
            json={"text": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        task_id = data["data"]["task_id"]

        # Poll until completed
        for _ in range(60):
            time.sleep(1)
            status_resp = requests.get(
                f"{self.api_base}/api/remember/tasks/{task_id}",
                timeout=self.timeout,
            )
            status_data = status_resp.json()
            if status_data.get("status") == "completed":
                return task_id
            elif status_data.get("status") == "failed":
                raise RuntimeError(f"TMG remember task failed: {status_data}")

        raise TimeoutError(f"TMG remember task {task_id} timed out")

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        start = time.time()
        params: dict = {"query": question, "expand": "true"}

        resp = requests.post(
            f"{self.api_base}/api/find",
            json=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract results
        results = data.get("data", {}).get("results", [])
        effective_top_k = top_k if top_k is not None else self.top_k
        context_parts = [r.get("text", "") for r in results[:effective_top_k]]
        context = "\n".join(context_parts)

        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=context_parts,
            latency_ms=(time.time() - start) * 1000,
        )

    def reset(self) -> None:
        """Reset TMG by deleting storage directory."""
        import os
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)
            logger.info(f"Reset TMG storage: {self.storage_path}")
