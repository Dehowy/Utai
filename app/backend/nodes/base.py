"""Abstract base class for all workflow nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.backend.schemas.node import NodeDefinition


class ExecutionContext:
    """Context passed to nodes during execution."""

    def __init__(self, device: str = "cpu", temp_dir: str = "", progress_callback=None,
                 cancelled_check=None):
        self.device = device
        self.temp_dir = temp_dir
        self._progress_callback = progress_callback
        self._cancelled_check = cancelled_check  # callable that returns True if cancelled
        self._child_processes: list = []  # track subprocesses for cleanup

    @property
    def is_cancelled(self) -> bool:
        if self._cancelled_check:
            return self._cancelled_check()
        return False

    def register_process(self, process):
        """Register a child process so it can be killed on cancellation."""
        self._child_processes.append(process)

    def kill_children(self):
        """Kill all registered child processes."""
        import sys
        for proc in self._child_processes:
            try:
                pid = proc.pid
                if pid and proc.returncode is None:
                    if sys.platform == "win32":
                        import subprocess
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True,
                            creationflags=0x08000000,
                        )
                    else:
                        import os, signal
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass
        self._child_processes.clear()

    async def report_progress(self, progress: float, message: str = ""):
        if self._progress_callback:
            await self._progress_callback(progress, message)


class BaseNode(ABC):
    """Abstract base class for workflow nodes.

    Each node type must implement:
    - definition(): returns the static NodeDefinition
    - execute(): performs the node's processing
    """

    @classmethod
    @abstractmethod
    def definition(cls) -> NodeDefinition:
        """Return the node type definition (ports, parameters, metadata)."""
        ...

    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute this node.

        Args:
            inputs: dict mapping input port names to their values.
                    For audio ports: (numpy_array, sample_rate) tuple.
            params: dict mapping parameter names to user-set values.
            context: execution context with device info and progress reporting.

        Returns:
            dict mapping output port names to their values.
        """
        ...
