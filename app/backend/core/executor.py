"""Workflow execution engine."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import numpy as np

from app.backend.config import get_device, TEMP_DIR
from app.backend.core.audio_buffer import AudioData, buffer_manager
from app.backend.core.graph import (
    GraphError,
    get_downstream_nodes,
    get_node_input_map,
    topological_sort,
    validate_graph,
)
from app.backend.core.node_registry import NodeRegistry
from app.backend.nodes.base import ExecutionContext
from app.backend.schemas.execution import ExecutionComplete, NodeStatus, NodeStatusUpdate
from app.backend.schemas.workflow import Workflow

logger = logging.getLogger(__name__)


class ExecutionState:
    """Tracks state of a running execution."""

    def __init__(self, execution_id: str, workflow: Workflow):
        self.execution_id = execution_id
        self.workflow = workflow
        self.node_outputs: dict[str, dict[str, Any]] = {}  # node_id -> {port: value}
        self.status_updates: list[dict] = []
        self._listeners: list[asyncio.Queue] = []
        self._started = False
        self._cancelled = False
        self._task: asyncio.Task | None = None  # reference to the running asyncio task

    def add_listener(self, queue: asyncio.Queue):
        self._listeners.append(queue)

    def remove_listener(self, queue: asyncio.Queue):
        self._listeners.remove(queue)

    def cancel(self):
        self._cancelled = True
        # Also cancel the running asyncio task to interrupt long-running node execution
        if self._task and not self._task.done():
            self._task.cancel()

    async def _broadcast(self, message: dict):
        self.status_updates.append(message)
        for q in self._listeners:
            await q.put(message)


class GraphExecutor:
    """Executes workflow graphs."""

    def __init__(self):
        self._running: dict[str, ExecutionState] = {}

    def get_state(self, execution_id: str) -> ExecutionState | None:
        return self._running.get(execution_id)

    async def execute(self, workflow: Workflow) -> str:
        """Prepare a workflow for execution. Returns execution_id.

        Execution does NOT start immediately — call start_if_needed()
        after the WebSocket listener is attached so status updates are
        delivered in real-time.
        """
        errors = validate_graph(workflow)
        if errors:
            raise GraphError(f"Invalid workflow: {'; '.join(errors)}")

        execution_id = uuid.uuid4().hex[:16]
        state = ExecutionState(execution_id, workflow)
        self._running[execution_id] = state
        return execution_id

    def start_if_needed(self, execution_id: str):
        """Start execution if it hasn't started yet. Call after WS connects."""
        state = self._running.get(execution_id)
        if state and not state._started:
            state._started = True
            from_node_id = getattr(state, '_from_node_id', None)
            if from_node_id:
                state._task = asyncio.create_task(self._run_partial(state, from_node_id))
            else:
                state._task = asyncio.create_task(self._run(state))

    async def execute_from_node(
        self,
        workflow: Workflow,
        from_node_id: str,
        cached_outputs: dict[str, dict[str, str]],
    ) -> str:
        """Start a partial execution from *from_node_id*, reusing cached upstream outputs.

        *cached_outputs* maps node_id -> {port_name: buffer_id}.  Buffer IDs are
        resolved back to (samples, sr) tuples via the audio buffer manager so that
        downstream nodes receive the same data format as a full execution.
        """
        errors = validate_graph(workflow)
        if errors:
            raise GraphError(f"Invalid workflow: {'; '.join(errors)}")

        # Verify from_node_id exists
        node_ids = {n.id for n in workflow.nodes}
        if from_node_id not in node_ids:
            raise GraphError(f"from_node_id '{from_node_id}' not found in workflow")

        execution_id = uuid.uuid4().hex[:16]
        state = ExecutionState(execution_id, workflow)

        # Resolve cached buffer IDs -> (samples, sr) and pre-populate node_outputs
        for node_id, port_buffers in cached_outputs.items():
            resolved: dict[str, Any] = {}
            for port_name, buf_id in port_buffers.items():
                audio_data = buffer_manager.get(buf_id)
                if audio_data is not None:
                    resolved[port_name] = (audio_data.samples, audio_data.sample_rate)
                else:
                    # Non-audio cached value (e.g. video path, title string)
                    resolved[port_name] = buf_id
            state.node_outputs[node_id] = resolved

        self._running[execution_id] = state
        # Don't start yet — wait for WS to connect via start_if_needed()
        state._from_node_id = from_node_id  # store for deferred start
        return execution_id

    async def _run_partial(self, state: ExecutionState, from_node_id: str):
        """Execute only *from_node_id* and its downstream dependents."""
        workflow = state.workflow
        try:
            order = topological_sort(workflow.nodes, workflow.connections)
            input_map = get_node_input_map(workflow.connections)
            node_lookup = {n.id: n for n in workflow.nodes}
            device = get_device()

            downstream = get_downstream_nodes(
                from_node_id, workflow.nodes, workflow.connections
            )

            # Only execute nodes that are both in topological order AND downstream
            nodes_to_run = [nid for nid in order if nid in downstream]

            # Broadcast skipped status for upstream cached nodes
            for node_id in order:
                if node_id not in downstream and node_id in state.node_outputs:
                    await state._broadcast(
                        NodeStatusUpdate(
                            node_id=node_id,
                            status=NodeStatus.COMPLETED,
                            progress=1.0,
                            message="Cached",
                        ).model_dump()
                    )

            failed_nodes: set[str] = set()
            has_errors = False

            for node_id in nodes_to_run:
                if state._cancelled:
                    await state._broadcast(
                        ExecutionComplete(
                            execution_id=state.execution_id,
                            success=False,
                            error="Cancelled by user",
                        ).model_dump()
                    )
                    return

                wf_node = node_lookup[node_id]

                # Skip if any upstream node failed
                upstream_failed = False
                if node_id in input_map:
                    for _tp, (src_node_id, _sp) in input_map[node_id].items():
                        if src_node_id in failed_nodes:
                            upstream_failed = True
                            break
                if upstream_failed:
                    failed_nodes.add(node_id)
                    await state._broadcast(
                        NodeStatusUpdate(
                            node_id=node_id,
                            status=NodeStatus.ERROR,
                            message="Skipped (upstream error)",
                        ).model_dump()
                    )
                    continue

                node_instance = NodeRegistry.create_instance(wf_node.type)

                # Gather inputs from upstream outputs (cached or freshly computed)
                inputs: dict[str, Any] = {}
                if node_id in input_map:
                    for target_port, (src_node_id, src_port) in input_map[node_id].items():
                        if src_node_id in state.node_outputs:
                            inputs[target_port] = state.node_outputs[src_node_id].get(src_port)

                async def make_progress_cb(nid: str, node_type: str):
                    async def cb(progress: float, message: str = ""):
                        await state._broadcast(
                            NodeStatusUpdate(
                                node_id=nid,
                                status=NodeStatus.RUNNING,
                                progress=progress,
                                message=message,
                            ).model_dump()
                        )
                        if message:
                            await state._broadcast({"type": "log", "message": f"[{node_type}] {message}"})
                    return cb

                progress_cb = await make_progress_cb(node_id, wf_node.type)
                context = ExecutionContext(
                    device=device,
                    temp_dir=str(TEMP_DIR),
                    progress_callback=progress_cb,
                    cancelled_check=lambda: state._cancelled,
                )

                await state._broadcast(
                    NodeStatusUpdate(
                        node_id=node_id,
                        status=NodeStatus.RUNNING,
                        progress=0.0,
                        message="Starting...",
                    ).model_dump()
                )

                try:
                    outputs = await node_instance.execute(inputs, wf_node.params, context)
                except asyncio.CancelledError:
                    context.kill_children()
                    raise
                except Exception as e:
                    logger.exception(f"Node {node_id} ({wf_node.type}) failed")
                    has_errors = True
                    failed_nodes.add(node_id)
                    await state._broadcast(
                        NodeStatusUpdate(
                            node_id=node_id,
                            status=NodeStatus.ERROR,
                            message=str(e),
                        ).model_dump()
                    )
                    continue

                state.node_outputs[node_id] = outputs or {}

                output_buffers = {}
                for port_name, value in (outputs or {}).items():
                    if isinstance(value, tuple) and len(value) == 2:
                        samples, sr = value
                        if isinstance(samples, np.ndarray):
                            buf_id = buffer_manager.store(
                                AudioData(samples=samples, sample_rate=sr, name=f"{node_id}_{port_name}")
                            )
                            output_buffers[port_name] = buf_id
                    elif isinstance(value, str):
                        # Pass through string metadata (video paths, titles, etc.)
                        output_buffers[port_name] = value

                await state._broadcast(
                    NodeStatusUpdate(
                        node_id=node_id,
                        status=NodeStatus.COMPLETED,
                        progress=1.0,
                        message="Done",
                        outputs=output_buffers if output_buffers else None,
                    ).model_dump()
                )

            if has_errors:
                await state._broadcast({"type": "log", "message": f"[exec] Partial execution finished with {len(failed_nodes)} error(s)"})
            await state._broadcast(
                ExecutionComplete(
                    execution_id=state.execution_id,
                    success=not has_errors,
                ).model_dump()
            )

        except asyncio.CancelledError:
            logger.info(f"Partial execution {state.execution_id} was force-stopped")
            try:
                for wn in workflow.nodes:
                    last = next((u for u in reversed(state.status_updates)
                                 if u.get("node_id") == wn.id), None)
                    if last and last.get("status") in ("running", "pending"):
                        await state._broadcast(
                            NodeStatusUpdate(
                                node_id=wn.id,
                                status=NodeStatus.ERROR,
                                message="Stopped",
                            ).model_dump()
                        )
                await state._broadcast({"type": "log", "message": "[exec] Force stopped by user"})
                await state._broadcast(
                    ExecutionComplete(
                        execution_id=state.execution_id,
                        success=False,
                        error="Force stopped by user",
                    ).model_dump()
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception("Partial execution failed")
            await state._broadcast(
                ExecutionComplete(
                    execution_id=state.execution_id,
                    success=False,
                    error=str(e),
                ).model_dump()
            )
        finally:
            asyncio.create_task(self._cleanup_state(state.execution_id, delay=600))

    async def _run(self, state: ExecutionState):
        workflow = state.workflow
        try:
            # Get execution order
            order = topological_sort(workflow.nodes, workflow.connections)
            input_map = get_node_input_map(workflow.connections)

            node_lookup = {n.id: n for n in workflow.nodes}
            device = get_device()
            logger.info(f"Execution device: {device.upper()}")
            await state._broadcast({"type": "log", "message": f"[exec] Device: {device.upper()}"})
            if device == "cuda":
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU: {gpu_name}")
                await state._broadcast({"type": "log", "message": f"[exec] GPU: {gpu_name}"})
            elif device == "cpu":
                logger.warning("Running on CPU — separation will be slow!")
                await state._broadcast({"type": "log", "message": "[exec] WARNING: Running on CPU — will be slow!"})

            await state._broadcast({"type": "log", "message": f"[exec] Executing {len(order)} nodes..."})

            # Track failed nodes so we can skip their downstream dependents
            failed_nodes: set[str] = set()
            has_errors = False

            for node_id in order:
                if state._cancelled:
                    await state._broadcast(
                        ExecutionComplete(
                            execution_id=state.execution_id,
                            success=False,
                            error="Cancelled by user",
                        ).model_dump()
                    )
                    return

                wf_node = node_lookup[node_id]

                # Skip if any upstream node failed
                upstream_failed = False
                if node_id in input_map:
                    for _tp, (src_node_id, _sp) in input_map[node_id].items():
                        if src_node_id in failed_nodes:
                            upstream_failed = True
                            break
                if upstream_failed:
                    failed_nodes.add(node_id)
                    await state._broadcast(
                        NodeStatusUpdate(
                            node_id=node_id,
                            status=NodeStatus.ERROR,
                            message="Skipped (upstream error)",
                        ).model_dump()
                    )
                    continue

                # Create node instance
                node_instance = NodeRegistry.create_instance(wf_node.type)

                # Gather inputs from upstream outputs
                inputs: dict[str, Any] = {}
                if node_id in input_map:
                    for target_port, (src_node_id, src_port) in input_map[node_id].items():
                        if src_node_id in state.node_outputs:
                            inputs[target_port] = state.node_outputs[src_node_id].get(src_port)

                # Progress callback for this node — also broadcast as log
                async def make_progress_cb(nid: str, node_type: str):
                    async def cb(progress: float, message: str = ""):
                        await state._broadcast(
                            NodeStatusUpdate(
                                node_id=nid,
                                status=NodeStatus.RUNNING,
                                progress=progress,
                                message=message,
                            ).model_dump()
                        )
                        if message:
                            await state._broadcast({"type": "log", "message": f"[{node_type}] {message}"})
                    return cb

                progress_cb = await make_progress_cb(node_id, wf_node.type)
                context = ExecutionContext(
                    device=device,
                    temp_dir=str(TEMP_DIR),
                    progress_callback=progress_cb,
                    cancelled_check=lambda: state._cancelled,
                )

                # Report start
                await state._broadcast({"type": "log", "message": f"[{wf_node.type}] Starting {node_id}..."})
                await state._broadcast(
                    NodeStatusUpdate(
                        node_id=node_id,
                        status=NodeStatus.RUNNING,
                        progress=0.0,
                        message="Starting...",
                    ).model_dump()
                )

                # Execute node — on failure, mark as failed and continue other branches
                try:
                    outputs = await node_instance.execute(inputs, wf_node.params, context)
                except asyncio.CancelledError:
                    # Force-stopped: kill any child processes this node spawned
                    context.kill_children()
                    raise  # re-raise to be caught by the outer CancelledError handler
                except Exception as e:
                    logger.exception(f"Node {node_id} ({wf_node.type}) failed")
                    has_errors = True
                    failed_nodes.add(node_id)
                    await state._broadcast({"type": "log", "message": f"[{wf_node.type}] ERROR: {e}"})
                    await state._broadcast(
                        NodeStatusUpdate(
                            node_id=node_id,
                            status=NodeStatus.ERROR,
                            message=str(e),
                        ).model_dump()
                    )
                    continue

                # Store outputs
                state.node_outputs[node_id] = outputs or {}

                # Store audio outputs in buffer manager and report completion
                # For audio_input nodes, reuse the existing buffer_id (no duplicate)
                output_buffers = {}
                existing_bid = wf_node.params.get("buffer_id") if wf_node.type == "audio_input" else None
                for port_name, value in (outputs or {}).items():
                    if isinstance(value, tuple) and len(value) == 2:
                        samples, sr = value
                        if isinstance(samples, np.ndarray):
                            if existing_bid and port_name == "audio":
                                buf_id = existing_bid
                            else:
                                buf_id = buffer_manager.store(
                                    AudioData(samples=samples, sample_rate=sr, name=f"{node_id}_{port_name}")
                                )
                            output_buffers[port_name] = buf_id
                    elif isinstance(value, str):
                        # Pass through string metadata (video paths, titles, etc.)
                        output_buffers[port_name] = value

                await state._broadcast({"type": "log", "message": f"[{wf_node.type}] Done — {len(output_buffers)} output(s)"})
                await state._broadcast(
                    NodeStatusUpdate(
                        node_id=node_id,
                        status=NodeStatus.COMPLETED,
                        progress=1.0,
                        message="Done",
                        outputs=output_buffers if output_buffers else None,
                    ).model_dump()
                )

            # All done
            if has_errors:
                await state._broadcast({"type": "log", "message": f"[exec] Workflow finished with {len(failed_nodes)} error(s)"})
            else:
                await state._broadcast({"type": "log", "message": "[exec] Workflow complete!"})
            await state._broadcast(
                ExecutionComplete(
                    execution_id=state.execution_id,
                    success=not has_errors,
                    error=f"{len(failed_nodes)} node(s) failed" if has_errors else None,
                ).model_dump()
            )

        except asyncio.CancelledError:
            logger.info(f"Execution {state.execution_id} was force-stopped")
            try:
                # Mark all still-running/pending nodes as stopped
                for wn in workflow.nodes:
                    last = next((u for u in reversed(state.status_updates)
                                 if u.get("node_id") == wn.id), None)
                    if last and last.get("status") in ("running", "pending"):
                        await state._broadcast(
                            NodeStatusUpdate(
                                node_id=wn.id,
                                status=NodeStatus.ERROR,
                                message="Stopped",
                            ).model_dump()
                        )
                await state._broadcast({"type": "log", "message": "[exec] Force stopped by user"})
                await state._broadcast(
                    ExecutionComplete(
                        execution_id=state.execution_id,
                        success=False,
                        error="Force stopped by user",
                    ).model_dump()
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception("Execution failed")
            await state._broadcast(
                ExecutionComplete(
                    execution_id=state.execution_id,
                    success=False,
                    error=str(e),
                ).model_dump()
            )
        finally:
            # Clean up state after a delay so late-joining clients can still retrieve results
            asyncio.create_task(self._cleanup_state(state.execution_id, delay=600))

    async def _cleanup_state(self, execution_id: str, delay: int = 600):
        """Remove execution state after a delay to free memory."""
        await asyncio.sleep(delay)
        state = self._running.pop(execution_id, None)
        if state:
            # Clean up audio buffers created during this execution
            for node_outputs in state.node_outputs.values():
                for value in node_outputs.values():
                    if isinstance(value, tuple) and len(value) == 2:
                        # Raw audio tuple — nothing to clean
                        pass
            logger.info(f"Cleaned up execution state: {execution_id}")


# Global executor instance
executor = GraphExecutor()
