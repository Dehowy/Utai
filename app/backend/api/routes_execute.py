"""Workflow execution and WebSocket endpoints."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.backend.core.executor import executor
from app.backend.core.graph import GraphError
from app.backend.schemas.execution import ExecutionRequest, PartialExecutionRequest
from app.backend.schemas.workflow import Workflow

router = APIRouter(tags=["execution"])


@router.post("/api/execute")
async def start_execution(request: ExecutionRequest):
    """Submit a workflow for execution."""
    workflow = Workflow(**request.workflow)
    execution_id = await executor.execute(workflow)
    return {"execution_id": execution_id, "status": "started"}


@router.post("/api/execute/from-node")
async def start_partial_execution(request: PartialExecutionRequest):
    """Execute a workflow starting from a specific node, reusing cached upstream outputs."""
    workflow = Workflow(**request.workflow)
    try:
        execution_id = await executor.execute_from_node(
            workflow,
            from_node_id=request.from_node_id,
            cached_outputs=request.cached_outputs,
        )
    except GraphError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"execution_id": execution_id, "status": "started"}


@router.post("/api/execute/stop/{execution_id}")
async def stop_execution(execution_id: str):
    """Cancel a running execution."""
    state = executor.get_state(execution_id)
    if state is None:
        return {"status": "not_found"}
    state.cancel()
    return {"status": "cancelling"}


@router.websocket("/ws/execution/{execution_id}")
async def execution_websocket(websocket: WebSocket, execution_id: str):
    """WebSocket for real-time execution progress."""
    await websocket.accept()

    state = executor.get_state(execution_id)
    if state is None:
        await websocket.send_json({"type": "error", "message": "Execution not found"})
        await websocket.close()
        return

    queue: asyncio.Queue = asyncio.Queue()
    state.add_listener(queue)

    # Send any existing updates (for late-joining clients)
    for update in state.status_updates:
        await websocket.send_json(update)

    # Now that the WS listener is attached, start execution if pending
    executor.start_if_needed(execution_id)

    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
            if message.get("type") == "execution_complete":
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.remove_listener(queue)
