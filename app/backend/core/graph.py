"""Graph validation and topological sort for workflow execution."""

from __future__ import annotations

from collections import defaultdict, deque

from app.backend.schemas.workflow import Workflow, WorkflowConnection, WorkflowNode
from app.backend.core.node_registry import NodeRegistry


class GraphError(Exception):
    pass


def validate_graph(workflow: Workflow) -> list[str]:
    """Validate a workflow graph. Returns list of error messages (empty = valid)."""
    errors = []
    node_ids = {n.id for n in workflow.nodes}

    # Check all connections reference valid nodes
    for conn in workflow.connections:
        if conn.from_node not in node_ids:
            errors.append(f"Connection references unknown source node: {conn.from_node}")
        if conn.to_node not in node_ids:
            errors.append(f"Connection references unknown target node: {conn.to_node}")

    # Check for cycles
    if not errors and _has_cycle(workflow.nodes, workflow.connections):
        errors.append("Workflow contains a cycle")

    # Check node types exist
    for node in workflow.nodes:
        if NodeRegistry.get_definition(node.type) is None:
            errors.append(f"Unknown node type: {node.type} (node {node.id})")

    return errors


def topological_sort(
    nodes: list[WorkflowNode],
    connections: list[WorkflowConnection],
) -> list[str]:
    """Return node IDs in topological execution order.

    Raises GraphError if the graph has a cycle.
    """
    # Build adjacency list and in-degree count
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for conn in connections:
        adjacency[conn.from_node].append(conn.to_node)
        in_degree[conn.to_node] = in_degree.get(conn.to_node, 0) + 1

    # Kahn's algorithm
    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    result = []

    while queue:
        node_id = queue.popleft()
        result.append(node_id)
        for neighbor in adjacency[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(nodes):
        raise GraphError("Workflow contains a cycle")

    return result


def _has_cycle(nodes: list[WorkflowNode], connections: list[WorkflowConnection]) -> bool:
    try:
        topological_sort(nodes, connections)
        return False
    except GraphError:
        return True


def get_node_input_map(
    connections: list[WorkflowConnection],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Build a mapping: target_node_id -> {target_port: (source_node_id, source_port)}."""
    input_map: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for conn in connections:
        input_map[conn.to_node][conn.to_port] = (conn.from_node, conn.from_port)
    return input_map


def get_downstream_nodes(
    from_node_id: str,
    nodes: list[WorkflowNode],
    connections: list[WorkflowConnection],
) -> set[str]:
    """Return the set of node IDs reachable from *from_node_id* (inclusive).

    Uses BFS over the directed adjacency list built from *connections*.
    """
    adjacency: dict[str, list[str]] = defaultdict(list)
    for conn in connections:
        adjacency[conn.from_node].append(conn.to_node)

    visited: set[str] = set()
    queue = deque([from_node_id])
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        for neighbor in adjacency[nid]:
            if neighbor not in visited:
                queue.append(neighbor)

    return visited
