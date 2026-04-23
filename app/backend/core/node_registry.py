"""Node type registry - discovers and manages all available node types."""

from __future__ import annotations

from app.backend.nodes.base import BaseNode
from app.backend.schemas.node import NodeDefinition


class NodeRegistry:
    _nodes: dict[str, type[BaseNode]] = {}

    @classmethod
    def register(cls, node_class: type[BaseNode]) -> type[BaseNode]:
        """Register a node class. Can be used as a decorator."""
        defn = node_class.definition()
        cls._nodes[defn.node_type] = node_class
        return node_class

    @classmethod
    def get_all_definitions(cls) -> list[NodeDefinition]:
        return [cls._nodes[k].definition() for k in cls._nodes]

    @classmethod
    def get_definition(cls, node_type: str) -> NodeDefinition | None:
        if node_type in cls._nodes:
            return cls._nodes[node_type].definition()
        return None

    @classmethod
    def create_instance(cls, node_type: str) -> BaseNode:
        if node_type not in cls._nodes:
            raise ValueError(f"Unknown node type: {node_type}")
        return cls._nodes[node_type]()

    @classmethod
    def registered_types(cls) -> list[str]:
        return list(cls._nodes.keys())


def register_all_nodes():
    """Import all node modules to trigger registration. Missing deps are skipped."""
    import logging
    _log = logging.getLogger("utai.nodes")
    _mods = [
        'audio_input', 'audio_output', 'msst_separation', 'vr_separation',
        'rvc_conversion', 'separation', 'ytdlp_download', 'video_output',
        'video_input', 'media_output', 'audio_merge', 'effects',
    ]
    for mod in _mods:
        try:
            __import__(f'app.backend.nodes.{mod}')
        except Exception as e:
            _log.warning(f"Skipped node module {mod}: {e}")
