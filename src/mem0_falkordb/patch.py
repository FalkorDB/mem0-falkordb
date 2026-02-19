"""Monkey-patching logic to register FalkorDB into Mem0's factory and config system."""

import logging
from typing import Union

from mem0_falkordb.config import FalkorDBConfig

logger = logging.getLogger(__name__)

_registered = False


def register():
    """Register FalkorDB as a graph store provider in Mem0.

    Call this before creating a Mem0 Memory instance. Safe to call multiple times.

    Example::

        from mem0_falkordb import register
        register()

        from mem0 import Memory
        config = {
            "graph_store": {
                "provider": "falkordb",
                "config": {"host": "localhost", "port": 6379, "database": "mem0"},
            },
            ...
        }
        m = Memory.from_config(config)
    """
    global _registered
    if _registered:
        return
    _registered = True

    _patch_factory()
    _patch_config()
    logger.info("mem0-falkordb: FalkorDB provider registered successfully.")


def _patch_factory():
    """Add FalkorDB to GraphStoreFactory.provider_to_class."""
    from mem0.utils.factory import GraphStoreFactory

    GraphStoreFactory.provider_to_class["falkordb"] = (
        "mem0_falkordb.graph_memory.MemoryGraph"
    )


def _patch_config():
    """Patch GraphStoreConfig to accept FalkorDBConfig."""
    from mem0.graphs.configs import GraphStoreConfig

    # 1. Update the Union type annotation to include FalkorDBConfig
    original_annotation = GraphStoreConfig.model_fields["config"].annotation
    GraphStoreConfig.model_fields["config"].annotation = Union[
        original_annotation, FalkorDBConfig
    ]

    # 2. Capture the original validator and wrap it to handle 'falkordb'.
    #    Pydantic V2 inspects function signature: for mode='after', it expects
    #    exactly 2 positional params (value, info) or 1 (value only).
    field_validators = GraphStoreConfig.__pydantic_decorators__.field_validators
    original_func = None
    for _dec_name, dec in field_validators.items():
        original_func = dec.func
        break

    def _patched_validate_config(v, values):
        provider = values.data.get("provider")
        if provider == "falkordb":
            if isinstance(v, FalkorDBConfig):
                return v
            if isinstance(v, dict):
                return FalkorDBConfig(**v)
            return FalkorDBConfig(**v.model_dump())
        # Delegate all other providers to the original validator
        return original_func(v, values)

    for _dec_name, dec in field_validators.items():
        dec.func = _patched_validate_config
        break  # only one field_validator on 'config'

    # 3. Rebuild the Pydantic model to recompile with patched validators and types
    GraphStoreConfig.model_rebuild(force=True)

    # 4. Rebuild MemoryConfig so it picks up the patched GraphStoreConfig
    from mem0.configs.base import MemoryConfig

    MemoryConfig.model_rebuild(force=True)
