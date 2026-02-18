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
    from mem0.graphs.configs import (
        GraphStoreConfig,
        KuzuConfig,
        MemgraphConfig,
        Neo4jConfig,
        NeptuneConfig,
    )

    # 1. Update the Union type annotation to include FalkorDBConfig
    GraphStoreConfig.model_fields["config"].annotation = Union[
        Neo4jConfig, MemgraphConfig, NeptuneConfig, KuzuConfig, FalkorDBConfig
    ]

    # 2. Replace the field validator with one that handles 'falkordb'.
    #    Pydantic V2 inspects function signature: for mode='after', it expects
    #    exactly 2 positional params (value, info) or 1 (value only).
    #    We replicate the original logic plus the falkordb case.
    def _patched_validate_config(v, values):
        provider = values.data.get("provider")
        if provider == "falkordb":
            if isinstance(v, FalkorDBConfig):
                return v
            if isinstance(v, dict):
                return FalkorDBConfig(**v)
            return FalkorDBConfig(**v.model_dump())
        elif provider == "neo4j":
            if isinstance(v, Neo4jConfig):
                return v
            return Neo4jConfig(**(v if isinstance(v, dict) else v.model_dump()))
        elif provider == "memgraph":
            if isinstance(v, MemgraphConfig):
                return v
            return MemgraphConfig(**(v if isinstance(v, dict) else v.model_dump()))
        elif provider in ("neptune", "neptunedb"):
            if isinstance(v, NeptuneConfig):
                return v
            return NeptuneConfig(**(v if isinstance(v, dict) else v.model_dump()))
        elif provider == "kuzu":
            if isinstance(v, KuzuConfig):
                return v
            return KuzuConfig(**(v if isinstance(v, dict) else v.model_dump()))
        else:
            raise ValueError(f"Unsupported graph store provider: {provider}")

    field_validators = GraphStoreConfig.__pydantic_decorators__.field_validators
    for _dec_name, dec in field_validators.items():
        dec.func = _patched_validate_config
        break  # only one field_validator on 'config'

    # 3. Rebuild the Pydantic model to recompile with patched validators and types
    GraphStoreConfig.model_rebuild(force=True)
