"""FalkorDB graph store plugin for Mem0.

Importing this package automatically registers FalkorDB as a graph store
provider in Mem0. No explicit setup is needed beyond:

    import mem0_falkordb  # noqa: F401
"""

from mem0_falkordb.patch import register

__all__ = ["register"]

# Auto-register on import
register()
