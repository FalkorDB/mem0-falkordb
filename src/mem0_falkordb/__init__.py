"""FalkorDB graph store plugin for Mem0.

Call register() before creating a Mem0 Memory instance::

    from mem0_falkordb import register
    register()
"""

from mem0_falkordb.patch import register

__all__ = ["register"]
