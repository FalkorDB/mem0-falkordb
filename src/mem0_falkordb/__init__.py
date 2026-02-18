"""FalkorDB graph store plugin for Mem0.

Importing this package automatically registers FalkorDB as a graph store
provider in Mem0. To avoid lint warnings for unused imports, call setup()::

    import mem0_falkordb
    mem0_falkordb.setup()
"""

from mem0_falkordb.patch import register


def setup():
    """No-op convenience function to satisfy linters.

    Registration happens automatically on import. This function exists so that
    users can reference ``mem0_falkordb`` without triggering F401 (unused import).
    """


__all__ = ["register", "setup"]

# Auto-register on import
register()
