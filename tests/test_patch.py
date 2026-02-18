"""Tests for the register/patch mechanism."""




def test_register_adds_falkordb_to_factory():
    """After register(), GraphStoreFactory should know about 'falkordb'."""
    from mem0_falkordb import register

    register()

    from mem0.utils.factory import GraphStoreFactory

    assert "falkordb" in GraphStoreFactory.provider_to_class
    assert (
        GraphStoreFactory.provider_to_class["falkordb"]
        == "mem0_falkordb.graph_memory.MemoryGraph"
    )


def test_register_is_idempotent():
    """Calling register() twice should not raise."""
    import mem0_falkordb.patch as patch_mod

    patch_mod._registered = False
    patch_mod.register()
    patch_mod.register()  # should not raise


def test_config_accepts_falkordb_provider():
    """GraphStoreConfig should accept provider='falkordb' after registration."""
    from mem0_falkordb import register

    register()

    from mem0.graphs.configs import GraphStoreConfig

    config = GraphStoreConfig(
        provider="falkordb",
        config={"host": "localhost", "port": 6379, "database": "mem0"},
    )
    assert config.provider == "falkordb"
    assert config.config.host == "localhost"
    assert config.config.port == 6379


def test_existing_providers_still_work():
    """Ensure Neo4j and other providers are not broken by the patch."""
    from mem0_falkordb import register

    register()

    from mem0.graphs.configs import GraphStoreConfig

    config = GraphStoreConfig(
        provider="neo4j",
        config={
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        },
    )
    assert config.provider == "neo4j"
    assert config.config.url == "bolt://localhost:7687"
