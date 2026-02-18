"""Tests for the FalkorDB config model."""

from mem0_falkordb.config import FalkorDBConfig


def test_default_config():
    config = FalkorDBConfig()
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.database == "mem0"
    assert config.username is None
    assert config.password is None
    assert config.base_label is True


def test_custom_config():
    config = FalkorDBConfig(
        host="db.example.com",
        port=6380,
        database="test_graph",
        username="admin",
        password="secret",
    )
    assert config.host == "db.example.com"
    assert config.port == 6380
    assert config.database == "test_graph"
    assert config.username == "admin"
    assert config.password == "secret"


def test_config_from_dict():
    data = {"host": "10.0.0.1", "port": 6379, "database": "mygraph"}
    config = FalkorDBConfig(**data)
    assert config.host == "10.0.0.1"
    assert config.database == "mygraph"


def test_auth_requires_both_username_and_password():
    """Setting only username or only password should raise."""
    import pytest

    with pytest.raises(ValueError, match="Both 'username' and 'password'"):
        FalkorDBConfig(username="admin")
    with pytest.raises(ValueError, match="Both 'username' and 'password'"):
        FalkorDBConfig(password="secret")


def test_auth_both_provided():
    """Both username and password together should work."""
    config = FalkorDBConfig(username="admin", password="secret")
    assert config.username == "admin"
    assert config.password == "secret"
