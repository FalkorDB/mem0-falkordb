"""Tests for FalkorDB MemoryGraph (mocked — no real DB needed)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_falkordb():
    """Provide a mock FalkorDB client."""
    with patch("mem0_falkordb.graph_memory.FalkorDB") as mock_db:
        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_result.header = []
        mock_graph.query.return_value = mock_result
        mock_db.return_value.select_graph.return_value = mock_graph
        yield mock_db, mock_graph


@pytest.fixture
def mock_config():
    """Provide a mock Mem0 config object."""
    config = MagicMock()
    config.graph_store.config.host = "localhost"
    config.graph_store.config.port = 6379
    config.graph_store.config.database = "mem0"
    config.graph_store.config.username = None
    config.graph_store.config.password = None
    config.graph_store.config.base_label = True
    config.graph_store.llm = None
    config.graph_store.custom_prompt = None
    config.graph_store.threshold = 0.7
    config.llm.provider = "openai"
    config.llm.config = MagicMock()
    config.embedder.provider = "openai"
    config.embedder.config = MagicMock()
    config.vector_store.config = MagicMock()
    return config


def test_memory_graph_init(mock_falkordb, mock_config):
    """MemoryGraph should initialize and connect to FalkorDB."""
    mock_db, mock_graph = mock_falkordb

    with patch("mem0_falkordb.graph_memory.EmbedderFactory") as mock_emb, \
         patch("mem0_falkordb.graph_memory.LlmFactory") as mock_llm:
        mock_emb.create.return_value = MagicMock()
        mock_llm.create.return_value = MagicMock()

        from mem0_falkordb.graph_memory import MemoryGraph

        mg = MemoryGraph(mock_config)

        assert mg.node_label == ":`__Entity__`"
        assert mg.threshold == 0.7
        mock_db.assert_called_once_with(host="localhost", port=6379)


def test_delete_all(mock_falkordb, mock_config):
    """delete_all should execute DETACH DELETE with proper filters."""
    mock_db, mock_graph = mock_falkordb

    with patch("mem0_falkordb.graph_memory.EmbedderFactory") as mock_emb, \
         patch("mem0_falkordb.graph_memory.LlmFactory") as mock_llm:
        mock_emb.create.return_value = MagicMock()
        mock_llm.create.return_value = MagicMock()

        from mem0_falkordb.graph_memory import MemoryGraph

        mg = MemoryGraph(mock_config)
        mg.delete_all({"user_id": "alice"})

        # The wrapper's query should have been called with DETACH DELETE
        calls = mg.graph._graph.query.call_args_list
        assert any("DETACH DELETE" in str(c) for c in calls)


def test_reset(mock_falkordb, mock_config):
    """reset() should clear the entire graph."""
    mock_db, mock_graph = mock_falkordb

    with patch("mem0_falkordb.graph_memory.EmbedderFactory") as mock_emb, \
         patch("mem0_falkordb.graph_memory.LlmFactory") as mock_llm:
        mock_emb.create.return_value = MagicMock()
        mock_llm.create.return_value = MagicMock()

        from mem0_falkordb.graph_memory import MemoryGraph

        mg = MemoryGraph(mock_config)
        mg.reset()

        calls = mg.graph._graph.query.call_args_list
        assert any("MATCH (n) DETACH DELETE n" in str(c) for c in calls)


def test_get_all(mock_falkordb, mock_config):
    """get_all should return formatted results."""
    mock_db, mock_graph = mock_falkordb

    # Set up mock to return a result
    mock_result = MagicMock()
    mock_result.result_set = [["alice", "likes", "pizza"]]
    mock_result.header = ["source", "relationship", "target"]
    mock_graph.query.return_value = mock_result

    with patch("mem0_falkordb.graph_memory.EmbedderFactory") as mock_emb, \
         patch("mem0_falkordb.graph_memory.LlmFactory") as mock_llm:
        mock_emb.create.return_value = MagicMock()
        mock_llm.create.return_value = MagicMock()

        from mem0_falkordb.graph_memory import MemoryGraph

        mg = MemoryGraph(mock_config)
        results = mg.get_all({"user_id": "alice"}, limit=10)

        assert len(results) == 1
        assert results[0]["source"] == "alice"
        assert results[0]["relationship"] == "likes"
        assert results[0]["target"] == "pizza"
