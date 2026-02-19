"""End-to-end tests against a real FalkorDB instance.

These tests require a running FalkorDB server (default localhost:6379).
Run with:  pytest -m e2e

In CI the server is started as a Docker service.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from mem0_falkordb.config import FalkorDBConfig

try:
    from falkordb import FalkorDB
except ImportError:
    FalkorDB = None

pytestmark = pytest.mark.e2e


def _falkordb_reachable():
    """Return True when a FalkorDB server responds on localhost:6379."""
    if FalkorDB is None:
        return False
    try:
        db = FalkorDB(host="localhost", port=6379)
        db.list_graphs()
        return True
    except Exception:
        return False


skip_no_falkordb = pytest.mark.skipif(
    not _falkordb_reachable(),
    reason="FalkorDB is not reachable on localhost:6379",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def unique_db():
    """Return a unique database prefix so tests don't collide."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def wrapper(unique_db):
    """Create a _FalkorDBGraphWrapper connected to the local FalkorDB."""
    from mem0_falkordb.graph_memory import _FalkorDBGraphWrapper

    w = _FalkorDBGraphWrapper(host="localhost", port=6379, database=unique_db)
    yield w
    # Cleanup: drop any graphs created during the test
    w.reset_all_graphs()


@pytest.fixture()
def memory_graph(unique_db):
    """Create a MemoryGraph with real FalkorDB but mocked LLM/embedder."""
    config = MagicMock()
    config.graph_store.config = FalkorDBConfig(
        host="localhost", port=6379, database=unique_db
    )
    config.graph_store.llm = None
    config.graph_store.custom_prompt = None
    config.graph_store.threshold = 0.7
    config.llm.provider = "openai"
    config.llm.config = MagicMock()
    config.embedder.provider = "openai"
    config.embedder.config = MagicMock()
    config.vector_store.config = MagicMock()

    with (
        patch("mem0_falkordb.graph_memory.EmbedderFactory") as mock_emb,
        patch("mem0_falkordb.graph_memory.LlmFactory") as mock_llm,
    ):
        mock_emb.create.return_value = MagicMock()
        mock_llm.create.return_value = MagicMock()

        from mem0_falkordb.graph_memory import MemoryGraph

        mg = MemoryGraph(config)
        yield mg
        # Cleanup
        mg.graph.reset_all_graphs()


# ---------------------------------------------------------------------------
# _FalkorDBGraphWrapper tests
# ---------------------------------------------------------------------------


@skip_no_falkordb
class TestFalkorDBGraphWrapper:
    """Tests for the low-level graph wrapper against real FalkorDB."""

    def test_connect_and_query_empty(self, wrapper):
        """Querying an empty graph returns an empty list."""
        result = wrapper.query("MATCH (n) RETURN n", user_id="u1")
        assert result == []

    def test_create_and_read_node(self, wrapper):
        """Create a node and read it back."""
        wrapper.query(
            "CREATE (n:Person {name: $name}) RETURN n.name AS name",
            params={"name": "alice"},
            user_id="u1",
        )
        result = wrapper.query("MATCH (n:Person) RETURN n.name AS name", user_id="u1")
        assert len(result) == 1
        assert result[0]["name"] == "alice"

    def test_create_relationship(self, wrapper):
        """Create two nodes and a relationship, then query the relationship."""
        wrapper.query(
            "CREATE (a:Person {name: 'alice'})-[:KNOWS]->(b:Person {name: 'bob'})",
            user_id="u1",
        )
        result = wrapper.query(
            "MATCH (a)-[r:KNOWS]->(b) "
            "RETURN a.name AS source, type(r) AS rel, b.name AS target",
            user_id="u1",
        )
        assert len(result) == 1
        assert result[0]["source"] == "alice"
        assert result[0]["rel"] == "KNOWS"
        assert result[0]["target"] == "bob"

    def test_user_isolation(self, wrapper):
        """Different user_ids get separate graphs."""
        wrapper.query("CREATE (n:Item {name: 'x'})", user_id="u1")
        wrapper.query("CREATE (n:Item {name: 'y'})", user_id="u2")

        r1 = wrapper.query("MATCH (n:Item) RETURN n.name AS name", user_id="u1")
        r2 = wrapper.query("MATCH (n:Item) RETURN n.name AS name", user_id="u2")

        assert [r["name"] for r in r1] == ["x"]
        assert [r["name"] for r in r2] == ["y"]

    def test_delete_graph(self, wrapper):
        """delete_graph removes the user's entire graph."""
        wrapper.query("CREATE (n:Item {name: 'x'})", user_id="u1")
        wrapper.delete_graph("u1")

        result = wrapper.query("MATCH (n) RETURN n", user_id="u1")
        assert result == []

    def test_reset_all_graphs(self, wrapper):
        """reset_all_graphs removes all graphs with the database prefix."""
        wrapper.query("CREATE (n:Item {name: 'a'})", user_id="u1")
        wrapper.query("CREATE (n:Item {name: 'b'})", user_id="u2")

        wrapper.reset_all_graphs()

        r1 = wrapper.query("MATCH (n) RETURN n", user_id="u1")
        r2 = wrapper.query("MATCH (n) RETURN n", user_id="u2")
        assert r1 == []
        assert r2 == []

    def test_parameterized_query(self, wrapper):
        """Parameters are correctly passed to FalkorDB."""
        wrapper.query(
            "CREATE (n:Thing {name: $name, value: $val})",
            params={"name": "widget", "val": 42},
            user_id="u1",
        )
        result = wrapper.query(
            "MATCH (n:Thing {name: $name}) RETURN n.value AS val",
            params={"name": "widget"},
            user_id="u1",
        )
        assert result[0]["val"] == 42

    def test_graph_cache_reuses_graph(self, wrapper):
        """Accessing the same user_id twice returns the cached graph object."""
        g1 = wrapper._get_graph("u1")
        g2 = wrapper._get_graph("u1")
        assert g1 is g2


# ---------------------------------------------------------------------------
# MemoryGraph integration tests (real DB, mocked LLM/embedder)
# ---------------------------------------------------------------------------


@skip_no_falkordb
class TestMemoryGraphIntegration:
    """Integration tests for MemoryGraph with real FalkorDB."""

    def test_get_all_empty(self, memory_graph):
        """get_all on a fresh graph returns an empty list."""
        result = memory_graph.get_all({"user_id": "alice"})
        assert result == []

    def test_get_all_returns_relationships(self, memory_graph):
        """get_all returns relationships inserted directly into the graph."""
        uid = "alice"
        memory_graph.graph.query(
            "CREATE (a:`__Entity__` {name: 'alice'})"
            "-[:LIKES]->"
            "(b:`__Entity__` {name: 'pizza'})",
            user_id=uid,
        )
        result = memory_graph.get_all({"user_id": uid})
        assert len(result) == 1
        assert result[0]["source"] == "alice"
        assert result[0]["relationship"] == "LIKES"
        assert result[0]["target"] == "pizza"

    def test_delete_all_drops_graph(self, memory_graph):
        """delete_all with only user_id should drop the entire graph."""
        uid = "bob"
        memory_graph.graph.query(
            "CREATE (a:`__Entity__` {name: 'bob'})"
            "-[:HAS]->"
            "(b:`__Entity__` {name: 'cat'})",
            user_id=uid,
        )
        memory_graph.delete_all({"user_id": uid})

        result = memory_graph.get_all({"user_id": uid})
        assert result == []

    def test_delete_all_scoped_by_agent_id(self, memory_graph):
        """delete_all with agent_id only deletes matching entities."""
        uid = "carol"
        mg = memory_graph

        mg.graph.query(
            "CREATE (a:`__Entity__` {name: 'carol', agent_id: 'a1'})"
            "-[:USES]->"
            "(b:`__Entity__` {name: 'tool', agent_id: 'a1'})",
            user_id=uid,
        )
        mg.graph.query(
            "CREATE (a:`__Entity__` {name: 'carol2', agent_id: 'a2'})"
            "-[:USES]->"
            "(b:`__Entity__` {name: 'tool2', agent_id: 'a2'})",
            user_id=uid,
        )

        mg.delete_all({"user_id": uid, "agent_id": "a1"})

        remaining = mg.graph.query(
            "MATCH (n:`__Entity__`) RETURN n.name AS name, n.agent_id AS aid",
            user_id=uid,
        )
        names = {r["name"] for r in remaining}
        assert "carol" not in names
        assert "carol2" in names

    def test_reset_clears_all_user_graphs(self, memory_graph):
        """reset() removes all graphs with the database prefix."""
        memory_graph.graph.query("CREATE (n:`__Entity__` {name: 'x'})", user_id="u1")
        memory_graph.graph.query("CREATE (n:`__Entity__` {name: 'y'})", user_id="u2")

        memory_graph.reset()

        assert memory_graph.get_all({"user_id": "u1"}) == []
        assert memory_graph.get_all({"user_id": "u2"}) == []

    def test_ensure_indexes_idempotent(self, memory_graph):
        """Calling _ensure_user_graph_indexes multiple times does not raise."""
        memory_graph._ensure_user_graph_indexes("idx_user")
        memory_graph._ensure_user_graph_indexes("idx_user")

    def test_get_all_with_limit(self, memory_graph):
        """get_all respects the limit parameter."""
        uid = "limited"
        for i in range(5):
            memory_graph.graph.query(
                f"CREATE (a:`__Entity__` {{name: 'src{i}'}})"
                f"-[:REL{i}]->"
                f"(b:`__Entity__` {{name: 'dst{i}'}})",
                user_id=uid,
            )
        result = memory_graph.get_all({"user_id": uid}, limit=3)
        assert len(result) == 3

    def test_node_label_matches_config(self, memory_graph):
        """When base_label is True, node_label should be :`__Entity__`."""
        assert memory_graph.node_label == ":`__Entity__`"


# ---------------------------------------------------------------------------
# Full-stack tests (register → Memory.from_config → add/search)
# Uses GitHub Models API (requires GITHUB_TOKEN env var)
# ---------------------------------------------------------------------------

_GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"


def _github_token():
    """Return GITHUB_TOKEN if set, else None."""
    import os

    return os.environ.get("GITHUB_TOKEN")


skip_no_github_token = pytest.mark.skipif(
    not _github_token(),
    reason="GITHUB_TOKEN is not set",
)


@skip_no_falkordb
@skip_no_github_token
class TestFullStack:
    """Full-stack tests: register → Memory.from_config → add/search/get_all."""

    @pytest.fixture(autouse=True)
    def _setup(self, unique_db):
        """Set up a Memory instance with FalkorDB + GitHub Models API."""
        import os

        from mem0_falkordb import register

        register()

        from mem0 import Memory

        self.user_id = f"user_{uuid.uuid4().hex[:8]}"
        config = {
            "graph_store": {
                "provider": "falkordb",
                "config": {
                    "host": "localhost",
                    "port": 6379,
                    "database": unique_db,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "openai/gpt-4.1-nano",
                    "openai_base_url": _GITHUB_MODELS_BASE_URL,
                    "api_key": os.environ["GITHUB_TOKEN"],
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "openai/text-embedding-3-small",
                    "openai_base_url": _GITHUB_MODELS_BASE_URL,
                    "api_key": os.environ["GITHUB_TOKEN"],
                },
            },
        }
        self.memory = Memory.from_config(config)
        yield
        # Cleanup
        self.memory.graph.graph.reset_all_graphs()

    def test_add_and_search(self):
        """Add a memory and search for it."""
        self.memory.add("I love pizza and pasta", user_id=self.user_id)
        results = self.memory.search("what food do I like?", user_id=self.user_id)
        assert len(results) > 0

    def test_add_and_get_all(self):
        """Add a memory and retrieve all graph relationships."""
        self.memory.add("Alice works at Acme Corp", user_id=self.user_id)
        results = self.memory.graph.get_all({"user_id": self.user_id})
        assert len(results) > 0
        # Verify the result structure
        for r in results:
            assert "source" in r
            assert "relationship" in r
            assert "target" in r

    def test_add_and_delete_all(self):
        """Add a memory, delete all, and verify it's gone."""
        self.memory.add("Bob likes hiking", user_id=self.user_id)
        self.memory.graph.delete_all({"user_id": self.user_id})
        results = self.memory.graph.get_all({"user_id": self.user_id})
        assert results == []
