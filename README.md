[![license](https://img.shields.io/github/license/falkordb/mem0-falkordb.svg)](https://github.com/falkordb/mem0-falkordb)
[![Release](https://img.shields.io/github/release/falkordb/mem0-falkordb.svg)](https://github.com/falkordb/mem0-falkordb/releases/latest)
[![PyPI version](https://badge.fury.io/py/mem0-falkordb.svg)](https://badge.fury.io/py/mem0-falkordb)
[![Codecov](https://codecov.io/gh/falkordb/mem0-falkordb/branch/main/graph/badge.svg)](https://codecov.io/gh/falkordb/mem0-falkordb)
[![Forum](https://img.shields.io/badge/Forum-falkordb-blue)](https://github.com/orgs/FalkorDB/discussions)
[![Discord](https://img.shields.io/discord/1146782921294884966?style=flat-square)](https://discord.gg/ErBEqN9E)

# mem0-falkordb

[![Try Free](https://img.shields.io/badge/Try%20Free-FalkorDB%20Cloud-FF8101?labelColor=FDE900&style=for-the-badge&link=https://app.falkordb.cloud)](https://app.falkordb.cloud)

FalkorDB graph store plugin for [Mem0](https://github.com/mem0ai/mem0). Adds FalkorDB as a graph memory backend **without modifying any Mem0 source code**.

## Installation

```bash
pip install mem0-falkordb
```

You also need Mem0 installed separately:

```bash
pip install mem0ai
```

## Quick Start

```python
from mem0_falkordb import register
register()

from mem0 import Memory

config = {
    "graph_store": {
        "provider": "falkordb",
        "config": {
            "host": "localhost",
            "port": 6379,
            "database": "mem0",
        },
    },
    # Add your LLM and embedder config as usual
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"},
    },
}

m = Memory.from_config(config)
m.add("I love pizza", user_id="alice")
results = m.search("what does alice like?", user_id="alice")
```

## Configuration

| Parameter    | Type   | Default     | Description                        |
|-------------|--------|-------------|------------------------------------|
| `host`      | str    | `localhost` | FalkorDB server host               |
| `port`      | int    | `6379`      | FalkorDB server port               |
| `database`  | str    | `mem0`      | Graph name prefix in FalkorDB      |
| `username`  | str    | `None`      | Authentication username (optional) |
| `password`  | str    | `None`      | Authentication password (optional) |
| `base_label`| bool   | `True`      | Use `__Entity__` base label        |
| `multi_graph`| bool  | `True`      | Use a separate graph per user_id   |

### Multi-Graph Mode (default)

When `multi_graph=True`, each user gets their own isolated FalkorDB graph (e.g. `mem0_alice`, `mem0_bob`). This provides:

- **Natural data isolation** — no user_id filtering needed in Cypher queries
- **Simpler, faster queries** — no WHERE clauses on user_id
- **Easy cleanup** — `delete_all` simply drops the user's graph
- **Leverages FalkorDB's native multi-graph support**

To disable and use a single shared graph with user_id property filtering (similar to Neo4j behavior):

```python
config = {
    "graph_store": {
        "provider": "falkordb",
        "config": {
            "host": "localhost",
            "port": 6379,
            "database": "mem0",
            "multi_graph": False,
        },
    },
}
```

## Running FalkorDB

Using Docker:

```bash
docker run --rm -p 6379:6379 falkordb/falkordb
```

## How It Works

This plugin uses Python's runtime patching to register FalkorDB into Mem0's existing factory system:

1. `GraphStoreFactory.provider_to_class` gets a new `"falkordb"` entry
2. `GraphStoreConfig` is patched to accept `FalkorDBConfig`
3. A `MemoryGraph` class translates Mem0's graph operations to FalkorDB-compatible Cypher

### Key Cypher Translations

| Neo4j                                    | FalkorDB                                          |
|------------------------------------------|---------------------------------------------------|
| `elementId(n)`                           | `id(n)`                                           |
| `vector.similarity.cosine()`             | `db.idx.vector.queryNodes()` procedure            |
| `db.create.setNodeVectorProperty()`      | `SET n.embedding = vecf32($vec)`                  |
| `CALL { ... UNION ... }` subqueries      | Separate outgoing + incoming queries              |

## Development

```bash
git clone <repo>
cd mem0-falkordb
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0
