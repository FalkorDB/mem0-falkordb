# mem0-falkordb

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
# 1. Register the FalkorDB plugin
from mem0_falkordb import register
register()

# 2. Use Mem0 as normal with the "falkordb" provider
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
| `database`  | str    | `mem0`      | Graph name in FalkorDB             |
| `username`  | str    | `None`      | Authentication username (optional) |
| `password`  | str    | `None`      | Authentication password (optional) |
| `base_label`| bool   | `True`      | Use `__Entity__` base label        |

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
