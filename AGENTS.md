# Project Guidelines

## Overview
mem0-falkordb is a FalkorDB graph store plugin for [Mem0](https://github.com/mem0ai/mem0). It adds FalkorDB as a graph memory backend **without modifying any Mem0 source code** by using runtime patching to register FalkorDB into Mem0's existing factory system.

## Build & Install
```bash
pip install -e ".[dev]"  # install in development mode with dev dependencies
```

Or for production:
```bash
pip install mem0-falkordb
```

## Testing
Tests require a running FalkorDB instance on `localhost:6379`:
```bash
docker run --rm -p 6379:6379 falkordb/falkordb:edge
```

Run all tests (excluding e2e):
```bash
pytest -v --ignore=tests/test_e2e.py
```

Run e2e tests (requires FalkorDB):
```bash
pytest tests/test_e2e.py -v
```

Run with coverage:
```bash
pytest --cov --cov-report=xml
```

## Pre-commit Checks
Always run these checks before every commit:
```bash
ruff format --check .
ruff check .
```

If formatting fails, fix with:
```bash
ruff format .
```

## Code Style
- **Formatter/linter**: Ruff (line length 88, target Python 3.10)
- **Lint rules**: `F` (Pyflakes), `E`/`W` (pycodestyle), `I` (isort)
- **Python**: requires >= 3.10; CI tests 3.10 through 3.14

## Project Structure
```
src/mem0_falkordb/
  __init__.py         # Main entry point — exposes register() function
  config.py           # FalkorDBConfig — Pydantic model for configuration
  graph_memory.py     # MemoryGraph — core implementation translating Mem0 operations to FalkorDB Cypher
  patch.py            # register() — runtime patching to add FalkorDB to Mem0's factory system

tests/
  test_config.py      # Configuration model validation tests
  test_patch.py       # Factory registration and patching tests
  test_graph_memory.py # Core graph operations tests
  test_e2e.py         # End-to-end integration tests (requires FalkorDB instance)
```

## Architecture Patterns

### Runtime Patching
The plugin uses Python's runtime patching to integrate with Mem0 without modifying its source:
1. `GraphStoreFactory.provider_to_class` gets a new `"falkordb"` entry pointing to `MemoryGraph`
2. `GraphStoreConfig` is patched to accept `FalkorDBConfig`
3. The `register()` function must be called before creating a Mem0 `Memory` instance
4. Safe to call `register()` multiple times (uses global flag)

### Per-User Graph Isolation
Each user automatically gets their own isolated FalkorDB graph (e.g., `mem0_alice`, `mem0_bob`):
- Graph name format: `{database}_{user_id}`
- Leverages FalkorDB's native multi-graph support
- No user_id filtering needed in Cypher queries
- Simple cleanup: `delete_all` drops the user's graph

### Key Cypher Translations
FalkorDB uses different Cypher syntax compared to Neo4j:
| Neo4j | FalkorDB |
|-------|----------|
| `elementId(n)` | `id(n)` |
| `vector.similarity.cosine()` | `db.idx.vector.queryNodes()` procedure |
| `db.create.setNodeVectorProperty()` | `SET n.embedding = vecf32($vec)` |
| `CALL { ... UNION ... }` subqueries | Separate outgoing + incoming queries |
| Parameterized `LIMIT $param` | Literal integers only (parser limitation) |

### Important Implementation Notes
- **FalkorDB result headers**: `result.header` returns `[[column_type, column_name], ...]` pairs, not plain strings. Extract names with `h[1]`.
- **No parameterized LIMIT**: FalkorDB's Cypher parser does not support `LIMIT $param`. Use literal integers instead.
- **Vector embeddings**: Use `vecf32()` to create vector properties, not the Neo4j property setter procedures.

## CI/CD
- **`ci.yml`**:
  - Test job: Runs unit tests with coverage (excludes e2e tests) on Python 3.10-3.14
  - E2E job: Spins up FalkorDB service container and runs integration tests
  - Lint job: Uses Ruff for formatting and linting
  - Build job: Tests package building
  - Codecov integration for Python 3.12
- **`publish.yml`**: Triggered on GitHub releases, builds and publishes to PyPI

## Before Finishing a Task
After completing any task, review whether your changes require updates to:
- **`README.md`** — if public API, usage examples, or installation instructions changed
- **`AGENTS.md`** — if project structure, build commands, architecture patterns, or conventions changed
