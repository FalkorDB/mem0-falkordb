# mem0-falkordb Multi-User Demo

This demo showcases the key advantages of **mem0-falkordb**: graph-structured memory, per-user graph isolation, and context-aware retrieval — all in a compelling, runnable script.

## What This Demo Demonstrates

### 1. Graph-Structured Memory
Memories are stored as a knowledge graph with nodes (entities) and relationships, not just flat facts. This enables rich, contextual retrieval.

### 2. Per-User Graph Isolation
Each user gets their own FalkorDB graph (`mem0_alice`, `mem0_bob`, etc.). This provides:
- **Native isolation** — no `WHERE user_id=X` filtering needed in queries
- **Simpler, faster queries** — architectural separation instead of logical filtering
- **Easy cleanup** — drop a user's graph to delete all their data

### 3. Memory Evolution
New information updates or extends the existing graph. Conflicts are resolved intelligently.

### 4. Contextual Search
Queries retrieve semantically relevant memories using vector embeddings and graph traversal.

### 5. Visual Inspection
See the actual FalkorDB graph structure — nodes, relationships, and properties — for each user.

---

## Prerequisites

- **Docker** (for running FalkorDB)
- **Python 3.10+**
- **OpenAI API Key** (for LLM and embeddings)

---

## Quick Start

### 1. Start FalkorDB

```bash
docker run --rm -p 6379:6379 falkordb/falkordb:latest
```

This starts FalkorDB on `localhost:6379`.

### 2. Install Python Dependencies

Using uv (recommended):
```bash
cd demo
uv sync
```

Alternative using pip:
```bash
cd demo
pip install -e ..  # Install local mem0-falkordb package in editable mode
pip install mem0ai openai falkordb rich
```

### 3. Set Your OpenAI API Key

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 4. Run the Demo

```bash
uv run python demo.py
```

Or using standard Python:
```bash
python demo.py
```

The demo will run through 5 scenes automatically, showcasing different aspects of mem0-falkordb.

### 5. Inspect the Raw Graphs (Optional)

After running the demo, inspect the actual graph structure:

```bash
uv run python inspect_graphs.py
```

Or using standard Python:
```bash
python inspect_graphs.py
```

This shows nodes, relationships, and properties for each user's memory graph.

---

## Demo Scenes

### Scene 1: Onboarding Multiple Users 👥

Creates 3 distinct users (`alice`, `bob`, `carol`) with different personalities, jobs, and preferences. Each user's memories are stored in a separate FalkorDB graph.

**Example:**
- **Alice**: Vegan software engineer, loves hiking, allergic to nuts
- **Bob**: Italian chef, coaches soccer, looking for restaurant software
- **Carol**: Cardiologist, marathon runner, researching AI in diagnostics

### Scene 2: Context-Aware Memory Retrieval 🔍

Runs semantic queries for each user and shows:
- Results are user-specific (no cross-contamination)
- Graph relationships surface richer results than keyword search
- The actual graph structure (nodes and relationships) for each user

**Example Queries:**
- "what should I cook for dinner?" (for Alice, a vegan)
- "what does he do on weekends?" (for Bob, a soccer coach)
- "what is her research focus?" (for Carol, studying AI)

### Scene 3: Memory Update & Conflict Resolution 🔄

Updates a user's preference and shows how the graph evolves:

**Example:**
- Alice was vegan
- Add: "Actually I moved to a pescatarian diet now"
- Query again: Results reflect the updated diet

### Scene 4: Per-User Graph Isolation Proof 🔒

Explicitly demonstrates zero cross-user leakage:

**Test:**
- Search for "marathons" in Alice's memories → Nothing found (correct!)
- Search for "marathons" in Carol's memories → Found (Carol runs marathons)

**Architecture:**
- Each user has their own FalkorDB graph:
  - `mem0_alice`
  - `mem0_bob`
  - `mem0_carol`
- No `WHERE user_id=X` clauses needed — isolation is architectural!

### Scene 5: Scale Simulation 📊

Creates 10 synthetic users with diverse memories to demonstrate:
- Scalability: per-user graph isolation keeps queries fast
- Total user count doesn't affect individual query performance
- Clean separation of concerns

---

## Inspecting Raw Graphs

The `inspect_graphs.py` script connects directly to FalkorDB and shows the actual knowledge graph structure:

```bash
python inspect_graphs.py
```

**Output includes:**
- **Nodes**: Entities with labels and properties (e.g., Alice, Python, Japan)
- **Relationships**: Connections between entities (e.g., Alice --[SPEAKS]--> Python)
- **Properties**: Metadata like `created`, `mentions`, `embedding` vectors
- **Tree Visualization**: Hierarchical view of relationships

**Example:**
```
User: alice
  Nodes:  5 nodes
  Relationships:  7 relationships

  alice --[IS_A]--> software_engineer
  alice --[LIKES]--> hiking
  alice --[PLANS_TO_VISIT]--> japan
  alice --[PREFERS]--> python
  alice --[ALLERGIC_TO]--> nuts
```

---

## Key Takeaways

✅ **One graph per user** — native FalkorDB multi-graph support
✅ **No WHERE user_id=X everywhere** — isolation is architectural
✅ **Relationships captured** — not just flat key-value memory
✅ **Drop a user = drop a graph** — `DELETE GRAPH mem0_alice`

---

## Configuration

The demo uses the following Mem0 configuration:

```python
config = {
    "graph_store": {
        "provider": "falkordb",
        "config": {
            "host": "localhost",
            "port": 6379,
            "database": "mem0",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-5-mini"},
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536,
        },
    },
}
```

You can customize:
- **host/port**: FalkorDB connection
- **database**: Graph name prefix (default: `mem0`)
- **llm.config.model**: OpenAI model (e.g., `gpt-4`, `gpt-3.5-turbo`)
- **embedder.config.model**: Embedding model
- **embedder.config.embedding_dims**: Embedding dimensions (1536 for text-embedding-3-small)

---

## Troubleshooting

### FalkorDB Connection Failed

**Error:** `Failed to connect to FalkorDB`

**Solution:**
```bash
# Check if FalkorDB is running
docker ps | grep falkordb

# Start FalkorDB if not running
docker run --rm -p 6379:6379 falkordb/falkordb:latest

# Check logs
docker logs $(docker ps -q --filter ancestor=falkordb/falkordb:latest)
```

### OpenAI API Key Not Set

**Error:** `OPENAI_API_KEY environment variable not set!`

**Solution:**
```bash
export OPENAI_API_KEY='your-key-here'
```

Or create a `.env` file (not included in repo):
```
OPENAI_API_KEY=your-key-here
```

### Rate Limits

If you hit OpenAI rate limits, the demo includes small delays between requests. You can also:
- Use a different model (e.g., `gpt-3.5-turbo` is faster/cheaper)
- Reduce the number of synthetic users in Scene 5

---

## Cleanup

To stop FalkorDB:

```bash
docker stop $(docker ps -q --filter ancestor=falkordb/falkordb:latest)
```

Or simply `Ctrl+C` if running in foreground mode (without `--rm -d` flags).

---

## Learn More

- [mem0-falkordb Repository](https://github.com/falkordb/mem0-falkordb)
- [Mem0 Documentation](https://docs.mem0.ai/)
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [FalkorDB Cloud (Free Tier)](https://app.falkordb.cloud)

---

## License

MIT License - see main repository for details.
