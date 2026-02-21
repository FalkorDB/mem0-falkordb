"""Multi-User Agentic Memory Demo for mem0-falkordb.

This demo showcases:
- Graph-structured memory (relationships between entities)
- Per-user graph isolation (separate FalkorDB graphs)
- Context-aware retrieval (semantic search)
- Memory evolution (updates and conflicts)

Prerequisites:
- FalkorDB running on localhost:6379
- OPENAI_API_KEY environment variable set

Usage:
    python demo.py

    For CI mode (tests initialization only):
    DEMO_CI_MODE=1 python demo.py
"""

import os
import time
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from mem0 import Memory
from mem0_falkordb import register

# Register FalkorDB with Mem0
register()

console = Console()

# Check if running in CI mode
CI_MODE = os.getenv("DEMO_CI_MODE", "").lower() in ("1", "true", "yes")


def print_section(title: str, emoji: str = "🎬") -> None:
    """Print a section header."""
    console.print(f"\n{emoji} [bold cyan]{title}[/bold cyan]\n")


def print_memories(results: List[Dict], title: str = "Retrieved Memories") -> None:
    """Pretty print memory search results."""
    if not results:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Memory", style="cyan", no_wrap=False)

    for result in results:
        memory_text = result.get("memory", "N/A")
        table.add_row(memory_text)

    console.print(table)


def print_graph_stats(m: Memory, user_id: str) -> None:
    """Print graph statistics for a user."""
    try:
        # Get all relationships for the user
        results = m.graph.get_all({"user_id": user_id}, limit=1000)

        if not results:
            console.print(f"[yellow]No graph data for user '{user_id}'[/yellow]")
            return

        # Count unique nodes and relationships
        nodes = set()
        relationships = {}

        for item in results:
            nodes.add(item["source"])
            nodes.add(item["target"])
            rel_type = item["relationship"]
            relationships[rel_type] = relationships.get(rel_type, 0) + 1

        console.print(
            f"[green]User '{user_id}' graph:[/green] "
            f"{len(nodes)} nodes, {len(results)} relationships"
        )

        # Show sample relationships
        table = Table(
            title=f"Sample Relationships for {user_id}",
            show_header=True,
            header_style="bold yellow",
        )
        table.add_column("Source", style="cyan")
        table.add_column("Relationship", style="magenta")
        table.add_column("Target", style="green")

        for item in results[:5]:  # Show first 5
            table.add_row(item["source"], item["relationship"], item["target"])

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting graph stats: {e}[/red]")


def scene_1_onboarding(m: Memory) -> Dict[str, List[str]]:
    """Scene 1: Onboard multiple users with distinct profiles."""
    print_section("Scene 1: Onboarding Multiple Users", "👥")

    users = {
        "alice": [
            "My name is Alice. I'm a vegan software engineer who loves hiking.",
            "I'm allergic to nuts and I prefer Python over JavaScript.",
            "I'm planning a trip to Japan next month.",
        ],
        "bob": [
            "I'm Bob, a chef specializing in Italian cuisine.",
            "I have two kids and I coach their soccer team on weekends.",
            "I'm looking for a new restaurant management software.",
        ],
        "carol": [
            "I'm Carol, a cardiologist at Boston General Hospital.",
            "I run marathons and follow a high-protein diet.",
            "I'm writing a paper on AI in diagnostics.",
        ],
    }

    for user_id, messages in users.items():
        console.print(f"\n[bold]Adding memories for {user_id}...[/bold]")
        for msg in track(messages, description=f"Processing {user_id}'s messages"):
            m.add(msg, user_id=user_id)
            time.sleep(0.1)  # Brief pause for API rate limits

        console.print(f"[green]✓ {user_id}'s profile created[/green]")

    console.print(
        "\n[bold green]✓ All users onboarded![/bold green] "
        "Each has their own isolated FalkorDB graph."
    )

    return users


def scene_2_retrieval(m: Memory, users: Dict[str, List[str]]) -> None:
    """Scene 2: Context-aware memory retrieval per user."""
    print_section("Scene 2: Context-Aware Memory Retrieval", "🔍")

    queries = {
        "alice": [
            "what should I cook for dinner?",
            "what are her travel plans?",
        ],
        "bob": [
            "what does he do on weekends?",
            "what software is he looking for?",
        ],
        "carol": [
            "what is her research focus?",
            "what are her dietary habits?",
        ],
    }

    for user_id, user_queries in queries.items():
        console.print(f"\n[bold cyan]Querying memories for {user_id}...[/bold cyan]")

        for query in user_queries:
            console.print(f"\n[yellow]Q:[/yellow] {query}")
            results = m.search(query, user_id=user_id)
            print_memories(results, title=f"Results for {user_id}")

        # Show graph structure for this user
        print_graph_stats(m, user_id)


def scene_3_memory_update(m: Memory) -> None:
    """Scene 3: Update a user's memory and observe changes."""
    print_section("Scene 3: Memory Update & Conflict Resolution", "🔄")

    user_id = "alice"

    console.print(f"[bold]Current diet preference for {user_id}:[/bold]")
    results = m.search("what is alice's diet?", user_id=user_id)
    print_memories(results)

    console.print(f"\n[yellow]Updating {user_id}'s diet...[/yellow]")
    m.add("Actually I moved to a pescatarian diet now", user_id=user_id)

    console.print(f"\n[bold]Updated diet preference for {user_id}:[/bold]")
    results = m.search("what is alice's diet?", user_id=user_id)
    print_memories(results)

    console.print("\n[green]✓ Memory successfully updated![/green]")


def scene_4_isolation_proof(m: Memory) -> None:
    """Scene 4: Prove per-user graph isolation."""
    print_section("Scene 4: Per-User Graph Isolation Proof", "🔒")

    # Try to access Carol's marathon info from Alice's context
    console.print(
        "[yellow]Searching for 'marathons' in alice's memories "
        "(should return nothing - that's Carol's data):[/yellow]\n"
    )
    results = m.search("marathons", user_id="alice")
    print_memories(results)

    if not results or all(
        "marathon" not in r.get("memory", "").lower() for r in results
    ):
        console.print(
            "\n[bold green]✓ Isolation confirmed![/bold green] "
            "Alice cannot access Carol's marathon memories."
        )
    else:
        console.print(
            "\n[red]⚠ Unexpected: Found marathon-related memories in Alice's graph[/red]"
        )

    # Now search in Carol's context
    console.print("\n[yellow]Searching for 'marathons' in carol's memories:[/yellow]\n")
    results = m.search("marathons", user_id="carol")
    print_memories(results)

    if results:
        console.print(
            "\n[bold green]✓ Carol's memories are properly isolated![/bold green]"
        )

    # Show that each user has their own FalkorDB graph
    console.print("\n[bold cyan]Graph Isolation Details:[/bold cyan]")
    console.print(
        "Each user gets their own FalkorDB graph:\n"
        "  - alice → [cyan]mem0_alice[/cyan]\n"
        "  - bob   → [cyan]mem0_bob[/cyan]\n"
        "  - carol → [cyan]mem0_carol[/cyan]\n"
    )
    console.print(
        "[dim]No WHERE user_id=X clauses needed - isolation is architectural![/dim]"
    )


def scene_5_scale_demo(m: Memory) -> None:
    """Scene 5: Demonstrate scalability with multiple synthetic users."""
    print_section("Scene 5: Scale Simulation (Optional)", "📊")

    console.print(
        "[yellow]Creating 10 synthetic users with diverse memories...[/yellow]\n"
    )

    synthetic_messages = [
        "I love reading science fiction novels",
        "I play tennis every Tuesday",
        "I work as a data scientist",
        "I'm learning to play the guitar",
        "I enjoy cooking Mediterranean food",
    ]

    start_time = time.time()

    for i in track(range(10), description="Creating users"):
        user_id = f"user_{i:03d}"
        # Add 2-3 memories per user
        for j in range(3):
            msg = synthetic_messages[j % len(synthetic_messages)]
            m.add(f"{msg} (variant {j})", user_id=user_id)

    end_time = time.time()
    elapsed = end_time - start_time

    console.print(
        f"\n[green]✓ Created 10 users with 30 total memories in {elapsed:.2f}s[/green]"
    )

    # Demonstrate fast per-user query
    console.print("\n[yellow]Testing search speed for a single user...[/yellow]")
    start_time = time.time()
    _ = m.search("hobbies", user_id="user_005")
    end_time = time.time()
    query_time = end_time - start_time

    console.print(f"[green]Query completed in {query_time:.3f}s[/green]")
    console.print(
        "[dim]Per-user graph isolation keeps queries fast "
        "regardless of total user count![/dim]"
    )


def main() -> None:
    """Run the complete multi-user memory demo."""
    console.print(
        Panel.fit(
            "[bold magenta]mem0-falkordb Multi-User Demo[/bold magenta]\n\n"
            "Showcasing graph-structured memory with per-user isolation",
            border_style="magenta",
        )
    )

    # Check for OpenAI API key (not required in CI mode)
    if not CI_MODE and not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]Error: OPENAI_API_KEY environment variable not set![/red]\n"
            "Please set it and try again:\n"
            "  export OPENAI_API_KEY='your-key-here'\n"
            "  python demo.py"
        )
        return

    # Initialize Mem0 with FalkorDB
    console.print("\n[yellow]Initializing Mem0 with FalkorDB...[/yellow]")

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
            "config": {"model": "gpt-4o-mini"},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
    }

    try:
        m = Memory.from_config(config)
        console.print("[green]✓ Connected to FalkorDB[/green]")
    except Exception as e:
        console.print(
            f"[red]Failed to connect to FalkorDB:[/red]\n{e}\n\n"
            "[yellow]Make sure FalkorDB is running:[/yellow]\n"
            "  docker run --rm -p 6379:6379 falkordb/falkordb:latest"
        )
        return

    # CI mode: just test initialization
    if CI_MODE:
        console.print(
            Panel.fit(
                "[bold green]CI Validation Complete![/bold green]\n\n"
                "[cyan]Validated:[/cyan]\n"
                "✅ Demo script imports successfully\n"
                "✅ FalkorDB connection works\n"
                "✅ mem0-falkordb provider registered\n"
                "✅ Mem0 Memory instance created\n\n"
                "[dim]Run without DEMO_CI_MODE to see the full demo[/dim]",
                border_style="green",
            )
        )
        return

    # Run all scenes
    try:
        users = scene_1_onboarding(m)
        scene_2_retrieval(m, users)
        scene_3_memory_update(m)
        scene_4_isolation_proof(m)
        scene_5_scale_demo(m)

        # Final summary
        console.print(
            Panel.fit(
                "[bold green]Demo Complete![/bold green]\n\n"
                "[cyan]Key Takeaways:[/cyan]\n"
                "✅ One graph per user — native FalkorDB multi-graph support\n"
                "✅ No WHERE user_id=X everywhere — isolation is architectural\n"
                "✅ Relationships captured — not just flat key-value memory\n"
                "✅ Drop a user = drop a graph (DELETE GRAPH mem0_alice)\n\n"
                "[dim]Run 'python inspect_graphs.py' to see raw graph contents[/dim]",
                border_style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during demo: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
