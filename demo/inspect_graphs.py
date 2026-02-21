"""Graph inspector for mem0-falkordb.

This script connects directly to FalkorDB and displays the raw graph structure
for each user's memory graph. Useful for understanding how memories are stored
as nodes and relationships.

Prerequisites:
- FalkorDB running on localhost:6379
- At least one user's memories created (run demo.py first)

Usage:
    python inspect_graphs.py
"""

from typing import Dict, List

from falkordb import FalkorDB
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


def get_all_mem0_graphs(db: FalkorDB, database_prefix: str = "mem0") -> List[str]:
    """List all graphs with the mem0 prefix."""
    all_graphs = db.list_graphs()
    return [g for g in all_graphs if g.startswith(f"{database_prefix}_")]


def extract_user_id(graph_name: str, database_prefix: str = "mem0") -> str:
    """Extract user_id from graph name (e.g., 'mem0_alice' -> 'alice')."""
    prefix = f"{database_prefix}_"
    if graph_name.startswith(prefix):
        return graph_name[len(prefix) :]
    return graph_name


def get_graph_nodes(db: FalkorDB, graph_name: str) -> List[Dict]:
    """Get all nodes from a graph."""
    graph = db.select_graph(graph_name)
    result = graph.query("MATCH (n) RETURN id(n) AS id, labels(n) AS labels, n AS node")

    nodes = []
    if result.result_set:
        # FalkorDB headers are [column_type, column_name] pairs
        header = [h[1] if isinstance(h, (list, tuple)) else h for h in result.header]
        for row in result.result_set:
            node_data = dict(zip(header, row))
            nodes.append(node_data)

    return nodes


def get_graph_relationships(db: FalkorDB, graph_name: str) -> List[Dict]:
    """Get all relationships from a graph."""
    graph = db.select_graph(graph_name)
    result = graph.query(
        """
        MATCH (a)-[r]->(b)
        RETURN id(a) AS source_id, a.name AS source,
               type(r) AS relationship,
               id(b) AS target_id, b.name AS target,
               r AS rel_props
        """
    )

    relationships = []
    if result.result_set:
        header = [h[1] if isinstance(h, (list, tuple)) else h for h in result.header]
        for row in result.result_set:
            rel_data = dict(zip(header, row))
            relationships.append(rel_data)

    return relationships


def format_node_properties(node) -> str:
    """Format node properties for display."""
    if not hasattr(node, "properties"):
        return ""

    props = node.properties
    # Filter out embedding vectors (too long to display)
    display_props = {k: v for k, v in props.items() if k != "embedding"}

    if not display_props:
        return ""

    prop_strs = [f"{k}={repr(v)}" for k, v in display_props.items()]
    return ", ".join(prop_strs)


def display_user_graph(
    db: FalkorDB, graph_name: str, user_id: str, detailed: bool = False
) -> None:
    """Display graph structure for a single user."""
    console.print(f"\n[bold cyan]User: {user_id}[/bold cyan]")
    console.print(f"[dim]Graph: {graph_name}[/dim]\n")

    # Get nodes and relationships
    nodes = get_graph_nodes(db, graph_name)
    relationships = get_graph_relationships(db, graph_name)

    if not nodes and not relationships:
        console.print("[yellow]Empty graph - no data[/yellow]")
        return

    # Display summary stats
    console.print(f"[green]Nodes:[/green] {len(nodes)}")
    console.print(f"[green]Relationships:[/green] {len(relationships)}\n")

    # Create node table
    if nodes:
        node_table = Table(
            title=f"Nodes in {user_id}'s Graph",
            show_header=True,
            header_style="bold yellow",
        )
        node_table.add_column("ID", style="dim")
        node_table.add_column("Labels", style="cyan")
        node_table.add_column("Name", style="green")
        if detailed:
            node_table.add_column("Properties", style="dim", no_wrap=False)

        for node_data in nodes[:20]:  # Limit to first 20 nodes
            node = node_data.get("node")
            labels_list = node_data.get("labels", [])
            node_id = str(node_data.get("id", "?"))

            if hasattr(node, "properties"):
                name = node.properties.get("name", "(unnamed)")
            else:
                name = "(unnamed)"

            labels_str = ", ".join(labels_list) if labels_list else "(no label)"

            if detailed:
                props = format_node_properties(node)
                node_table.add_row(node_id, labels_str, name, props)
            else:
                node_table.add_row(node_id, labels_str, name)

        console.print(node_table)

    # Create relationship table
    if relationships:
        rel_table = Table(
            title=f"Relationships in {user_id}'s Graph",
            show_header=True,
            header_style="bold magenta",
        )
        rel_table.add_column("Source", style="cyan")
        rel_table.add_column("Relationship", style="yellow")
        rel_table.add_column("Target", style="green")
        if detailed:
            rel_table.add_column("Mentions", style="dim")

        for rel in relationships[:20]:  # Limit to first 20 relationships
            source = rel.get("source", "(unknown)")
            target = rel.get("target", "(unknown)")
            rel_type = rel.get("relationship", "?")

            if detailed:
                rel_props = rel.get("rel_props")
                mentions = "?"
                if hasattr(rel_props, "properties"):
                    mentions = str(rel_props.properties.get("mentions", "?"))
                rel_table.add_row(source, rel_type, target, mentions)
            else:
                rel_table.add_row(source, rel_type, target)

        console.print(rel_table)

    # Create a tree visualization (simplified)
    if relationships:
        console.print(f"\n[bold]Graph Structure Preview for {user_id}:[/bold]")
        tree = Tree(f"[bold cyan]{user_id}[/bold cyan]")

        # Group relationships by source
        rel_by_source = {}
        for rel in relationships[:10]:  # Limit for readability
            source = rel.get("source", "?")
            if source not in rel_by_source:
                rel_by_source[source] = []
            rel_by_source[source].append(rel)

        for source, rels in list(rel_by_source.items())[:5]:  # Show max 5 sources
            source_branch = tree.add(f"[cyan]{source}[/cyan]")
            for rel in rels[:3]:  # Max 3 relationships per source
                rel_type = rel.get("relationship", "?")
                target = rel.get("target", "?")
                source_branch.add(
                    f"--[[yellow]{rel_type}[/yellow]]--> [green]{target}[/green]"
                )

        console.print(tree)


def main() -> None:
    """Inspect all mem0-falkordb graphs."""
    console.print(
        Panel.fit(
            "[bold magenta]mem0-falkordb Graph Inspector[/bold magenta]\n\n"
            "View raw graph structure for each user's memory",
            border_style="magenta",
        )
    )

    # Connect to FalkorDB
    try:
        db = FalkorDB(host="localhost", port=6379)
        console.print("\n[green]✓ Connected to FalkorDB[/green]")
    except Exception as e:
        console.print(
            f"[red]Failed to connect to FalkorDB:[/red]\n{e}\n\n"
            "[yellow]Make sure FalkorDB is running:[/yellow]\n"
            "  docker compose up -d"
        )
        return

    # Find all mem0 graphs
    mem0_graphs = get_all_mem0_graphs(db, database_prefix="mem0")

    if not mem0_graphs:
        console.print(
            "\n[yellow]No mem0 graphs found![/yellow]\n"
            "Run 'python demo.py' first to create some user memories."
        )
        return

    console.print(f"\n[cyan]Found {len(mem0_graphs)} user graph(s):[/cyan]")
    for graph_name in mem0_graphs:
        user_id = extract_user_id(graph_name, "mem0")
        console.print(f"  - {graph_name} ([bold]{user_id}[/bold])")

    # Display each user's graph
    for graph_name in mem0_graphs:
        user_id = extract_user_id(graph_name, "mem0")
        display_user_graph(db, graph_name, user_id, detailed=True)

    # Summary
    console.print(
        Panel.fit(
            "[bold green]Inspection Complete![/bold green]\n\n"
            "[cyan]Key Observations:[/cyan]\n"
            "• Each user has a separate FalkorDB graph\n"
            "• Nodes represent entities (people, places, concepts)\n"
            "• Relationships show how entities are connected\n"
            "• Properties store metadata (created, mentions, embeddings)\n\n"
            "[dim]This is the power of graph-structured memory![/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
