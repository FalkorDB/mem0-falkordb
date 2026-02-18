"""FalkorDB graph memory implementation for Mem0."""

import logging

from mem0.memory.utils import format_entities, sanitize_relationship_for_cypher

try:
    from falkordb import FalkorDB
except ImportError:
    raise ImportError(
        "falkordb is not installed. Please install it using pip install falkordb"
    )

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is not installed. Please install it using pip install rank-bm25"
    )

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class _FalkorDBGraphWrapper:
    """Thin wrapper around the FalkorDB client to provide a .query() interface
    consistent with what the MemoryGraph methods expect (list-of-dict results)."""

    def __init__(self, host, port, database, username=None, password=None):
        connect_kwargs = {"host": host, "port": port}
        if username and password:
            connect_kwargs["username"] = username
            connect_kwargs["password"] = password
        self._db = FalkorDB(**connect_kwargs)
        self._graph = self._db.select_graph(database)

    def query(self, cypher, params=None):
        """Execute a Cypher query and return results as a list of dicts."""
        result = self._graph.query(cypher, params=params)
        if not result.result_set:
            return []
        header = result.header
        return [dict(zip(header, row)) for row in result.result_set]


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = _FalkorDBGraphWrapper(
            host=self.config.graph_store.config.host,
            port=self.config.graph_store.config.port,
            database=self.config.graph_store.config.database,
            username=self.config.graph_store.config.username,
            password=self.config.graph_store.config.password,
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )

        self.use_base_label = getattr(
            self.config.graph_store.config, "base_label", True
        )
        self.node_label = ":`__Entity__`" if self.use_base_label else ""

        if self.use_base_label:
            self._ensure_indexes()

        # Determine embedding dimension for vector index
        self._embedding_dim = None

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if (
            self.config.graph_store
            and self.config.graph_store.llm
            and self.config.graph_store.llm.provider
        ):
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if (
            self.config.graph_store
            and self.config.graph_store.llm
            and hasattr(self.config.graph_store.llm, "config")
        ):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        self.threshold = (
            self.config.graph_store.threshold
            if hasattr(self.config.graph_store, "threshold")
            else 0.7
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _ensure_indexes(self):
        """Create property indexes in FalkorDB. Silently ignores if they already exist."""
        label = "__Entity__"
        for prop in ("user_id", "name"):
            try:
                self.graph.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
            except Exception:
                pass

    def _ensure_vector_index(self, dim):
        """Create vector index if not already created."""
        if self._embedding_dim == dim:
            return
        label = "__Entity__" if self.use_base_label else "Node"
        try:
            self.graph.query(
                f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding) "
                f"OPTIONS {{dimension: {dim}, similarityFunction: 'cosine'}}"
            )
        except Exception:
            # Index may already exist
            pass
        self._embedding_dim = dim

    # ------------------------------------------------------------------
    # Public API (matches Mem0 graph store interface)
    # ------------------------------------------------------------------

    def add(self, data, filters):
        """Add data to the graph."""
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(
            data, filters, entity_type_map
        )
        search_output = self._search_graph_db(
            node_list=list(entity_type_map.keys()), filters=filters
        )
        to_be_deleted = self._get_delete_entities_from_search_output(
            search_output, data, filters
        )

        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """Search for memories and related graph data."""
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(
            node_list=list(entity_type_map.keys()), filters=filters
        )

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]]
            for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append(
                {"source": item[0], "relationship": item[1], "destination": item[2]}
            )

        logger.info(f"Returned {len(search_results)} search results")
        return search_results

    def delete_all(self, filters):
        """Delete all entities and relationships for the given filters."""
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        cypher = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """Retrieve all nodes and relationships from the graph."""
        params = {"user_id": filters["user_id"], "limit": limit}

        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        node_props_str = ", ".join(node_props)

        query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})-[r]->(m {self.node_label} {{{node_props_str}}})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")
        return final_results

    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        return self.graph.query("MATCH (n) DETACH DELETE n")

    # ------------------------------------------------------------------
    # LLM-based entity extraction (reuses Mem0's tools)
    # ------------------------------------------------------------------

    def _retrieve_nodes_from_data(self, data, filters):
        """Extract all entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}
        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {
            k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
            for k, v in entity_type_map.items()
        }
        logger.debug(f"Entity type map: {entity_type_map}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            system_content = system_content.replace(
                "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}",
                },
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = (
                extracted_entities["tool_calls"][0]
                .get("arguments", {})
                .get("entities", [])
            )

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(
            search_output_string, data, user_identity
        )

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [DELETE_MEMORY_STRUCT_TOOL_GRAPH]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    # ------------------------------------------------------------------
    # FalkorDB-specific Cypher: graph search with vector similarity
    # ------------------------------------------------------------------

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes and their incoming/outgoing relations using FalkorDB vector search."""
        result_relations = []

        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)
            self._ensure_vector_index(len(n_embedding))

            label = "__Entity__" if self.use_base_label else "Node"

            # Use FalkorDB's vector search procedure to find similar nodes
            vector_query = f"""
            CALL db.idx.vector.queryNodes('{label}', 'embedding', $limit, vecf32($n_embedding))
            YIELD node, score
            WITH node, score
            WHERE node.user_id = $user_id
            {"AND node.agent_id = $agent_id" if filters.get("agent_id") else ""}
            {"AND node.run_id = $run_id" if filters.get("run_id") else ""}
            AND score >= $threshold
            WITH node, score
            ORDER BY score DESC
            LIMIT $limit
            RETURN id(node) AS node_id, node.name AS node_name, score
            """

            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            similar_nodes = self.graph.query(vector_query, params=params)

            # For each similar node, fetch outgoing and incoming relationships
            for sn in similar_nodes:
                node_id = sn["node_id"]

                # Outgoing relationships
                out_query = f"""
                MATCH (n {self.node_label})-[r]->(m {self.node_label} {{{node_props_str}}})
                WHERE id(n) = $node_id
                RETURN n.name AS source, id(n) AS source_id, type(r) AS relationship,
                       id(r) AS relation_id, m.name AS destination, id(m) AS destination_id
                """
                out_params = {"node_id": node_id, "user_id": filters["user_id"]}
                if filters.get("agent_id"):
                    out_params["agent_id"] = filters["agent_id"]
                if filters.get("run_id"):
                    out_params["run_id"] = filters["run_id"]
                out_results = self.graph.query(out_query, params=out_params)

                # Incoming relationships
                in_query = f"""
                MATCH (n {self.node_label})<-[r]-(m {self.node_label} {{{node_props_str}}})
                WHERE id(n) = $node_id
                RETURN m.name AS source, id(m) AS source_id, type(r) AS relationship,
                       id(r) AS relation_id, n.name AS destination, id(n) AS destination_id
                """
                in_results = self.graph.query(in_query, params=out_params)

                result_relations.extend(out_results)
                result_relations.extend(in_results)

        # Deduplicate by relation_id
        seen = set()
        unique_results = []
        for r in result_relations:
            rid = r.get("relation_id")
            if rid not in seen:
                seen.add(rid)
                unique_results.append(r)

        return unique_results

    # ------------------------------------------------------------------
    # FalkorDB-specific Cypher: entity deletion
    # ------------------------------------------------------------------

    def _delete_entities(self, to_be_deleted, filters):
        """Delete entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }
            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            cypher = f"""
            MATCH (n {self.node_label} {{{source_props_str}}})
            -[r:{relationship}]->
            (m {self.node_label} {{{dest_props_str}}})
            DELETE r
            RETURN
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """
            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # FalkorDB-specific Cypher: entity addition with vector embeddings
    # ------------------------------------------------------------------

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add new entities to the graph. Merge nodes if they already exist."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_added:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            source_type = entity_type_map.get(source, "__User__")
            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""
            destination_type = entity_type_map.get(destination, "__User__")
            destination_label = (
                self.node_label if self.node_label else f":`{destination_type}`"
            )
            destination_extra_set = (
                f", destination:`{destination_type}`" if self.node_label else ""
            )

            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)
            self._ensure_vector_index(len(source_embedding))

            source_node = self._search_node_by_embedding(
                source_embedding, filters, "source_candidate"
            )
            dest_node = self._search_node_by_embedding(
                dest_embedding, filters, "destination_candidate"
            )

            if not dest_node and source_node:
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (source)
                WHERE id(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1,
                    destination.embedding = vecf32($destination_embedding)
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1,
                    destination.embedding = vecf32($destination_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """
                params = {
                    "source_id": source_node,
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif dest_node and not source_node:
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (destination)
                WHERE id(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1,
                    source.embedding = vecf32($source_embedding)
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1,
                    source.embedding = vecf32($source_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """
                params = {
                    "destination_id": dest_node,
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif source_node and dest_node:
                cypher = f"""
                MATCH (source)
                WHERE id(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MATCH (destination)
                WHERE id(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1
                ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """
                params = {
                    "source_id": source_node,
                    "destination_id": dest_node,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            else:
                # Neither node exists — create both
                source_props = ["name: $source_name", "user_id: $user_id"]
                dest_props = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_props.append("agent_id: $agent_id")
                    dest_props.append("agent_id: $agent_id")
                if run_id:
                    source_props.append("run_id: $run_id")
                    dest_props.append("run_id: $run_id")
                source_props_str = ", ".join(source_props)
                dest_props_str = ", ".join(dest_props)

                cypher = f"""
                MERGE (source {source_label} {{{source_props_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1,
                            source.embedding = vecf32($source_embedding)
                            {source_extra_set}
                ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1,
                            source.embedding = vecf32($source_embedding)
                WITH source
                MERGE (destination {destination_label} {{{dest_props_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1,
                            destination.embedding = vecf32($dest_embedding)
                            {destination_extra_set}
                ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1,
                            destination.embedding = vecf32($dest_embedding)
                WITH source, destination
                MERGE (source)-[rel:{relationship}]->(destination)
                ON CREATE SET rel.created = timestamp(), rel.mentions = 1
                ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
                RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
                """
                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # FalkorDB-specific Cypher: node search by embedding similarity
    # ------------------------------------------------------------------

    def _search_node_by_embedding(self, embedding, filters, alias="candidate"):
        """Search for a node by embedding similarity.

        Returns the node id (integer) if found, or None.
        Uses FalkorDB's db.idx.vector.queryNodes procedure.
        """
        label = "__Entity__" if self.use_base_label else "Node"

        cypher = f"""
        CALL db.idx.vector.queryNodes('{label}', 'embedding', 10, vecf32($embedding))
        YIELD node, score
        WITH node, score
        WHERE node.user_id = $user_id
        {"AND node.agent_id = $agent_id" if filters.get("agent_id") else ""}
        {"AND node.run_id = $run_id" if filters.get("run_id") else ""}
        AND score >= $threshold
        ORDER BY score DESC
        LIMIT 1
        RETURN id(node) AS node_id
        """

        params = {
            "embedding": embedding,
            "user_id": filters["user_id"],
            "threshold": self.threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        if result:
            return result[0]["node_id"]
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = sanitize_relationship_for_cypher(
                item["relationship"].lower().replace(" ", "_")
            )
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list
