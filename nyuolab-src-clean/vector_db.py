# NYU OLAB, 2023

# imports
import os
import shutil
import time

import chromadb
from tqdm import tqdm

from embeddings import *
from utils import *


# Vector Database class
class VectorDatabase:
    """
    Class for Vector Database construction and retrieval.
    Stores two databases: one for concepts, another for relationships.
    """

    def __init__(
        self,
        db_root: str,
        relationships: dict,
        concepts: dict,
        embedding_model: str = "medcpt",
        embedding_batch_size: int = 512,
        rebuild_vector_db: bool = True,
    ):
        if os.path.exists(db_root):
            if rebuild_vector_db:
                num_concepts = len(concepts) if concepts else 0
                print(
                    f"[VectorDB] rebuild=True. Removing existing DB at: {db_root}",
                    flush=True,
                )
                if num_concepts > 50000:
                    print(
                        f"[VectorDB] WARNING: {num_concepts} concepts will be re-embedded. "
                        f"This may take hours on CPU. Set rebuild_vector_db: false to reuse.",
                        flush=True,
                    )
                shutil.rmtree(db_root)
                os.makedirs(db_root)
            else:
                print(f"[VectorDB] rebuild=False. Reusing existing DB at: {db_root}", flush=True)
        else:
            os.makedirs(db_root, exist_ok=True)

        self.client = chromadb.PersistentClient(db_root)

        base_ef = self._get_embedding_function(embedding_model)
        self.embedding_function = CachedEmbeddingFunction(base_ef)
        self.batch_size = embedding_batch_size
        self.relationship_collection = self._initialize_relationship_collection(
            relationships
        )
        self.concept_collection = self._initialize_concept_collection(concepts)

        self.relationships = set(self.relationship_collection.get()["documents"])
        print("Relationships: ", self.relationships, flush=True)

        if isinstance(self.embedding_function, CachedEmbeddingFunction):
            self.embedding_function.report()

    def _get_embedding_function(self, embedding_model: str = "medcpt"):
        """
        Get embedding function from embedding model name.
        """
        if embedding_model in EMBED_MODEL_DICT:
            return EMBED_MODEL_DICT[embedding_model]
        else:
            raise NotImplementedError(
                f"embedding model {embedding_model} not implemented"
            )

    def _initialize_get_collection(
        self,
        collection_name: str,
        ef: chromadb.EmbeddingFunction = None,
        delete_existing: bool = False,
    ):
        """
        Initialize collection.
        """
        if ef is None:
            ef = self.embedding_function

        if delete_existing:
            # delete existing collection if it exists
            try:
                collection = self.client.delete_collection(collection_name)
            except:
                pass

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        return collection

    def target_match_collection(
        self,
        valid_targets: list[str],
        target_to_match: str,
    ):
        """
        Match a target to a list of valid targets.
        """
        # get or create concept/atom collection
        target_match_collection = self._initialize_get_collection(
            "target_match", self.embedding_function, delete_existing=True
        )

        # initialize list of entries
        target_match_collection.add(
            documents=valid_targets,
            ids=[hash_string(t) for t in valid_targets],
        )

        # query
        results = target_match_collection.query(
            query_texts=[string_preprocess(target_to_match)],
            n_results=1,
            include=["documents", "distances"],
        )

        matched = results["documents"][0][0]
        distance = results["distances"][0][0]

        return matched, distance

    def _initialize_concept_collection(self, concepts: dict):
        """
        Initialize concept collection.
        """
        concept_collection = self._initialize_get_collection("concepts")

        if concept_collection.count() >= 1:
            print(
                f"Initializing concept collection already containing "
                f"{concept_collection.count()} entries...",
                flush=True,
            )
            return concept_collection

        entries = []
        uids = []
        print(f"Adding {len(concepts)} concepts/atoms to collection...", flush=True)
        for k, v in tqdm(concepts.items(), desc="prep concepts"):
            entries.append(string_preprocess(k))
            uids.append({"uids": " ".join(v)})

        hashes = [hash_string(entry) for entry in entries]

        total_batches = (len(entries) + self.batch_size - 1) // self.batch_size
        t0 = time.time()
        for batch_num, i in enumerate(
            tqdm(range(0, len(entries), self.batch_size), desc="embed concepts")
        ):
            batch = entries[i : i + self.batch_size]
            batch_uids = uids[i : i + self.batch_size]
            batch_hashes = hashes[i : i + self.batch_size]

            try:
                concept_collection.add(
                    documents=batch,
                    ids=batch_hashes,
                    metadatas=batch_uids,
                )
            except Exception as e:
                print(f"Failed to add concept batch {batch_num}: {e}", flush=True)
                raise e

            if (batch_num + 1) % 20 == 0 or (batch_num + 1) == total_batches:
                elapsed = time.time() - t0
                done = min(i + self.batch_size, len(entries))
                throughput = done / elapsed if elapsed > 0 else 0
                remaining = (len(entries) - done) / throughput if throughput > 0 else 0
                print(
                    f"[concepts] {done}/{len(entries)} "
                    f"({elapsed:.0f}s elapsed, {throughput:.0f}/s, ~{remaining:.0f}s left)",
                    flush=True,
                )

        return concept_collection

    def _initialize_relationship_collection(self, relationships: dict):
        """
        Initialize relationship collection.
        """
        relationship_collection = self._initialize_get_collection("relationships")

        if relationship_collection.count() >= 1:
            print(
                f"Initializing relationship collection already containing "
                f"{relationship_collection.count()} entries...",
                flush=True,
            )
            return relationship_collection

        entries = []
        uids = []
        print(f"Adding {len(relationships)} relationships to collection...", flush=True)
        for k, v in tqdm(relationships.items(), desc="prep relationships"):
            if "ddx" in k.lower():
                k = k.replace("ddx", "differential diagnosis")
            entries.append(string_preprocess(k))

            origins = [i for i in v["origins"] if type(i) is str]
            targets = [i for i in v["targets"] if type(i) is str]

            try:
                origins = " ".join(origins) if len(origins) > 0 else " "
            except Exception:
                raise ValueError(f"Failed to join origins: {origins}")

            try:
                targets = " ".join(targets) if len(targets) > 0 else " "
            except Exception:
                raise ValueError(f"Failed to join targets: {targets}")

            uids.append({"origins": origins, "targets": targets})

        hashes = [hash_string(entry) for entry in entries]

        t0 = time.time()
        for i in tqdm(range(0, len(entries), self.batch_size), desc="embed relationships"):
            batch = entries[i : i + self.batch_size]
            batch_uids = uids[i : i + self.batch_size]
            batch_hashes = hashes[i : i + self.batch_size]

            try:
                relationship_collection.add(
                    documents=batch,
                    ids=batch_hashes,
                    metadatas=batch_uids,
                )
            except Exception as e:
                print(f"Failed to add relationship batch: {e}", flush=True)
                raise e

        elapsed = time.time() - t0
        print(f"[relationships] {len(entries)} entries indexed in {elapsed:.1f}s", flush=True)
        return relationship_collection

    def match_relationship(self, candidate_relationship: str, k_r: int = 1) -> str:
        """
        Match a relationship.
        """
        query = string_preprocess(candidate_relationship)
        count = self.relationship_collection.count()
        if count == 0:
            raise RuntimeError(
                "[VectorDB] relationship_collection is empty. "
                "Verify that the JSONL was built correctly and is non-empty."
            )
        k_r = max(1, min(k_r, count))

        try:
            results = self.relationship_collection.query(
                query_texts=[query],
                n_results=k_r,
                include=["documents", "metadatas", "distances"],
            )
        except RuntimeError:
            results = self.relationship_collection.query(
                query_texts=[query],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )

        return results

    # ChromaDB $in filter breaks above this many UIDs; skip filter instead
    _MAX_IN_FILTER = 10000

    def match_concept(
        self,
        concept: str,
        k_c: int = 1,
        uid_str: str = "",
    ) -> str:
        """
        Match a concept.
        """
        query = string_preprocess(concept)
        count = self.concept_collection.count()
        if count == 0:
            raise RuntimeError(
                "[VectorDB] concept_collection is empty. "
                "Verify that the JSONL was built correctly and is non-empty."
            )
        k_c = max(1, min(k_c, count))

        _empty = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        use_filter = False
        uid_list = []
        if uid_str:
            uid_list = uid_str.split(" ")
            if len(uid_list) <= self._MAX_IN_FILTER:
                use_filter = True

        try:
            if use_filter:
                results = self.concept_collection.query(
                    query_texts=[query],
                    n_results=k_c,
                    include=["documents", "distances", "metadatas"],
                    where={"uids": {"$in": uid_list}},
                )
            else:
                results = self.concept_collection.query(
                    query_texts=[query],
                    n_results=k_c,
                    include=["documents", "distances", "metadatas"],
                )
        except RuntimeError:
            try:
                results = self.concept_collection.query(
                    query_texts=[query],
                    n_results=1,
                    include=["documents", "distances", "metadatas"],
                )
            except Exception:
                print(f"[VectorDB] query fallback failed for: {query}", flush=True)
                return _empty
        except Exception:
            print(f"[VectorDB] query failed for concept: {query}", flush=True)
            return _empty

        return results


# Knowledge Graph class
class KnowledgeGraph:
    """
    Class for biomedical knowledge graph.
    """

    def __init__(self, graph: list[dict]):
        (
            self.edges,
            self.nodes,
            self.relationships,
            self.concepts,
        ) = self.initialize_edges(graph)

    def initialize_edges(self, graph: list[dict]):
        """
        Initialize edges from a graph.
        """
        edge_processed = {}
        nodes = set()
        relationships = {}
        concepts = {}
        num_edges = 0
        t0 = time.time()

        # O(1) membership tracking per relationship
        _rel_origin_seen: dict[str, set] = {}
        _rel_target_seen: dict[str, set] = {}
        _concept_ids_seen: dict[str, set] = {}

        for edge in graph:
            origin = edge["origin"]
            target = edge["target"]
            relationship = edge["relationship"]

            origin_id = hash_string(origin)
            target_id = hash_string(target)

            if relationship not in edge_processed:
                edge_processed[relationship] = {}
            if origin not in edge_processed[relationship]:
                edge_processed[relationship][origin] = {}
            if target not in edge_processed[relationship][origin]:
                edge_processed[relationship][origin][target] = True

            nodes.add(origin)
            nodes.add(target)

            if relationship not in relationships:
                relationships[relationship] = {"origins": [], "targets": []}
                _rel_origin_seen[relationship] = set()
                _rel_target_seen[relationship] = set()
            if origin_id not in _rel_origin_seen[relationship]:
                _rel_origin_seen[relationship].add(origin_id)
                relationships[relationship]["origins"].append(origin_id)
            if target_id not in _rel_target_seen[relationship]:
                _rel_target_seen[relationship].add(target_id)
                relationships[relationship]["targets"].append(target_id)

            if origin not in concepts:
                concepts[origin] = [origin_id]
                _concept_ids_seen[origin] = {origin_id}
            elif origin_id not in _concept_ids_seen[origin]:
                _concept_ids_seen[origin].add(origin_id)
                concepts[origin].append(origin_id)

            if target not in concepts:
                concepts[target] = [target_id]
                _concept_ids_seen[target] = {target_id}
            elif target_id not in _concept_ids_seen[target]:
                _concept_ids_seen[target].add(target_id)
                concepts[target].append(target_id)

            num_edges += 1
            if num_edges % 100000 == 0:
                elapsed = time.time() - t0
                print(
                    f"[KG] {num_edges} edges loaded... "
                    f"({elapsed:.0f}s, {len(concepts)} concepts so far)",
                    flush=True,
                )

        del _rel_origin_seen, _rel_target_seen, _concept_ids_seen

        nodes = list(nodes)

        elapsed = time.time() - t0
        print(
            f"Total edges: {num_edges} | unique concepts: {len(concepts)} | "
            f"unique relationships: {len(relationships)} | nodes: {len(nodes)} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

        if num_edges == 0:
            print(
                "[KG] WARNING: 0 edges loaded. Check that the JSONL file is non-empty.",
                flush=True,
            )

        return edge_processed, nodes, relationships, concepts

    def node_exact_match(self, concept: str) -> bool:
        """
        Check if a node exists in the knowledge graph.
        """
        # Check if node exists
        concept = string_preprocess(concept).lower()
        node_exists_result = concept in self.nodes

        # Return result
        return node_exists_result, concept

    def path_exists(self, origin: str, target: str, relationship: str) -> bool:
        """
        Check if a path exists from origin to target via relationship in the knowledge graph.
        """
        # Check if edge exists
        path_exists_result = (
            relationship in self.edges
            and origin in self.edges[relationship]
            and target in self.edges[relationship][origin]
        )

        # Return result
        return path_exists_result

    def query(self, origins: list, targets: list, relationship: str):
        """
        Query the knowledge graph for the existence of a path from origin to target via relationship.
        """
        # Iterate over origins
        for origin in origins:
            # Iterate over targets
            for target in targets:
                # Check if path exists
                if self.path_exists(origin, target, relationship):
                    # Append to valid paths
                    return True, " ".join([origin, relationship, target])

        # Return True if valid paths exist
        return False, ""
