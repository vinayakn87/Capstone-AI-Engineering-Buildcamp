# =============================================================================
# store/vector_store.py — ChromaDB implementation of VectorStore
#
# WHY THIS FILE EXISTS:
# base.py defined the abstract contract. This file is the concrete ChromaDB
# implementation. The ingestion pipeline writes here; the retriever reads here.
# Nothing outside this file should import chromadb directly — swap to Qdrant
# by replacing this file with a new implementation of the same interface.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `chromadb`                        → the vector store client
#   - `chromadb.config.Settings`        → to disable anonymous telemetry
#   - `VectorStore` and `Chunk`         → from store.base (the interface we implement)
#
# Write your imports here:
import chromadb
from chromadb.config import Settings
from store.base import VectorStore, Chunk

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
# Define the ChromaDB collection name as a module-level constant.
# A collection is like a table — all factsheet chunks go into one collection.
#
#   _COLLECTION_NAME = "factsheet_chunks"
#
# Write it here:
_COLLECTION_NAME = "factsheet_chunks"

# -----------------------------------------------------------------------------
# CLASS: ChromaVectorStore(VectorStore)
#
# WHY: Wraps ChromaDB behind the VectorStore interface so the rest of the
# codebase never has to know it's ChromaDB underneath.
# =============================================================================

class ChromaVectorStore(VectorStore):
# Write the class definition here, inheriting from VectorStore:

    # -------------------------------------------------------------------------
    # __init__(self, persist_dir: str = "data/chroma")
    #
    # WHY: Sets up the ChromaDB client and collection once at construction time.
    # All subsequent method calls reuse the same client and collection.
    #
    # STEPS:
    #   1. Create a PersistentClient pointing at `persist_dir`
    #      Use Settings(anonymized_telemetry=False) to suppress telemetry noise
    #   2. Call get_or_create_collection() with _COLLECTION_NAME
    #      Pass metadata={"hnsw:space": "cosine"} so similarity is measured
    #      by cosine distance (appropriate for sentence-transformer embeddings)
    #   3. Store both client and collection as instance variables (self._client, self._col)
    #
    # Write __init__ here:

    def __init__(self, persist_dir: str = "data/chroma"):
        # Step 1: Create a PersistentClient with telemetry disabled
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        # Step 2: Get or create the collection with cosine similarity
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    # -------------------------------------------------------------------------
    # add(self, chunks: list[Chunk]) -> None
    #
    # WHY: Persists a batch of embedded chunks into ChromaDB.
    # Called by the ingestion pipeline after embedding a page of text.
    #
    # STEPS:
    #   1. Guard: if chunks is empty, return immediately (nothing to do)
    #   2. Call self._col.upsert() with four parallel lists:
    #        ids        → [c.chunk_id for c in chunks]
    #        embeddings → [c.embedding for c in chunks]
    #        documents  → [c.text for c in chunks]
    #        metadatas  → one dict per chunk containing:
    #                       fund_name, source_file, page, + anything in c.metadata
    #      Use upsert (not add) so re-ingesting a PDF updates existing chunks
    #      rather than raising a duplicate error.
    #
    # Write add() here:

    def add(self, chunks: list[Chunk]):
        
        if not(chunks):
            return
        
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas= [{**c.metadata, "fund_name":c.fund_name, "source_file":c.source_file, "page":c.page} for c in chunks]
        )


    # -------------------------------------------------------------------------
    # search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]
    #
    # WHY: Given an already-computed query vector, finds the top_k most similar
    # chunks in the collection. Called by the retriever for every user question.
    #
    # STEPS:
    #   1. Guard: if the collection is empty (self._col.count() == 0), return []
    #   2. Call self._col.query():
    #        query_embeddings = [query_embedding]   ← note: list of lists
    #        n_results        = min(top_k, self._col.count())  ← avoid requesting
    #                                                            more than exists
    #        include          = ["documents", "metadatas", "embeddings"]
    #   3. ChromaDB returns results as parallel lists nested inside outer lists
    #      (one inner list per query). Since we only sent one query, index [0]:
    #        results["ids"][0], results["documents"][0], etc.
    #   4. Reconstruct a Chunk object for each result:
    #        - chunk_id    ← from results["ids"][0][i]
    #        - text        ← from results["documents"][0][i]
    #        - embedding   ← from results["embeddings"][0][i]
    #        - fund_name   ← pop "fund_name" from metadatas[i]
    #        - source_file ← pop "source_file" from metadatas[i]
    #        - page        ← pop "page" from metadatas[i]
    #        - metadata    ← whatever remains in metadatas[i] after popping above
    #   5. Return the list of Chunk objects
    #
    # Write search() here:

    def search(self, query_embedding: list[float], top_k = 5) -> list[Chunk]:

        if self._col.count() == 0:
            return []
        
        results = self._col.query(
            query_embeddings = [query_embedding],
            n_results = min(self._col.count(), top_k),
            include = ["documents", "embeddings", "metadatas"]
        )

        chunk_id = results["ids"][0]
        documents = results["documents"][0]
        embeddings = results["embeddings"][0]
        metadatas = results["metadatas"][0]

        result_chunks = []
        for i in range(len(chunk_id)):

            meta = metadatas[i]
            fund_name = meta.pop("fund_name")
            source_file = meta.pop("source_file")
            page = meta.pop("page")
            metadata = {**meta}


            chunk = Chunk(
                chunk_id = chunk_id[i],
                text = documents[i],
                embedding = embeddings[i],
                fund_name = fund_name,
                source_file = source_file,
                page = page,
                metadata = metadata
            )

            result_chunks.append(chunk)
        
        return result_chunks



    # -------------------------------------------------------------------------
    # has_fund(self, fund_name: str) -> bool
    #
    # WHY: Lets the retriever check whether any data exists for a given fund
    # before deciding to raise FundNotFoundError.
    #
    # STEPS:
    #   1. Call self._col.get() with:
    #        where = {"fund_name": fund_name}   ← metadata filter
    #        limit = 1                           ← we only need to know if ≥1 exists
    #   2. Return True if results["ids"] is non-empty, False otherwise
    #
    # Write has_fund() here:

    def has_fund(self, fund_name: str) -> bool:
        # Exact metadata lookup (no embedding) — just check existence
        results = self._col.get(
            where={"fund_name": fund_name},   # metadata filter
            limit=1                           # we only need to know if ≥1 exists
        )
        # get() returns a flat list; non-empty means at least one match
        return len(results["ids"]) > 0


    # -------------------------------------------------------------------------
    # count(self) -> int
    #
    # WHY: Returns total number of chunks stored. Used in tests and for debugging.
    #
    # STEPS:
    #   1. Return self._col.count()
    #      (ChromaDB tracks this internally — no need to query)
    #
    # Write count() here:

    def count(self) -> int:
        # ChromaDB tracks the total internally — no query needed
        return self._col.count()
