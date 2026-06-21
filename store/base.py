# =============================================================================
# store/base.py — Abstract interfaces for the store layer
#
# WHY THIS FILE EXISTS:
# The agent and ingestion pipeline need to read/write data, but they should not
# care whether the backend is ChromaDB, Qdrant, SQLite, or anything else.
# By coding to these abstract interfaces, we can swap backends freely without
# touching the agent or ingestion code.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# We need three things from the standard library:
#   - `dataclass` decorator  → to define clean data-holding classes (Chunk, FundMetadata)
#   - `field`                → to set mutable defaults (e.g. empty dict) safely in dataclasses
#   - `ABC` and `abstractmethod` from `abc` module → to define abstract base classes
#
# Write your imports here:
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# -----------------------------------------------------------------------------
# DATA CLASS: Chunk - data about the chunk
#
# WHY: Every piece of text extracted from a factsheet PDF becomes a "Chunk".
# This is the unit that gets embedded and stored in the vector store.
# The agent retrieves a list of Chunks when searching for relevant context.
#
# FIELDS TO DEFINE:
#   chunk_id    : str          — unique identifier (e.g. "mirae_asset_p3_chunk2")
#   text        : str          — the raw text content of this chunk
#   embedding   : list[float]  — the vector representation (384 floats for all-MiniLM-L6-v2)
#   fund_name   : str          — which fund this chunk came from (used for filtering)
#   source_file : str          — original PDF filename
#   page        : int          — page number in the PDF
#   metadata    : dict         — any extra key-value pairs (use `field(default_factory=dict)`)
#
# Write the Chunk dataclass here:

@dataclass
class Chunk:
    chunk_id: str
    text: str
    embedding: list[float]
    fund_name: str
    source_file: str
    page: int
    metadata: dict = field(default_factory=dict)



# -----------------------------------------------------------------------------
# DATA CLASS: FundMetadata - data about the fund
#
# WHY: Structured, queryable facts about a fund (expense ratio, NAV, AUM etc.)
# are stored separately in SQLite — not in the vector store — because they need
# exact/range queries ("show funds with expense ratio < 1%"), not semantic search.
#
# FIELDS TO DEFINE:
#   fund_name      : str         — canonical fund name (primary key)
#   fund_house     : str         — AMC name (e.g. "Mirae Asset")
#   category       : str         — fund category (e.g. "Large Cap", "ELSS")
#   benchmark      : str | None  — benchmark index, optional
#   nav            : float | None — latest NAV
#   aum_cr         : float | None — AUM in crores
#   expense_ratio  : float | None — total expense ratio (%)
#   inception_date : str | None  — fund launch date
#   extra          : dict        — any additional fields (use `field(default_factory=dict)`)
#
# Write the FundMetadata dataclass here:

@dataclass
class FundMetadata:
    fund_name: str
    fund_house: str
    fund_manager: str
    category: str
    risk_level: str
    benchmark: str | None = None
    nav: float | None = None
    aum_cr: float | None = None
    expense_ratio: float | None = None
    inception_date: str | None = None
    extra: dict = field(default_factory=dict)



# -----------------------------------------------------------------------------
# ABSTRACT BASE CLASS: VectorStore
#
# WHY: Defines the contract for any vector store backend (ChromaDB, Qdrant, etc.)
# ChromaDB is the default implementation; Qdrant can be swapped in by implementing
# this same interface. The agent and ingestion code only ever call these methods.
#
# METHODS TO DEFINE (all abstract):
#
#   add(chunks: list[Chunk]) -> None
#       Stores a list of Chunk objects (text + embedding + metadata) in the vector store.
#       Called by the ingestion pipeline after embedding a batch of chunks.
#
#   search(query_embedding: list[float], top_k: int = 5) -> list[Chunk]
#       Given a query vector, returns the top_k most similar Chunks.
#       Called by the retriever when a user asks a question.
#
#   has_fund(fund_name: str) -> bool
#       Returns True if any chunk for this fund exists in the store.
#       Called by the retriever to detect missing funds (FundNotFoundError).
#
#   count() -> int
#       Returns total number of chunks stored.
#       Useful for debugging and tests.
#
# Write the VectorStore ABC here:

class VectorStore(ABC):

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        pass

    @abstractmethod
    def has_fund(self, fund_name: str) -> bool:
        pass

    @abstractmethod
    def count(self) -> int:
        pass





# -----------------------------------------------------------------------------
# ABSTRACT BASE CLASS: StructuredStore
#
# WHY: Defines the contract for the structured (SQL) store backend.
# Holds fund-level metadata that needs exact queries, not semantic search.
#
# METHODS TO DEFINE (all abstract):
#
#   upsert_fund(metadata: FundMetadata) -> None
#       Inserts or updates a fund's structured metadata.
#       Called during ingestion when fund-level fields are extracted from the PDF.
#
#   query(filters: dict) -> list[dict]
#       Runs a filtered query and returns matching fund records as dicts.
#       Example: query({"expense_ratio__lt": 1.0}) → funds cheaper than 1%
#       Called by the retriever for structured questions.
#
#   has_fund(fund_name: str) -> bool
#       Returns True if this fund exists in the structured store.
#       Used alongside VectorStore.has_fund() for completeness checks.
#
#   list_funds() -> list[str]
#       Returns all fund names currently in the store.
#       Useful for the UI to show available funds.
#
# Write the StructuredStore ABC here:

class StructuredStore(ABC):
    @abstractmethod
    def upsert_fund(self, metadata: FundMetadata) -> None:
        pass

    @abstractmethod
    def query(self, filters: dict) -> list[dict]:
        pass

    @abstractmethod
    def has_fund(self, fund_name: str) -> bool:
        pass

    def list_funds(self) -> list[str]:
        return []