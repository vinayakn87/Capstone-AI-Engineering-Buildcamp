from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    embedding: list[float]
    fund_name: str
    source_file: str
    page: int
    metadata: dict = field(default_factory=dict)


@dataclass
class FundMetadata:
    fund_name: str
    fund_house: str
    category: str
    benchmark: str | None = None
    nav: float | None = None
    aum_cr: float | None = None
    expense_ratio: float | None = None
    inception_date: str | None = None
    extra: dict = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]: ...

    @abstractmethod
    def has_fund(self, fund_name: str) -> bool: ...

    @abstractmethod
    def count(self) -> int: ...


class StructuredStore(ABC):
    @abstractmethod
    def upsert_fund(self, metadata: FundMetadata) -> None: ...

    @abstractmethod
    def query(self, filters: dict) -> list[dict]: ...

    @abstractmethod
    def has_fund(self, fund_name: str) -> bool: ...

    @abstractmethod
    def list_funds(self) -> list[str]: ...
