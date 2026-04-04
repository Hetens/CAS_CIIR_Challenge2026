from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class WebDocument:
    title: str
    url: str
    content: str
    snippet: str = ""


@dataclass
class CellValue:
    value: str
    source_url: str
    source_title: str
    evidence: str


@dataclass
class EntityRecord:
    entity_name: CellValue
    entity_type: CellValue
    description: CellValue
    attributes: dict[str, CellValue] = field(default_factory=dict)
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        base = asdict(self)
        base["attributes"] = {
            key: asdict(value) for key, value in self.attributes.items()
        }
        return base


@dataclass
class RunMetrics:
    search_latency_ms: float = 0.0
    fetch_latency_ms: float = 0.0
    extract_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    search_results_count: int = 0
    fetched_documents_count: int = 0
    extracted_entities_count: int = 0
    final_entities_count: int = 0
    fetch_failures_count: int = 0
    pages_per_second: float = 0.0
    entities_per_second: float = 0.0
    llm_prompt_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_tokens_per_second: float | None = None


@dataclass
class RuntimeInfo:
    llm_provider: str
    llm_model: str
    search_provider: str
    requested_search_results: int
    actual_search_results: int


@dataclass
class DebugInfo:
    query: str
    preprocessed_query: str
    search_provider: str
    llm_provider: str
    llm_model: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    searched_results: list[dict[str, str]] = field(default_factory=list)
    fetched_documents: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    log_file: str = ""


@dataclass
class SearchResponse:
    query: str
    results: list[EntityRecord]
    searched_urls: list[str]
    warnings: list[str] = field(default_factory=list)
    log_file: str = ""
    runtime: RuntimeInfo | None = None
    metrics: RunMetrics | None = None
    debug: DebugInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "query": self.query,
            "results": [item.to_dict() for item in self.results],
            "searched_urls": self.searched_urls,
            "warnings": self.warnings,
            "log_file": self.log_file,
        }
        if self.runtime is not None:
            payload["runtime"] = asdict(self.runtime)
        if self.metrics is not None:
            payload["metrics"] = asdict(self.metrics)
        if self.debug is not None:
            payload["debug"] = asdict(self.debug)
        return payload
