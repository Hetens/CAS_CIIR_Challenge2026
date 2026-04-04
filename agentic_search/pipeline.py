from __future__ import annotations

import logging
import time
from urllib.error import HTTPError, URLError

from agentic_search.config import Settings
from agentic_search.llm import Extractor, build_extractor
from agentic_search.logging_utils import append_jsonl_log
from agentic_search.models import DebugInfo, EntityRecord, RunMetrics, RuntimeInfo, SearchResponse, WebDocument
from agentic_search.scrape import fetch_document
from agentic_search.search import SearchProvider, build_search_provider

LOGGER = logging.getLogger("agentic_search.pipeline")


class AgenticSearchPipeline:
    def __init__(
        self,
        settings: Settings,
        search_provider: SearchProvider | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self.settings = settings
        self.search_provider = search_provider or build_search_provider(settings)
        self.extractor = extractor or build_extractor(settings)

    def run(
        self,
        query: str,
        *,
        debug: bool = False,
        log_path: str | None = None,
    ) -> SearchResponse:
        preprocessed_query = self._preprocess_query(query)
        LOGGER.info(
            "Starting query with llm_provider=%s llm_model=%s search_provider=%s requested_search_results=%s",
            self.settings.llm_provider,
            self._configured_llm_model_label(),
            self.settings.search_provider,
            self.settings.search_result_count,
        )
        if debug:
            LOGGER.debug("Step 0 query_preprocessing input=%r output=%r", query, preprocessed_query)
        total_start = time.perf_counter()
        search_start = time.perf_counter()
        results = self.search_provider.search(
            query=preprocessed_query,
            limit=self.settings.search_result_count,
        )
        search_latency_ms = (time.perf_counter() - search_start) * 1000
        LOGGER.info(
            "Search completed with actual_search_results=%s latency_ms=%.2f",
            len(results),
            search_latency_ms,
        )
        warnings: list[str] = []
        documents: list[WebDocument] = []
        searched_results = [
            {"title": result.title, "url": result.url, "snippet": result.snippet}
            for result in results
        ]
        if debug:
            LOGGER.debug("Step 1 web_search_results %s", searched_results)

        fetch_start = time.perf_counter()
        for result in results:
            try:
                documents.append(
                    fetch_document(
                        result=result,
                        timeout_seconds=self.settings.request_timeout_seconds,
                        max_chars=self.settings.max_page_chars,
                    )
                )
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                warnings.append(f"Failed to fetch {result.url}: {exc}")
        fetch_latency_ms = (time.perf_counter() - fetch_start) * 1000
        LOGGER.info(
            "Fetch completed with fetched_documents=%s fetch_failures=%s latency_ms=%.2f",
            len(documents),
            len(warnings),
            fetch_latency_ms,
        )

        if not documents:
            raise RuntimeError("No documents could be fetched from search results")

        extract_start = time.perf_counter()
        try:
            extracted = self.extractor.extract(query=preprocessed_query, documents=documents)
            llm_raw_output = getattr(self.extractor, "last_raw_output_text", "")
        except Exception as exc:
            warnings.append(f"LLM extraction failed, using fallback entity shaping: {exc}")
            extracted = self._fallback_entities(documents)
            llm_raw_output = ""
        extract_latency_ms = (time.perf_counter() - extract_start) * 1000
        llm_model = self._resolved_llm_model_label()
        LOGGER.info(
            "Extraction completed with llm_provider=%s llm_model=%s extracted_entities=%s latency_ms=%.2f",
            self.settings.llm_provider,
            llm_model,
            len(extracted),
            extract_latency_ms,
        )
        if debug:
            LOGGER.debug("Step 2 llm_structuring_output raw_output=%s", llm_raw_output or "<fallback_used>")
            LOGGER.debug(
                "Step 2 structured_entities %s",
                [item.to_dict() for item in extracted],
            )
        deduped = self._dedupe_entities(extracted)
        total_latency_ms = (time.perf_counter() - total_start) * 1000
        metrics = RunMetrics(
            search_latency_ms=round(search_latency_ms, 2),
            fetch_latency_ms=round(fetch_latency_ms, 2),
            extract_latency_ms=round(extract_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            search_results_count=len(results),
            fetched_documents_count=len(documents),
            extracted_entities_count=len(extracted),
            final_entities_count=len(deduped),
            fetch_failures_count=len(warnings),
            pages_per_second=round(
                len(documents) / max(fetch_latency_ms / 1000, 0.001),
                2,
            ),
            entities_per_second=round(
                len(deduped) / max(total_latency_ms / 1000, 0.001),
                2,
            ),
        )
        LOGGER.info(
            "Final summary llm_provider=%s llm_model=%s search_provider=%s actual_search_results=%s final_entities=%s total_latency_ms=%.2f pages_per_second=%.2f entities_per_second=%.2f",
            self.settings.llm_provider,
            llm_model,
            self.settings.search_provider,
            len(results),
            len(deduped),
            total_latency_ms,
            metrics.pages_per_second,
            metrics.entities_per_second,
        )
        if debug:
            LOGGER.debug("Step 3 final_summary %s", metrics)
        runtime = RuntimeInfo(
            llm_provider=self.settings.llm_provider,
            llm_model=llm_model,
            search_provider=self.settings.search_provider,
            requested_search_results=self.settings.search_result_count,
            actual_search_results=len(results),
        )
        debug_info = DebugInfo(
            query=query,
            preprocessed_query=preprocessed_query,
            search_provider=self.settings.search_provider,
            llm_provider=self.settings.llm_provider,
            llm_model=llm_model,
            steps=[
                {
                    "step": 0,
                    "name": "query_preprocessing",
                    "input_query": query,
                    "preprocessed_query": preprocessed_query,
                },
                {
                    "step": 1,
                    "name": "web_search_results",
                    "results": searched_results,
                    "result_count": len(searched_results),
                    "latency_ms": round(search_latency_ms, 2),
                },
                {
                    "step": 2,
                    "name": "llm_structuring_output",
                    "llm_provider": self.settings.llm_provider,
                    "llm_model": llm_model,
                    "raw_output_text": llm_raw_output,
                    "structured_entities": [item.to_dict() for item in extracted],
                    "latency_ms": round(extract_latency_ms, 2),
                    "used_fallback": llm_raw_output == "",
                },
                {
                    "step": 3,
                    "name": "final_summary",
                    "metrics": {
                        "llm_provider": self.settings.llm_provider,
                        "llm_model": llm_model,
                        "search_provider": self.settings.search_provider,
                        "search_latency_ms": round(search_latency_ms, 2),
                        "fetch_latency_ms": round(fetch_latency_ms, 2),
                        "extract_latency_ms": round(extract_latency_ms, 2),
                        "total_latency_ms": round(total_latency_ms, 2),
                        "pages_per_second": metrics.pages_per_second,
                        "entities_per_second": metrics.entities_per_second,
                        "search_results_count": len(results),
                        "fetched_documents_count": len(documents),
                        "final_entities_count": len(deduped),
                        "fetch_failures_count": len(warnings),
                    },
                },
            ],
            searched_results=searched_results,
            fetched_documents=[
                {
                    "title": doc.title,
                    "url": doc.url,
                    "snippet": doc.snippet,
                    "content_chars": len(doc.content),
                }
                for doc in documents
            ],
            warnings=warnings,
        )
        response = SearchResponse(
            query=query,
            results=deduped,
            searched_urls=[doc.url for doc in documents],
            warnings=warnings,
            runtime=runtime,
            metrics=metrics,
            debug=debug_info if debug else None,
        )
        log_file = append_jsonl_log(
            log_path,
            {
                "query": query,
                "preprocessed_query": preprocessed_query,
                "runtime": response.to_dict().get("runtime", {}),
                "search_provider": self.settings.search_provider,
                "llm_provider": self.settings.llm_provider,
                "metrics": response.to_dict().get("metrics", {}),
                "warnings": warnings,
                "steps": debug_info.steps,
                "searched_results": searched_results,
                "fetched_documents": debug_info.fetched_documents,
                "results": response.to_dict()["results"],
            },
        )
        response.log_file = log_file
        if response.debug is not None:
            response.debug.log_file = log_file
        return response

    @staticmethod
    def _preprocess_query(query: str) -> str:
        return " ".join(query.strip().split())

    def _resolved_llm_model_label(self) -> str:
        model = getattr(self.extractor, "last_used_model", "")
        if model:
            return model
        return self._configured_llm_model_label()

    def _configured_llm_model_label(self) -> str:
        provider = self.settings.llm_provider
        if provider == "openai":
            return self.settings.openai_model
        if provider == "gemini":
            return self.settings.gemini_model
        if provider == "ollama":
            return self.settings.ollama_model or "auto-discover"
        if provider == "auto":
            if self.settings.gemini_api_key:
                return self.settings.gemini_model
            if self.settings.openai_api_key:
                return self.settings.openai_model
            return self.settings.ollama_model or "auto-discover"
        return "unknown"

    @staticmethod
    def _fallback_entities(documents: list[WebDocument]) -> list[EntityRecord]:
        from agentic_search.models import CellValue

        entities: list[EntityRecord] = []
        for doc in documents[:5]:
            title = doc.title.strip() or doc.url
            description = (doc.snippet or doc.content[:240]).strip()
            entities.append(
                EntityRecord(
                    entity_name=CellValue(
                        value=title,
                        source_url=doc.url,
                        source_title=doc.title or doc.url,
                        evidence=title,
                    ),
                    entity_type=CellValue(
                        value="web_result",
                        source_url=doc.url,
                        source_title=doc.title or doc.url,
                        evidence="Derived from search result metadata",
                    ),
                    description=CellValue(
                        value=description,
                        source_url=doc.url,
                        source_title=doc.title or doc.url,
                        evidence=description[:160],
                    ),
                    attributes={},
                    confidence=0.25,
                    sources=[doc.url],
                )
            )
        return entities

    @staticmethod
    def _dedupe_entities(entities: list[EntityRecord]) -> list[EntityRecord]:
        merged: dict[str, EntityRecord] = {}
        for entity in entities:
            key = entity.entity_name.value.strip().lower()
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = entity
                continue
            if entity.confidence > existing.confidence:
                winner, loser = entity, existing
            else:
                winner, loser = existing, entity
            for attr_name, attr_value in loser.attributes.items():
                winner.attributes.setdefault(attr_name, attr_value)
            winner.sources = sorted(set(winner.sources + loser.sources))
            merged[key] = winner
        return sorted(
            merged.values(),
            key=lambda item: (-item.confidence, item.entity_name.value.lower()),
        )
