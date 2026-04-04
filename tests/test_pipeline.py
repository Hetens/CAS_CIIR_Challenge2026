import unittest
from pathlib import Path

from agentic_search.config import Settings
from agentic_search.llm import ExtractionPayload, OllamaExtractor, build_extractor
from agentic_search.models import CellValue, EntityRecord, SearchResponse, SearchResult
from agentic_search.pipeline import AgenticSearchPipeline
from agentic_search.search import _DuckDuckGoHTMLParser


class FakeSearchProvider:
    def search(self, query: str, limit: int) -> list[SearchResult]:
        return [
            SearchResult(title="One", url="https://example.com/1", snippet="first"),
            SearchResult(title="Two", url="https://example.com/2", snippet="second"),
        ]


class FakeExtractor:
    def extract(self, query, documents):
        self.seen_query = query
        self.seen_documents = documents
        return [
            EntityRecord(
                entity_name=CellValue(
                    value="Acme Health",
                    source_url="https://example.com/1",
                    source_title="One",
                    evidence="Acme Health is a care AI company",
                ),
                entity_type=CellValue(
                    value="startup",
                    source_url="https://example.com/1",
                    source_title="One",
                    evidence="startup",
                ),
                description=CellValue(
                    value="Builds clinical workflow AI tools",
                    source_url="https://example.com/1",
                    source_title="One",
                    evidence="clinical workflow AI tools",
                ),
                attributes={
                    "location": CellValue(
                        value="New York",
                        source_url="https://example.com/1",
                        source_title="One",
                        evidence="based in New York",
                    )
                },
                confidence=0.8,
                sources=["https://example.com/1"],
            ),
            EntityRecord(
                entity_name=CellValue(
                    value="Acme Health",
                    source_url="https://example.com/2",
                    source_title="Two",
                    evidence="Acme Health",
                ),
                entity_type=CellValue(
                    value="company",
                    source_url="https://example.com/2",
                    source_title="Two",
                    evidence="company",
                ),
                description=CellValue(
                    value="Healthcare AI startup",
                    source_url="https://example.com/2",
                    source_title="Two",
                    evidence="healthcare AI startup",
                ),
                attributes={
                    "funding": CellValue(
                        value="$10M",
                        source_url="https://example.com/2",
                        source_title="Two",
                        evidence="$10M seed round",
                    )
                },
                confidence=0.9,
                sources=["https://example.com/2"],
            ),
        ]


class FailingExtractor:
    def extract(self, query, documents):
        raise TimeoutError("local llm timed out")


class PipelineTests(unittest.TestCase):
    def test_pipeline_dedupes_and_merges_sources(self):
        settings = Settings(
            llm_provider="auto",
            openai_api_key="key",
            openai_model="model",
            gemini_api_key="",
            gemini_model="gemini-test",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="gemma-test",
            search_provider="duckduckgo",
            brave_api_key="key",
            serpapi_api_key="",
            search_result_count=5,
            max_page_chars=1000,
            request_timeout_seconds=5,
        )
        extractor = FakeExtractor()
        pipeline = AgenticSearchPipeline(
            settings=settings,
            search_provider=FakeSearchProvider(),
            extractor=extractor,
        )

        import agentic_search.pipeline as pipeline_module

        original_fetch = pipeline_module.fetch_document
        pipeline_module.fetch_document = lambda result, timeout_seconds, max_chars: type(
            "Doc",
            (),
            {
                "title": result.title,
                "url": result.url,
                "content": f"Content for {result.url}",
                "snippet": result.snippet,
            },
        )()
        log_path = Path("tests") / "artifacts" / "search_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()
        try:
            response = pipeline.run(
                "AI startups in healthcare",
                debug=True,
                log_path=str(log_path),
            )
        finally:
            pipeline_module.fetch_document = original_fetch

        self.assertIsInstance(response, SearchResponse)
        self.assertEqual(response.query, "AI startups in healthcare")
        self.assertEqual(len(response.results), 1)
        entity = response.results[0]
        self.assertEqual(entity.entity_name.value, "Acme Health")
        self.assertIn("location", entity.attributes)
        self.assertIn("funding", entity.attributes)
        self.assertEqual(
            entity.sources,
            ["https://example.com/1", "https://example.com/2"],
        )
        self.assertTrue(response.log_file.endswith("search_log.jsonl"))
        self.assertTrue(log_path.exists())
        self.assertIsNotNone(response.metrics)
        self.assertIsNotNone(response.debug)
        self.assertEqual(response.debug.preprocessed_query, "AI startups in healthcare")
        self.assertEqual([step["step"] for step in response.debug.steps], [0, 1, 2, 3])
        self.assertEqual(response.debug.steps[0]["name"], "query_preprocessing")
        self.assertEqual(response.debug.steps[1]["name"], "web_search_results")
        self.assertEqual(response.debug.steps[2]["name"], "llm_structuring_output")
        self.assertEqual(response.debug.steps[3]["name"], "final_summary")

    def test_build_extractor_prefers_gemini_in_auto_mode(self):
        settings = Settings(
            llm_provider="auto",
            openai_api_key="",
            openai_model="gpt-test",
            gemini_api_key="gemini-key",
            gemini_model="gemini-test",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="gemma-test",
            search_provider="duckduckgo",
            brave_api_key="",
            serpapi_api_key="",
            search_result_count=5,
            max_page_chars=1000,
            request_timeout_seconds=5,
        )
        extractor = build_extractor(settings)
        self.assertEqual(extractor.__class__.__name__, "GeminiExtractor")

    def test_pipeline_falls_back_when_extractor_fails(self):
        settings = Settings(
            llm_provider="auto",
            openai_api_key="",
            openai_model="model",
            gemini_api_key="",
            gemini_model="gemini-test",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="gemma4:e2b",
            search_provider="duckduckgo",
            brave_api_key="",
            serpapi_api_key="",
            search_result_count=5,
            max_page_chars=1000,
            request_timeout_seconds=5,
        )
        pipeline = AgenticSearchPipeline(
            settings=settings,
            search_provider=FakeSearchProvider(),
            extractor=FailingExtractor(),
        )

        import agentic_search.pipeline as pipeline_module

        original_fetch = pipeline_module.fetch_document
        pipeline_module.fetch_document = lambda result, timeout_seconds, max_chars: type(
            "Doc",
            (),
            {
                "title": result.title,
                "url": result.url,
                "content": f"Content for {result.url}",
                "snippet": result.snippet,
            },
        )()
        try:
            response = pipeline.run("AI startups in healthcare")
        finally:
            pipeline_module.fetch_document = original_fetch

        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.results[0].entity_type.value, "web_result")
        self.assertTrue(any("LLM extraction failed" in warning for warning in response.warnings))
        self.assertTrue(response.debug is None)

    def test_pydantic_payload_parses_entity_schema(self):
        payload = ExtractionPayload.model_validate(
            {
                "entities": [
                    {
                        "entity_name": {
                            "value": "Acme Health",
                            "source_url": "https://example.com/1",
                            "source_title": "One",
                            "evidence": "Acme Health",
                        },
                        "entity_type": {
                            "value": "startup",
                            "source_url": "https://example.com/1",
                            "source_title": "One",
                            "evidence": "startup",
                        },
                        "description": {
                            "value": "Healthcare AI startup",
                            "source_url": "https://example.com/1",
                            "source_title": "One",
                            "evidence": "Healthcare AI startup",
                        },
                        "attributes": {
                            "location": {
                                "value": "New York",
                                "source_url": "https://example.com/1",
                                "source_title": "One",
                                "evidence": "based in New York",
                            }
                        },
                        "confidence": 1.4,
                        "sources": ["https://example.com/1"],
                    }
                ]
            }
        )
        runtime = payload.entities[0].to_runtime()
        self.assertEqual(runtime.entity_name.value, "Acme Health")
        self.assertEqual(runtime.attributes["location"].value, "New York")
        self.assertEqual(runtime.confidence, 1.0)

    def test_duckduckgo_parser_extracts_results(self):
        parser = _DuckDuckGoHTMLParser(limit=5)
        parser.feed(
            """
            <html><body>
              <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Acme Health</a>
              <div class="result__snippet">Clinical AI for providers</div>
              <a class="result__a" href="https://example.com/b">Beta Care</a>
              <div class="result__snippet">Care coordination platform</div>
            </body></html>
            """
        )
        parser.close()
        self.assertEqual(len(parser.results), 2)
        self.assertEqual(parser.results[0].title, "Acme Health")
        self.assertEqual(parser.results[0].url, "https://example.com/a")
        self.assertEqual(parser.results[1].snippet, "Care coordination platform")

    def test_ollama_extractor_discovers_installed_model(self):
        extractor = OllamaExtractor(
            base_url="http://127.0.0.1:11434",
            model="",
            timeout_seconds=5,
        )
        extractor._get_json = lambda url: {
            "models": [{"name": "gemma4:e2b"}]
        }
        self.assertEqual(extractor._discover_model(), "gemma4:e2b")


if __name__ == "__main__":
    unittest.main()
