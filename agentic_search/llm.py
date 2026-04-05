from __future__ import annotations

import json
import socket
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import quote_plus
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from agentic_search.config import Settings
from agentic_search.models import CellValue, EntityRecord, WebDocument


SYSTEM_PROMPT = """You extract entities from web documents for a topic search system.

Return strict JSON with this shape:
{
  "entities": [
    {
      "entity_name": {"value": "...", "source_url": "...", "source_title": "...", "evidence": "..."},
      "entity_type": {"value": "...", "source_url": "...", "source_title": "...", "evidence": "..."},
      "description": {"value": "...", "source_url": "...", "source_title": "...", "evidence": "..."},
      "attributes": {
        "attribute_name": {"value": "...", "source_url": "...", "source_title": "...", "evidence": "..."}
      },
      "confidence": 0.0,
      "sources": ["..."]
    }
  ]
}

Rules:
- Extract only entities relevant to the topic query.
- Every field must cite a source_url and source_title from the provided documents.
- Evidence must be a short supporting quote or close paraphrase from the source text.
- Keep attribute names concise and useful.
- Prefer facts directly supported by the documents.
- Confidence must be between 0 and 1.
- Return JSON only.
"""


class Extractor(Protocol):
    def extract(self, query: str, documents: list[WebDocument]) -> list[EntityRecord]:
        ...


class CellValueModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: str = ""
    source_url: str = ""
    source_title: str = ""
    evidence: str = ""

    def to_runtime(self) -> CellValue:
        return CellValue(
            value=self.value,
            source_url=self.source_url,
            source_title=self.source_title,
            evidence=self.evidence,
        )


class EntityRecordModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entity_name: CellValueModel
    entity_type: CellValueModel
    description: CellValueModel
    attributes: dict[str, CellValueModel] = Field(default_factory=dict)
    confidence: float = 0.0
    sources: list[str] = Field(default_factory=list)

    def to_runtime(self) -> EntityRecord:
        confidence = max(0.0, min(1.0, self.confidence))
        return EntityRecord(
            entity_name=self.entity_name.to_runtime(),
            entity_type=self.entity_type.to_runtime(),
            description=self.description.to_runtime(),
            attributes={
                key: value.to_runtime() for key, value in self.attributes.items()
            },
            confidence=confidence,
            sources=self.sources,
        )


class ExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[EntityRecordModel] = Field(default_factory=list)


class BaseExtractor(ABC):
    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.last_raw_output_text = ""
        self.last_used_model = ""
        self.last_usage: dict[str, Any] = {}
        self.on_llm_chunk: Callable[[str], None] | None = None

    def extract(self, query: str, documents: list[WebDocument]) -> list[EntityRecord]:
        self.last_usage = {}
        raw_text = self._generate_json_text(query=query, documents=documents)
        self.last_raw_output_text = raw_text
        parsed = ExtractionPayload.model_validate_json(raw_text)
        return [item.to_runtime() for item in parsed.entities]

    @abstractmethod
    def _generate_json_text(self, query: str, documents: list[WebDocument]) -> str:
        raise NotImplementedError

    def _build_prompt(self, query: str, documents: list[WebDocument]) -> str:
        document_blocks = []
        for index, doc in enumerate(documents, start=1):
            document_blocks.append(
                "\n".join(
                    [
                        f"Document {index}",
                        f"URL: {doc.url}",
                        f"Title: {doc.title}",
                        f"Snippet: {doc.snippet}",
                        f"Content: {doc.content}",
                    ]
                )
            )
        return "\n\n".join([SYSTEM_PROMPT, f"Topic query: {query}", *document_blocks])

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            detail = body.strip() or exc.reason
            raise ValueError(f"HTTP {exc.code} calling {url}: {detail}") from exc
        except socket.timeout as exc:
            raise TimeoutError(
                f"Timed out after {self.timeout_seconds}s calling {url}"
            ) from exc

    def _get_json(self, url: str) -> dict:
        request = Request(url=url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            detail = body.strip() or exc.reason
            raise ValueError(f"HTTP {exc.code} calling {url}: {detail}") from exc


class OpenAIExtractor(BaseExtractor):
    def __init__(self, api_key: str, model: str, timeout_seconds: int) -> None:
        super().__init__(timeout_seconds)
        self.api_key = api_key
        self.model = model

    def _generate_json_text(self, query: str, documents: list[WebDocument]) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for extraction")
        self.last_used_model = self.model
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._build_prompt(query=query, documents=documents),
                        }
                    ],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        }
        raw = self._post_json(
            url="https://api.openai.com/v1/responses",
            payload=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        usage = raw.get("usage", {})
        self.last_usage = {
            "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        return self._extract_openai_output_text(raw)

    @staticmethod
    def _extract_openai_output_text(payload: dict) -> str:
        if "output_text" in payload:
            return payload["output_text"]
        texts: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("text"):
                    texts.append(content["text"])
        if texts:
            return "".join(texts)
        raise ValueError("OpenAI response did not include output text")


class GeminiExtractor(BaseExtractor):
    def __init__(self, api_key: str, model: str, timeout_seconds: int) -> None:
        super().__init__(timeout_seconds)
        self.api_key = api_key
        self.model = model

    def _generate_json_text(self, query: str, documents: list[WebDocument]) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for extraction")
        self.last_used_model = self.model
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": self._build_prompt(query=query, documents=documents),
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
            },
        }
        raw = self._post_json(
            url=(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{quote_plus(self.model)}:generateContent?key={quote_plus(self.api_key)}"
            ),
            payload=payload,
            headers={},
        )
        usage = raw.get("usageMetadata", {})
        self.last_usage = {
            "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "total_tokens": int(usage.get("totalTokenCount", 0) or 0),
        }
        candidates = raw.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    return text
        raise ValueError("Gemini response did not include text output")


class OllamaExtractor(BaseExtractor):
    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        super().__init__(timeout_seconds)
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _generate_json_text(self, query: str, documents: list[WebDocument]) -> str:
        model = self.model or self._discover_model()
        self.last_used_model = model
        prepared_documents = self._prepare_documents(documents)
        use_stream = self.on_llm_chunk is not None
        payload = {
            "model": model,
            "prompt": self._build_prompt(query=query, documents=prepared_documents),
            "stream": use_stream,
            "format": "json",
        }
        if use_stream:
            return self._stream_generate(payload)
        raw = self._post_json(
            url=f"{self.base_url}/api/generate",
            payload=payload,
            headers={},
        )
        self._extract_ollama_usage(raw)
        response_text = raw.get("response")
        if not response_text:
            raise ValueError("Ollama response did not include generated JSON")
        return response_text

    def _extract_ollama_usage(self, raw: dict) -> None:
        prompt_tokens = int(raw.get("prompt_eval_count", 0) or 0)
        output_tokens = int(raw.get("eval_count", 0) or 0)
        total_tokens = prompt_tokens + output_tokens
        eval_duration_ns = raw.get("eval_duration", 0) or 0
        tokens_per_second = None
        if eval_duration_ns and output_tokens:
            tokens_per_second = round(output_tokens / (eval_duration_ns / 1_000_000_000), 2)
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
        }

    def _stream_generate(self, payload: dict) -> str:
        url = f"{self.base_url}/api/generate"
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        accumulated: list[str] = []
        final_chunk: dict = {}
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        accumulated.append(token)
                        if self.on_llm_chunk:
                            self.on_llm_chunk("".join(accumulated))
                    if chunk.get("done"):
                        final_chunk = chunk
                        break
        except socket.timeout as exc:
            raise TimeoutError(
                f"Timed out after {self.timeout_seconds}s streaming from {url}"
            ) from exc
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            detail = body.strip() or exc.reason
            raise ValueError(f"HTTP {exc.code} streaming from {url}: {detail}") from exc
        self._extract_ollama_usage(final_chunk)
        response_text = "".join(accumulated)
        if not response_text:
            raise ValueError("Ollama streaming did not produce any output")
        return response_text

    def _discover_model(self) -> str:
        running = self._get_json(f"{self.base_url}/api/ps").get("models", [])
        if running:
            name = running[0].get("name", "").strip()
            if name:
                return name
        installed = self._get_json(f"{self.base_url}/api/tags").get("models", [])
        if installed:
            name = installed[0].get("name", "").strip()
            if name:
                return name
        raise ValueError(
            "No Ollama model configured or installed. Set OLLAMA_MODEL or pull a model first."
        )

    @staticmethod
    def _prepare_documents(documents: list[WebDocument]) -> list[WebDocument]:
        prepared: list[WebDocument] = []
        for doc in documents[:2]:
            prepared.append(
                WebDocument(
                    title=doc.title,
                    url=doc.url,
                    snippet=doc.snippet[:500],
                    content=doc.content[:3000],
                )
            )
        return prepared


def build_extractor(settings: Settings) -> Extractor:
    provider = settings.llm_provider
    if provider == "auto":
        if settings.gemini_api_key:
            provider = "gemini"
        elif settings.openai_api_key:
            provider = "openai"
        else:
            provider = "ollama"

    if provider == "openai":
        return OpenAIExtractor(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if provider == "gemini":
        return GeminiExtractor(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if provider == "ollama":
        return OllamaExtractor(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=max(settings.request_timeout_seconds, 300),
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
