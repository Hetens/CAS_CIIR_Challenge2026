from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from agentic_search.config import Settings
from agentic_search.models import SearchResult


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int) -> list[SearchResult]:
        raise NotImplementedError


class BraveSearchProvider(SearchProvider):
    def __init__(self, api_key: str, timeout_seconds: int) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int) -> list[SearchResult]:
        request = Request(
            url=(
                "https://api.search.brave.com/res/v1/web/search"
                f"?q={quote_plus(query)}&count={limit}"
            ),
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        items = payload.get("web", {}).get("results", [])
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
            for item in items
            if item.get("url")
        ]


class SerpApiSearchProvider(SearchProvider):
    def __init__(self, api_key: str, timeout_seconds: int) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int) -> list[SearchResult]:
        request = Request(
            url=(
                "https://serpapi.com/search.json"
                f"?engine=google&q={quote_plus(query)}&num={limit}&api_key={quote_plus(self.api_key)}"
            ),
            headers={"Accept": "application/json"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        items = payload.get("organic_results", [])
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
            for item in items
            if item.get("link")
        ]


class DuckDuckGoSearchProvider(SearchProvider):
    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, limit: int) -> list[SearchResult]:
        request = Request(
            url=f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; AgenticSearchChallenge/1.0; "
                    "+https://example.com)"
                )
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="ignore")

        parser = _DuckDuckGoHTMLParser(limit=limit)
        parser.feed(payload)
        return parser.results


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.results: list[SearchResult] = []
        self.current_title: list[str] = []
        self.current_snippet: list[str] = []
        self.current_url = ""
        self.capture_title = False
        self.capture_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = attr_map.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._finalize_current()
            self.capture_title = True
            self.current_title = []
            self.current_snippet = []
            self.current_url = _normalize_duckduckgo_url(attr_map.get("href", ""))
            return
        if ("result__snippet" in classes) and tag in {"a", "div"} and self.current_url:
            self.capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.capture_title:
            self.capture_title = False
        if tag in {"a", "div"} and self.capture_snippet:
            self.capture_snippet = False
            self._finalize_current()

    def handle_data(self, data: str) -> None:
        if self.capture_title:
            self.current_title.append(data)
        if self.capture_snippet:
            self.current_snippet.append(data)

    def close(self) -> None:
        super().close()
        self._finalize_current()

    def _finalize_current(self) -> None:
        if len(self.results) >= self.limit:
            return
        title = _strip_html(" ".join(self.current_title))
        snippet = _strip_html(" ".join(self.current_snippet))
        if title and self.current_url:
            self.results.append(
                SearchResult(
                    title=title,
                    url=self.current_url,
                    snippet=snippet,
                )
            )
        self.current_title = []
        self.current_snippet = []
        self.current_url = ""
        self.capture_title = False
        self.capture_snippet = False


def _strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("&amp;", "&")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalize_duckduckgo_url(value: str) -> str:
    parsed = urlparse(value)
    params = parse_qs(parsed.query)
    if "uddg" in params and params["uddg"]:
        return unquote(params["uddg"][0])
    return value


def build_search_provider(settings: Settings) -> SearchProvider:
    if settings.search_provider == "duckduckgo":
        return DuckDuckGoSearchProvider(
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.search_provider == "brave":
        if not settings.brave_api_key:
            raise ValueError("BRAVE_API_KEY is required when SEARCH_PROVIDER=brave")
        return BraveSearchProvider(
            api_key=settings.brave_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.search_provider == "serpapi":
        if not settings.serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY is required when SEARCH_PROVIDER=serpapi")
        return SerpApiSearchProvider(
            api_key=settings.serpapi_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    raise ValueError(f"Unsupported SEARCH_PROVIDER: {settings.search_provider}")
