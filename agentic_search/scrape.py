from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.request import Request, urlopen

from agentic_search.models import SearchResult, WebDocument


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = _clean_text(data)
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.parts.append(text)


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def fetch_document(
    result: SearchResult,
    timeout_seconds: int,
    max_chars: int,
) -> WebDocument:
    request = Request(
        url=result.url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; AgenticSearchChallenge/1.0; "
                "+https://example.com)"
            )
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8", errors="ignore")

    parser = _HTMLTextExtractor()
    parser.feed(raw)
    content = " ".join(parser.parts)
    content = _clean_text(content)[:max_chars]
    title = " ".join(parser.title_parts).strip() or result.title or result.url

    return WebDocument(
        title=title,
        url=result.url,
        content=content,
        snippet=result.snippet,
    )
