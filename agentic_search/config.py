from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str
    ollama_base_url: str
    ollama_model: str
    search_provider: str
    brave_api_key: str
    serpapi_api_key: str
    search_result_count: int
    max_page_chars: int
    request_timeout_seconds: int


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "auto").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", ""),
        search_provider=os.getenv("SEARCH_PROVIDER", "duckduckgo").strip().lower(),
        brave_api_key=os.getenv("BRAVE_API_KEY", ""),
        serpapi_api_key=os.getenv("SERPAPI_API_KEY", ""),
        search_result_count=int(os.getenv("SEARCH_RESULT_COUNT", "5")),
        max_page_chars=int(os.getenv("MAX_PAGE_CHARS", "12000")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")),
    )
