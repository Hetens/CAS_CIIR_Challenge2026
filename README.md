# Customizable Agentic Search

CIIR Agentic Search Challenge submission -- an end-to-end system that takes a topic query, searches the web, scrapes result pages, uses an LLM to extract structured entities, and returns a source-traceable table.

## Demo

<video src="https://github.com/Hetens/CAS_CIIR_Challenge2026/raw/main/assets/CIIR_Challenge.mp4" controls width="700"></video>

## Approach

The pipeline runs in four stages:

1. **Search** -- query a pluggable provider (DuckDuckGo, Brave, SerpAPI) and collect result URLs.
2. **Fetch** -- download each page and extract clean text with a stdlib-only scraper.
3. **Extract** -- send documents to an LLM (Gemini, OpenAI, or local Ollama) with a structured JSON schema prompt. Output is validated with Pydantic; if the LLM returns malformed JSON, a fallback shaper builds entities from page metadata.
4. **Deduplicate** -- merge entities by name, keeping the higher-confidence version and combining attributes and sources.

Every cell value carries `source_url`, `source_title`, and `evidence` so each fact is traceable to its origin page.

## Setup

```bash
pip install -r requirements.txt
```

### Environment variables

At least one LLM provider is required:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OLLAMA_MODEL` | Ollama model name (requires a running Ollama server) |

Search defaults to DuckDuckGo (no key needed). For paid providers set `BRAVE_API_KEY` or `SERPAPI_API_KEY`.

Optional tuning:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto`, `gemini`, `ollama`, or `openai` |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Gemini model name |
| `OPENAI_MODEL` | `gpt-4.1-mini` | OpenAI model name |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo`, `brave`, or `serpapi` |
| `SEARCH_RESULT_COUNT` | `5` | Results to request from search |
| `MAX_PAGE_CHARS` | `12000` | Max characters per fetched page |
| `REQUEST_TIMEOUT_SECONDS` | `15` | HTTP timeout |

### Run

```bash
# Streamlit UI
py -3 app.py streamlit

# CLI
py -3 app.py cli "AI startups in healthcare"

# CLI with debug logging
py -3 app.py cli "AI startups in healthcare" --debug --log-file logs/search_runs.jsonl
```

### Tests

```bash
python -m unittest discover -s tests -v
```

## Design Decisions

| Decision | Rationale |
|---|---|
| **Modular pipeline** (`search`, `scrape`, `llm`, `pipeline`) | Providers and strategies can be swapped independently without touching other layers. |
| **Pydantic validation on LLM output** | Catches malformed JSON from weaker models and keeps the downstream schema stable. |
| **Per-cell provenance** (`source_url`, `evidence`) | Every value is traceable to a specific page and quote, not just at the row level. |
| **Fallback entity shaping** | When the LLM returns invalid JSON, entities are built from page metadata so the pipeline never returns empty. |
| **Stdlib-only scraper** | Keeps dependencies light and setup portable; trades off JavaScript rendering. |
| **JSONL run logging** | Every run is appended to a log file for latency/throughput auditing across experiments. |
| **Streaming output for local models** | Ollama responses stream token-by-token into the Streamlit UI so the user sees progress during long extractions. |
| **DuckDuckGo as default search** | No API key required, which means zero-config setup for evaluation. |

## Beyond the Basics

- **Three LLM providers** with automatic fallback ordering (Gemini → OpenAI → Ollama).
- **Streaming LLM output** in the Streamlit UI for slow local models -- tokens appear as they generate.
- **Runtime-configurable sidebar** -- switch providers, models, result counts, and timeouts without restarting.
- **Confidence scoring and entity deduplication** -- merges duplicate entities from multiple pages, keeping the highest-confidence version.
- **Full debug trace** -- step-by-step pipeline inspection (query preprocessing, search results, raw LLM output, structured entities, timing breakdown).
- **Benchmark logging** -- every run is logged to JSONL with latency, token counts, and throughput metrics.

## Benchmark Comparison

Performance comparison from real runs logged in `logs/`.

| Metric | Gemini (gemini-3-flash-preview) | Ollama (gemma4:e2b) |
|---|---|---|
| Avg total latency | ~25 s | ~155 s |
| Avg LLM latency | ~20 s | ~152 s |
| Avg tokens/sec | 142.86 | ~9.8 |
| Structured output quality | Correct schema every run | Frequent fallback to metadata shaping |
| Token usage per run | ~26K | ~2.5K (truncated context) |

<details>
<summary>Run-by-run detail</summary>

| Query | LLM | Model | Results | Entities | Total (s) | LLM (s) | Tokens | Tok/s | Fallback |
|---|---|---|---|---|---|---|---|---|---|
| Database configs for healthcare | gemini | gemini-3-flash-preview | 6 | 4 | 24.82 | 19.95 | 26,206 | 142.86 | No |
| AI startups in Healthcare | auto→ollama | gemma4:e2b | 5 | 5 | 283.50 | 277.76 | 3,724 | 9.24 | No |
| Database systems in healthcare | ollama | gemma4:e2b | 3 | 1 | 114.49 | 112.54 | 2,405 | 11.10 | No |
| Database systems in healthcare | ollama | gemma4:e2b | 3 | 3 | 63.91 | 61.55 | 1,970 | 7.91 | Yes |
| Database systems in healthcare | ollama | gemma4:e2b | 4 | 4 | 151.43 | 147.76 | 2,851 | 10.22 | Yes |
| Database systems in healthcare | ollama | gemma4:e2b | 3 | 3 | 162.80 | 160.30 | 2,977 | 10.42 | Yes |

</details>

**Key takeaways:** Gemini is ~6x faster end-to-end and ~15x faster in token throughput. Ollama gemma4:e2b frequently falls back to metadata shaping because its JSON doesn't conform to the nested schema. The LLM extraction step dominates total runtime; search and fetch are consistently 1-5 s.

## Known Limitations

- The scraper does not execute JavaScript, so JS-rendered pages return minimal content.
- Search quality depends on the external provider; DuckDuckGo occasionally returns lower-relevance results.
- The extraction schema is intentionally generic -- domain-specific prompts would improve precision for categories like restaurants or developer tools.
- Network retries, caching, and parallel fetches are omitted to keep the submission small and readable.
