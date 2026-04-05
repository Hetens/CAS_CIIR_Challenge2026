# Customizable Agent Search 

Customizable Agent Search challenge submission for 2026.

An end-to-end Python system that:

- accepts a topic query
- searches the web with a pluggable provider
- scrapes result pages
- uses an LLM to extract structured entities
- returns a traceable table in JSON or a Streamlit UI

This version now supports:

- `Gemini` via API key
- `Ollama` via local HTTP
- `OpenAI` via API key
- `DuckDuckGo` search with no search API key

## Features

- Provider abstraction for Brave Search and SerpAPI
- HTML fetching and lightweight text extraction
- LLM-driven entity extraction with source-backed fields
- Entity deduplication and confidence scoring
- CLI output and a Streamlit frontend
- Tests with mocked search, fetch, and LLM layers

## Quick start

1. Create a virtual environment.
2. Set the required environment variables.
3. Run either the CLI or the Streamlit frontend.

### Environment variables

Required:

- one LLM path:
- `GEMINI_API_KEY`
- or `OLLAMA_MODEL` with a running Ollama server
- or `OPENAI_API_KEY`

One search provider is required:

- none if you use `SEARCH_PROVIDER=duckduckgo`
- `BRAVE_API_KEY`
- or `SERPAPI_API_KEY`

Optional:

- `LLM_PROVIDER`: `auto`, `gemini`, `ollama`, or `openai`
- `OPENAI_MODEL`: defaults to `gpt-4.1-mini`
- `GEMINI_MODEL`: defaults to `gemini-3.0-flash-preview`
- `OLLAMA_BASE_URL`: defaults to `http://127.0.0.1:11434`
- `OLLAMA_MODEL`: defaults to `gemma4:e2b`
- `SEARCH_PROVIDER`: `duckduckgo`, `brave`, or `serpapi`
- `SEARCH_RESULT_COUNT`: defaults to `5`
- `MAX_PAGE_CHARS`: defaults to `12000`
- `REQUEST_TIMEOUT_SECONDS`: defaults to `15`

## Local-first examples

### Gemini

```powershell
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="your-key"
$env:SEARCH_PROVIDER="duckduckgo"
py -3 app.py cli "open source database tools"
```

### Ollama

Set `OLLAMA_MODEL` to whichever model you already have locally. If you want to try a Gemma-family model, use the exact name shown by your Ollama install.

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="your-local-model-name"
$env:SEARCH_PROVIDER="duckduckgo"
py -3 app.py cli "top pizza places in Brooklyn"
```

## CLI usage

```bash
py -3 app.py cli "AI startups in healthcare"
```

### CLI debug mode

```bash
py -3 app.py cli "AI startups in healthcare" --debug --log-file logs/search_runs.jsonl
```

This includes:

- timing breakdown for search, fetch, extraction, and total runtime
- throughput metrics like pages per second and entities per second
- searched result metadata and fetched document summaries
- a persistent JSONL log file for each run

## Streamlit UI

```bash
py -3 app.py streamlit
```

Then open [http://127.0.0.1:8501](http://127.0.0.1:8501).

The Streamlit sidebar also supports:

- `Debug mode`
- custom log file path
- in-app metrics and raw debug payload viewing

## Output schema

Each entity row contains:

- `entity_name`
- `entity_type`
- `description`
- `attributes`
- `confidence`
- `sources`

Each cell is source-traceable:

```json
{
  "value": "Acme Health",
  "source_url": "https://example.com/company",
  "source_title": "Acme Health",
  "evidence": "Acme Health builds AI tools for clinical teams."
}
```

## Design choices

- The pipeline is split into `search`, `scrape`, `llm`, and `pipeline` modules so providers and extraction strategies can be swapped independently.
- The extractor output is validated with `pydantic`, which makes provider output safer and keeps the downstream schema stable.
- The frontend is entirely `Streamlit`, which gives a faster local demo loop and easier inspectability for raw JSON and warnings.
- The scraper is standard-library only for portability; that keeps setup light, though it is less effective on JavaScript-heavy pages.
- The extraction schema forces provenance on every field rather than attaching sources only at the row level.
- Deduplication happens after extraction so evidence from multiple pages can merge into one entity row.
- Every run is logged to JSONL, which makes latency and throughput easy to audit across repeated searches.

## Benchmark Comparison

Performance comparison across providers and models, collected from real search runs logged in `logs/`.

### Gemini vs Ollama (gemma4:e2b) Summary

| Metric | Gemini (gemini-3-flash-preview) | Ollama (gemma4:e2b) |
|---|---|---|
| Avg total latency | ~25 s | ~155 s |
| Avg LLM latency | ~20 s | ~152 s |
| Avg tokens/sec | 142.86 | ~9.8 |
| Structured output quality | Correct schema on every run | Frequent fallback to metadata shaping |
| Token usage | ~26K tokens per run | ~2.5K tokens per run (smaller context window) |

### Run-by-Run Detail

| Query | LLM | Model | Search Results | Entities | Total (s) | LLM (s) | Tokens | Tok/s | Fallback |
|---|---|---|---|---|---|---|---|---|---|
| Database configs for healthcare | gemini | gemini-3-flash-preview | 6 | 4 | 24.82 | 19.95 | 26,206 | 142.86 | No |
| AI startups in Healthcare | auto (ollama) | gemma4:e2b | 5 | 5 | 283.50 | 277.76 | 3,724 | 9.24 | No |
| Database systems in healthcare | ollama | gemma4:e2b | 3 | 1 | 114.49 | 112.54 | 2,405 | 11.10 | No |
| Database systems in healthcare | ollama | gemma4:e2b | 3 | 3 | 63.91 | 61.55 | 1,970 | 7.91 | Yes |
| Database systems in healthcare | ollama | gemma4:e2b | 4 | 4 | 151.43 | 147.76 | 2,851 | 10.22 | Yes |
| Database systems in healthcare | ollama | gemma4:e2b | 3 | 3 | 162.80 | 160.30 | 2,977 | 10.42 | Yes |

### Key Observations

- **Gemini is ~6x faster** end-to-end and ~15x faster in token throughput compared to local Ollama with gemma4:e2b.
- **Ollama gemma4:e2b frequently falls back** to metadata-based entity shaping because its JSON output does not always conform to the required nested schema. 3 out of 4 dedicated Ollama runs used the fallback path.
- **Local models process far fewer tokens** (~2-4K vs ~26K) because the pipeline truncates documents to stay within the smaller context window, which reduces extraction quality.
- **Search and fetch latency is consistent** across providers (~1-5 s), so the LLM extraction step dominates total runtime, especially for local models.

## Known limitations

- The current scraper does not execute JavaScript.
- Search quality depends on the configured external provider.
- The schema is intentionally generic; domain-specific prompts or schemas would improve precision for categories like restaurants, startups, or developer tools.
- Network retries, caching, and parallel fetches are straightforward next steps but omitted to keep the submission small and readable.

## Testing

```bash
python -m unittest discover -s tests -v
```
