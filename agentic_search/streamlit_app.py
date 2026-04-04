from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from agentic_search.config import Settings, load_settings
from agentic_search.pipeline import AgenticSearchPipeline

GEMINI_MODEL_OPTIONS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

OPENAI_MODEL_OPTIONS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
]

COMMON_OLLAMA_MODEL_OPTIONS = [
    "gemma4:e2b",
    "gemma4:4b",
    "gemma3:4b",
    "qwen2.5:7b",
    "llama3.2:3b",
    "mistral:7b",
]


def _get_installed_ollama_models(base_url: str) -> list[str]:
    request = Request(
        url=f"{base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return []
    models = [item.get("name", "").strip() for item in payload.get("models", [])]
    return sorted([model for model in models if model])


def _resolve_index(options: list[str], current: str) -> int:
    try:
        return options.index(current)
    except ValueError:
        return 0


def _model_selectbox(label: str, options: list[str], current: str, key: str) -> str:
    normalized = []
    seen: set[str] = set()
    for option in [current, *options, "Custom..."]:
        if not option or option in seen:
            continue
        normalized.append(option)
        seen.add(option)
    selected = st.selectbox(
        label,
        normalized,
        index=_resolve_index(normalized, current if current in normalized else normalized[0]),
        key=f"{key}_select",
    )
    if selected == "Custom...":
        return st.text_input(
            f"Custom value for {label}",
            value=current,
            key=f"{key}_custom",
        ).strip()
    return selected


def _build_runtime_settings() -> tuple[Settings, bool, str]:
    base = load_settings()
    installed_ollama_models = _get_installed_ollama_models(base.ollama_base_url)

    with st.sidebar:
        st.subheader("Runtime Config")
        st.caption("`duckduckgo` is the safest choice when you want to preserve paid search quota.")

        llm_provider = st.selectbox(
            "LLM provider",
            ["auto", "gemini", "ollama", "openai"],
            index=_resolve_index(["auto", "gemini", "ollama", "openai"], base.llm_provider),
        )
        search_provider = st.selectbox(
            "Search provider",
            ["duckduckgo", "brave", "serpapi"],
            index=_resolve_index(["duckduckgo", "brave", "serpapi"], base.search_provider),
        )
        search_result_count = st.slider(
            "Requested search results",
            min_value=3,
            max_value=10,
            value=max(3, min(base.search_result_count, 10)),
        )
        request_timeout_seconds = st.slider(
            "Request timeout (seconds)",
            min_value=15,
            max_value=180,
            value=max(15, min(base.request_timeout_seconds, 180)),
        )
        max_page_chars = st.slider(
            "Max page characters",
            min_value=2000,
            max_value=12000,
            step=1000,
            value=max(2000, min(base.max_page_chars, 12000)),
        )

        gemini_model = base.gemini_model
        openai_model = base.openai_model
        ollama_model = base.ollama_model

        if llm_provider == "auto":
            st.caption("Auto resolution order: Gemini, then OpenAI, then Ollama.")
            gemini_model = _model_selectbox(
                "Auto Gemini model",
                GEMINI_MODEL_OPTIONS,
                base.gemini_model,
                "auto_gemini_model",
            )
            openai_model = _model_selectbox(
                "Auto OpenAI model",
                OPENAI_MODEL_OPTIONS,
                base.openai_model,
                "auto_openai_model",
            )
            ollama_model = _model_selectbox(
                "Auto Ollama model",
                installed_ollama_models or COMMON_OLLAMA_MODEL_OPTIONS,
                base.ollama_model or (installed_ollama_models[0] if installed_ollama_models else COMMON_OLLAMA_MODEL_OPTIONS[0]),
                "auto_ollama_model",
            )
        elif llm_provider == "gemini":
            gemini_model = _model_selectbox(
                "Gemini model",
                GEMINI_MODEL_OPTIONS,
                base.gemini_model,
                "gemini_model",
            )
        elif llm_provider == "openai":
            openai_model = _model_selectbox(
                "OpenAI model",
                OPENAI_MODEL_OPTIONS,
                base.openai_model,
                "openai_model",
            )
        elif llm_provider == "ollama":
            ollama_model = _model_selectbox(
                "Ollama model",
                installed_ollama_models or COMMON_OLLAMA_MODEL_OPTIONS,
                base.ollama_model or (installed_ollama_models[0] if installed_ollama_models else COMMON_OLLAMA_MODEL_OPTIONS[0]),
                "ollama_model",
            )

        if search_provider == "brave" and not base.brave_api_key:
            st.warning("`BRAVE_API_KEY` is not set. Brave search requests will fail.")
        if search_provider == "serpapi" and not base.serpapi_api_key:
            st.warning("`SERPAPI_API_KEY` is not set. SerpAPI requests will fail.")
        if llm_provider == "gemini" and not base.gemini_api_key:
            st.warning("`GEMINI_API_KEY` is not set. Gemini requests will fail.")
        if llm_provider == "openai" and not base.openai_api_key:
            st.warning("`OPENAI_API_KEY` is not set. OpenAI requests will fail.")

        debug_mode = st.checkbox("Debug mode", value=False)
        log_file = st.text_input("Log file", value="logs/search_runs.jsonl")

    settings = replace(
        base,
        llm_provider=llm_provider,
        openai_model=openai_model,
        gemini_model=gemini_model,
        ollama_model=ollama_model,
        search_provider=search_provider,
        search_result_count=search_result_count,
        max_page_chars=max_page_chars,
        request_timeout_seconds=request_timeout_seconds,
    )
    return settings, debug_mode, log_file


def _render_cell(label: str, cell: dict[str, Any]) -> None:
    st.markdown(f"**{label}:** {cell.get('value', '')}")
    evidence = str(cell.get("evidence", "")).strip()
    if evidence:
        st.caption(evidence)
    source_url = str(cell.get("source_url", "")).strip()
    source_title = str(cell.get("source_title", "")).strip() or source_url
    if source_url:
        st.markdown(f"[Source: {source_title}]({source_url})")


def _render_results(payload: dict[str, Any]) -> None:
    rows = payload.get("results", [])
    st.subheader("Structured Output")
    if not rows:
        st.info("No entities found.")
        return

    st.caption(
        f"{len(rows)} entities found from {len(payload.get('searched_urls', []))} pages."
    )
    for row in rows:
        with st.container(border=True):
            left, right = st.columns([1.3, 1])
            with left:
                _render_cell("Name", row["entity_name"])
                _render_cell("Description", row["description"])
            with right:
                _render_cell("Type", row["entity_type"])
                attributes = row.get("attributes", {})
                if attributes:
                    st.markdown("**Attributes**")
                    for key, value in attributes.items():
                        _render_cell(key, value)
                st.metric("Confidence", f"{float(row.get('confidence', 0.0)):.2f}")


def _render_metrics(payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics")
    if not metrics:
        return
    st.subheader("Performance")
    row1 = st.columns(4)
    row1[0].metric("Total Latency", f"{metrics.get('total_latency_ms', 0):.2f} ms")
    row1[1].metric("Search Latency", f"{metrics.get('search_latency_ms', 0):.2f} ms")
    row1[2].metric("Fetch Latency", f"{metrics.get('fetch_latency_ms', 0):.2f} ms")
    row1[3].metric("LLM Latency", f"{metrics.get('extract_latency_ms', 0):.2f} ms")

    row2 = st.columns(4)
    row2[0].metric("Fetch Throughput", f"{metrics.get('pages_per_second', 0):.2f} pages/s")
    row2[1].metric("Entity Throughput", f"{metrics.get('entities_per_second', 0):.2f} entities/s")
    row2[2].metric("Fetched Documents", str(metrics.get("fetched_documents_count", 0)))
    row2[3].metric("Final Entities", str(metrics.get("final_entities_count", 0)))

    row3 = st.columns(4)
    row3[0].metric("LLM Prompt Tokens", str(metrics.get("llm_prompt_tokens", 0)))
    row3[1].metric("LLM Output Tokens", str(metrics.get("llm_output_tokens", 0)))
    row3[2].metric("LLM Total Tokens", str(metrics.get("llm_total_tokens", 0)))
    tokens_per_second = metrics.get("llm_tokens_per_second")
    row3[3].metric(
        "LLM Tokens / Sec",
        "N/A" if tokens_per_second in (None, "") else f"{float(tokens_per_second):.2f}",
    )


def _render_runtime(payload: dict[str, Any]) -> None:
    runtime = payload.get("runtime")
    if not runtime:
        return
    st.subheader("Runtime")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM Provider", str(runtime.get("llm_provider", "")))
    col2.metric("LLM Model", str(runtime.get("llm_model", "")))
    col3.metric("Search Provider", str(runtime.get("search_provider", "")))
    col4.metric(
        "Search Results",
        f"{runtime.get('actual_search_results', 0)}/{runtime.get('requested_search_results', 0)}",
    )


def _render_debug_steps(payload: dict[str, Any]) -> None:
    debug = payload.get("debug") or {}
    steps = debug.get("steps") or []
    if not steps:
        return
    st.subheader("Debug Trace")
    for step in steps:
        label = f"Step {step.get('step')}: {step.get('name')}"
        with st.expander(label, expanded=step.get("step") == 0):
            st.code(json.dumps(step, indent=2), language="json")


def main() -> None:
    st.set_page_config(
        page_title="CAS_CIIR_Challenge2026",
        page_icon="Search",
        layout="wide",
    )

    st.title("Customizable Agentic Search")
    st.write("Customizable Agentic Search with configurable providers, model selection, and source-backed entity extraction.")

    settings, debug_mode, log_file = _build_runtime_settings()

    query = st.text_input("Topic query", placeholder="AI startups in healthcare")
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked:
        if not query.strip():
            st.error("Please enter a query.")
            return
        try:
            pipeline = AgenticSearchPipeline(settings)
            with st.spinner("Searching, scraping, and extracting entities..."):
                payload = pipeline.run(
                    query.strip(),
                    debug=debug_mode,
                    log_path=log_file.strip() or None,
                ).to_dict()
            _render_runtime(payload)
            _render_metrics(payload)
            _render_results(payload)
            if payload.get("warnings"):
                st.warning("\n".join(payload["warnings"]))
            if payload.get("log_file"):
                st.caption(f"Run logged to {payload['log_file']}")
            if payload.get("debug"):
                _render_debug_steps(payload)
                with st.expander("Full debug payload"):
                    st.code(json.dumps(payload["debug"], indent=2), language="json")
            with st.expander("Raw JSON"):
                st.code(json.dumps(payload, indent=2), language="json")
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
