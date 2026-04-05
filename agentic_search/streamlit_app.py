from __future__ import annotations

import html as _html
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
            min_value=30,
            max_value=180,
            value=max(30, min(base.request_timeout_seconds, 180)),
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
            st.caption("Auto order: Gemini, then OpenAI, then Ollama.")
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


def _summarize_attributes(attributes: dict[str, dict[str, Any]]) -> str:
    if not attributes:
        return ""
    parts = []
    for key, value in attributes.items():
        text = str(value.get("value") or "").strip()
        if text:
            parts.append(f"{key}: {text}")
    return " | ".join(parts)


_TH = "text-align:left;padding:8px 12px;border-bottom:2px solid rgba(128,128,128,0.3);white-space:nowrap;"
_TD = "padding:8px 12px;border-bottom:1px solid rgba(128,128,128,0.15);word-wrap:break-word;overflow-wrap:break-word;white-space:normal;"


def _html_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    p = [f'<table style="width:100%;border-collapse:collapse;font-size:0.875rem;"><thead><tr>']
    for h in headers:
        p.append(f'<th style="{_TH}">{_html.escape(h)}</th>')
    p.append("</tr></thead><tbody>")
    for row in rows:
        p.append("<tr>")
        for h in headers:
            p.append(f'<td style="{_TD}">{_html.escape(str(row.get(h, "")))}</td>')
        p.append("</tr>")
    p.append("</tbody></table>")
    return "".join(p)


def _metrics_html_table(rows: list[tuple[str, str, str]]) -> str:
    """Each row is (metric_name, value, tooltip_description)."""
    th = "text-align:left;padding:6px 10px;border-bottom:2px solid rgba(128,128,128,0.3);"
    td = "padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.15);"
    p = [
        '<table style="width:100%;border-collapse:collapse;font-size:0.875rem;">',
        f'<thead><tr><th style="{th}">Metric</th><th style="{th}text-align:right;">Value</th></tr></thead>',
        "<tbody>",
    ]
    for name, value, tooltip in rows:
        tip = f' title="{_html.escape(tooltip)}"' if tooltip else ""
        icon = (' <span style="display:inline-flex;align-items:center;justify-content:center;'
                'width:1em;height:1em;border-radius:50%;border:1px solid currentColor;'
                'font-size:0.65em;opacity:0.45;cursor:help;vertical-align:middle;'
                'margin-left:4px;">?</span>') if tooltip else ""
        p.append(
            f"<tr>"
            f'<td style="{td}"{tip}>{_html.escape(name)}{icon}</td>'
            f'<td style="{td}text-align:right;font-family:monospace;">{_html.escape(value)}</td>'
            f"</tr>"
        )
    p.append("</tbody></table>")
    return "".join(p)


def _table_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("results", []):
        rows.append(
            {
                "Name": row["entity_name"]["value"],
                "Type": row["entity_type"]["value"],
                "Description": row["description"]["value"],
                "Attributes": _summarize_attributes(row.get("attributes", {})),
                "Confidence": f"{float(row.get('confidence', 0.0)):.2f}",
                "Sources": len(row.get("sources", [])),
            }
        )
    return rows


def _render_cell(label: str, cell: dict[str, Any]) -> None:
    st.markdown(f"**{label}:** {cell.get('value') or ''}")
    evidence = str(cell.get("evidence") or "").strip()
    if evidence:
        st.caption(evidence)
    source_url = str(cell.get("source_url") or "").strip()
    source_title = str(cell.get("source_title") or "").strip() or source_url
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
    st.markdown(_html_table(_table_rows(payload)), unsafe_allow_html=True)

    with st.expander("Entity details and source traceability"):
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
                    st.write(f"Confidence: {float(row.get('confidence', 0.0)):.2f}")


def _render_runtime(payload: dict[str, Any]) -> None:
    runtime = payload.get("runtime")
    if not runtime:
        return
    st.subheader("Runtime")
    st.markdown(
        _html_table([
            {
                "LLM Provider": str(runtime.get("llm_provider") or ""),
                "LLM Model": str(runtime.get("llm_model") or ""),
                "Search Provider": str(runtime.get("search_provider") or ""),
                "Search Results": f"{runtime.get('actual_search_results', 0)}/{runtime.get('requested_search_results', 0)}",
            }
        ]),
        unsafe_allow_html=True,
    )


def _render_metrics(payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics")
    if not metrics:
        return
    st.subheader("Summary Statistics")

    tokens_per_second = metrics.get("llm_tokens_per_second")
    tps_display = "N/A" if tokens_per_second in (None, "") else f"{float(tokens_per_second):.2f}"

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Latency & Throughput")
        st.markdown(
            _metrics_html_table([
                ("Total latency", f"{metrics.get('total_latency_ms', 0) / 1000:.2f} s", "End-to-end time from query submission to structured output"),
                ("Search latency", f"{metrics.get('search_latency_ms', 0) / 1000:.2f} s", "Time spent querying the search provider API"),
                ("Fetch latency", f"{metrics.get('fetch_latency_ms', 0) / 1000:.2f} s", "Time spent downloading and parsing web pages"),
                ("LLM latency", f"{metrics.get('extract_latency_ms', 0) / 1000:.2f} s", "Time the LLM spent generating the entity extraction JSON"),
                ("Fetch throughput", f"{metrics.get('pages_per_second', 0):.2f} pages/s", "Web pages fetched and parsed per second"),
                ("Entity throughput", f"{metrics.get('entities_per_second', 0):.2f} entities/s", "Final entities produced per second of total runtime"),
            ]),
            unsafe_allow_html=True,
        )
    with col2:
        st.caption("Counts & Tokens")
        st.markdown(
            _metrics_html_table([
                ("Search results", str(metrics.get("search_results_count", 0)), "Number of results returned by the search provider"),
                ("Fetched documents", str(metrics.get("fetched_documents_count", 0)), "Web pages successfully downloaded and parsed"),
                ("Final entities", str(metrics.get("final_entities_count", 0)), "Entities remaining after deduplication and merging"),
                ("Prompt tokens", str(metrics.get("llm_prompt_tokens", 0)), "Tokens sent to the LLM in the extraction prompt"),
                ("Output tokens", str(metrics.get("llm_output_tokens", 0)), "Tokens generated by the LLM"),
                ("Total tokens", str(metrics.get("llm_total_tokens", 0)), "Combined prompt and output token count"),
                ("Tokens/sec", tps_display, "LLM generation speed in output tokens per second"),
            ]),
            unsafe_allow_html=True,
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
    st.write("Customizable Agent Search with configurable providers, model selection, and source-backed entity extraction.")

    settings, debug_mode, log_file = _build_runtime_settings()

    query = st.text_input("Topic query", placeholder="AI startups in healthcare")
    search_clicked = st.button("Search", type="primary", width="stretch")

    if search_clicked:
        if not query.strip():
            st.error("Please enter a query.")
            return
        try:
            pipeline = AgenticSearchPipeline(settings)

            status = st.status("Searching, scraping, and extracting entities...", expanded=True)
            llm_output_area = [None]

            def _on_llm_chunk(text: str) -> None:
                if llm_output_area[0] is None:
                    status.write("**LLM generating output...**")
                    llm_output_area[0] = status.empty()
                preview = text if len(text) <= 3000 else f"...{text[-3000:]}"
                llm_output_area[0].code(preview, language="json")

            if hasattr(pipeline.extractor, "on_llm_chunk"):
                pipeline.extractor.on_llm_chunk = _on_llm_chunk

            payload = pipeline.run(
                query.strip(),
                debug=debug_mode,
                log_path=log_file.strip() or None,
            ).to_dict()

            status.update(label="Search complete!", state="complete", expanded=False)

            _render_results(payload)
            _render_runtime(payload)
            _render_metrics(payload)
            if debug_mode and payload.get("warnings"):
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
