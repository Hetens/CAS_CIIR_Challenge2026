from __future__ import annotations

import json
from typing import Any

import streamlit as st

from agentic_search.config import load_settings
from agentic_search.pipeline import AgenticSearchPipeline


@st.cache_resource
def get_pipeline() -> AgenticSearchPipeline:
    return AgenticSearchPipeline(load_settings())


def _cell_html(cell: dict[str, Any]) -> str:
    value = str(cell.get("value", ""))
    evidence = str(cell.get("evidence", ""))
    source_url = str(cell.get("source_url", ""))
    source_title = str(cell.get("source_title", "") or source_url)
    return (
        f"<div><strong>{value}</strong></div>"
        f"<div style='color:#6d6255; font-size:0.92rem; margin-top:0.25rem'>{evidence}</div>"
        f"<div style='margin-top:0.35rem'><a href='{source_url}' target='_blank'>{source_title}</a></div>"
    )


def _attributes_html(attributes: dict[str, dict[str, Any]]) -> str:
    if not attributes:
        return ""
    blocks = []
    for key, value in attributes.items():
        blocks.append(
            f"<div style='margin-bottom:0.8rem'><div style='font-weight:700'>{key}</div>{_cell_html(value)}</div>"
        )
    return "".join(blocks)


def _render_results(payload: dict[str, Any]) -> None:
    rows = payload.get("results", [])
    if not rows:
        st.info("No entities found.")
        return

    st.caption(
        f"{len(rows)} entities found from {len(payload.get('searched_urls', []))} pages."
    )
    for row in rows:
        with st.container(border=True):
            left, right = st.columns([1.2, 1])
            with left:
                st.markdown(_cell_html(row["entity_name"]), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='margin-top:0.9rem'>{_cell_html(row['description'])}</div>",
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown(
                    f"<div style='margin-bottom:0.9rem'>{_cell_html(row['entity_type'])}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(_attributes_html(row.get("attributes", {})), unsafe_allow_html=True)
                st.metric("Confidence", f"{float(row.get('confidence', 0.0)):.2f}")


def _render_metrics(payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics")
    if not metrics:
        return
    st.subheader("Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Latency", f"{metrics.get('total_latency_ms', 0):.2f} ms")
    col2.metric("Search Latency", f"{metrics.get('search_latency_ms', 0):.2f} ms")
    col3.metric("Fetch Throughput", f"{metrics.get('pages_per_second', 0):.2f} pages/s")
    col4.metric("Entity Throughput", f"{metrics.get('entities_per_second', 0):.2f} entities/s")


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
        page_title="Agentic Search",
        page_icon="Search",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #efe2cf 0%, #f6f2ea 28%, #f8f5ef 100%);
        }
        [data-testid="stAppViewContainer"] {
            color: #1d1a16;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Agentic Search")
    st.write("Search a topic, scrape relevant pages, and extract source-backed entities.")

    with st.sidebar:
        settings = load_settings()
        st.subheader("Runtime")
        st.write(f"LLM provider: `{settings.llm_provider}`")
        st.write(f"Search provider: `{settings.search_provider}`")
        st.write(f"Result count: `{settings.search_result_count}`")
        debug_mode = st.checkbox("Debug mode", value=False)
        log_file = st.text_input("Log file", value="logs/search_runs.jsonl")

    query = st.text_input("Topic query", placeholder="AI startups in healthcare")
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked:
        if not query.strip():
            st.error("Please enter a query.")
            return
        try:
            with st.spinner("Searching, scraping, and extracting entities..."):
                payload = get_pipeline().run(
                    query.strip(),
                    debug=debug_mode,
                    log_path=log_file.strip() or None,
                ).to_dict()
            _render_runtime(payload)
            _render_metrics(payload)
            _render_results(payload)
            if payload.get("warnings"):
                st.warning("\n".join(payload["warnings"]))
            if payload.get("debug", {}).get("log_file"):
                st.caption(f"Run logged to {payload['debug']['log_file']}")
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
