from typing import Any, Dict, List
from uuid import uuid4

import streamlit as st

from backend.core import LOADED_COLLECTION_NAMES, run_llm


# ---------------------------------------------------------------------------
# UI text
# ---------------------------------------------------------------------------
# The app is no longer only about LangChain documentation.
# It can now answer from all Chroma collections inside chroma_db:
#   - LangChain docs collection
#   - legacy backend source/code chunks
#   - LangExtract structured records
#   - any other collection we add later

WELCOME_MESSAGE = (
    "Ask me anything about the local knowledge base. "
    "I can answer using retrieved context from the LangChain docs and the legacy backend code."
)


# ---------------------------------------------------------------------------
# Source formatting helpers
# ---------------------------------------------------------------------------
# Backend returns retrieved LangChain Document objects.
#
# Each Document has:
#   - page_content
#   - metadata
#
# For the UI, we do not want to show the full page_content in the source list.
# We only show useful metadata that helps identify where the answer came from.


def _metadata_value(metadata: Dict[str, Any], key: str) -> str:
    """
    Safely get one metadata value as a clean string.

    Metadata values may be missing, None, numbers, lists, etc.
    For display purposes, we convert them to strings.
    """
    value = metadata.get(key)

    if value is None:
        return ""

    return str(value).strip()


def _format_sources(context_docs: List[Any]) -> List[Dict[str, str]]:
    """
    Extract unique source records from retrieved Document objects.

    Earlier version:
        Returned only source strings.

    New version:
        Returns richer dictionaries like:

        {
            "collection": "legacy_backend_structured",
            "source": "code/billing/BillingService.java",
            "file_path": "code/billing/BillingService.java",
            "class_name": "BillingService",
            "method_name": "calculateInvoice",
            "record_type": "method"
        }

    Why this is useful:
        Since we now retrieve from multiple Chroma collections, simply showing
        a source URL/path is not always enough. The collection name tells us
        whether the retrieved chunk came from LangChain docs, raw backend code,
        structured LangExtract records, etc.
    """
    sources: List[Dict[str, str]] = []
    seen = set()

    for doc in context_docs or []:
        metadata = getattr(doc, "metadata", None) or {}

        collection = _metadata_value(metadata, "collection") or "Unknown collection"

        source = (
            _metadata_value(metadata, "source")
            or _metadata_value(metadata, "source_path")
            or _metadata_value(metadata, "file_path")
            or _metadata_value(metadata, "path")
            or "Unknown source"
        )

        file_path = _metadata_value(metadata, "file_path")
        class_name = _metadata_value(metadata, "class_name")
        method_name = _metadata_value(metadata, "method_name")
        function_name = _metadata_value(metadata, "function_name")
        record_type = _metadata_value(metadata, "record_type")
        module = _metadata_value(metadata, "module")
        package = _metadata_value(metadata, "package")
        page = _metadata_value(metadata, "page")

        source_record = {
            "collection": collection,
            "source": source,
            "file_path": file_path,
            "class_name": class_name,
            "method_name": method_name,
            "function_name": function_name,
            "record_type": record_type,
            "module": module,
            "package": package,
            "page": page,
        }

        # Deduplicate source records.
        key = tuple(source_record.items())

        if key not in seen:
            sources.append(source_record)
            seen.add(key)

    return sorted(
        sources,
        key=lambda item: (
            item.get("collection", ""),
            item.get("source", ""),
            item.get("class_name", ""),
            item.get("method_name", ""),
        ),
    )


def _render_one_source(source_record: Dict[str, str]) -> None:
    """
    Render one source record in Streamlit.

    If source is a URL, render it as clickable markdown.
    Otherwise, render it as code-like text.
    """
    collection = source_record.get("collection") or "Unknown collection"
    source = source_record.get("source") or "Unknown source"

    st.markdown(f"**Collection:** `{collection}`")

    if source.startswith(("http://", "https://")):
        st.markdown(f"**Source:** [{source}]({source})")
    else:
        st.markdown(f"**Source:** `{source}`")

    # Optional metadata fields.
    # These are shown only if present.
    optional_fields = [
        ("File path", "file_path"),
        ("Class", "class_name"),
        ("Method", "method_name"),
        ("Function", "function_name"),
        ("Record type", "record_type"),
        ("Module", "module"),
        ("Package", "package"),
        ("Page", "page"),
    ]

    for label, key in optional_fields:
        value = source_record.get(key)

        if value:
            st.markdown(f"**{label}:** `{value}`")


def _render_sources(sources: List[Any]) -> None:
    for index, source in enumerate(sources, start=1):
        if isinstance(source, dict):
            st.markdown(f"#### Source {index}")
            _render_one_source(source)
        else:
            # Backward compatibility for old string-only sources.
            source_text = str(source)

            if source_text.startswith(("http://", "https://")):
                st.markdown(f"- [{source_text}]({source_text})")
            else:
                st.markdown(f"- `{source_text}`")

        if index != len(sources):
            st.divider()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _initialize_session_state() -> None:
    """
    Initialize all Streamlit session-level state.

    thread_id:
        Used by LangGraph/LangChain checkpointer to maintain real backend
        conversation memory.

    chat_messages:
        Used only for clean UI display.
        This is of course separate from the agent's conversation memory
    """
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
                "sources": [],
            }
        ]


def _reset_chat() -> None:
    """
    Reset the visible chat and backend agent thread.

    Important:
        Creating a new thread_id is necessary because the checkpointer stores
        agent memory by thread_id.

    If we only cleared chat_messages but kept the same thread_id, the backend
    could still remember the old conversation.
    """
    st.session_state.thread_id = str(uuid4())

    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "sources": [],
        }
    ]


# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Local RAG Knowledge Base Helper",
    layout="centered",
)

st.title("Local RAG Knowledge Base Helper")

_initialize_session_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Session")

    if st.button("Clear chat", use_container_width=True):
        _reset_chat()
        st.rerun()

    with st.expander("Session details"):
        st.markdown(f"**Thread ID:** `{st.session_state.thread_id}`")

    with st.expander("Loaded Chroma collections"):
        if LOADED_COLLECTION_NAMES:
            for collection_name in LOADED_COLLECTION_NAMES:
                st.markdown(f"- `{collection_name}`")
        else:
            st.markdown("_No collections loaded._")


# ---------------------------------------------------------------------------
# Render previous chat messages
# ---------------------------------------------------------------------------

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("Sources"):
                _render_sources(msg["sources"])


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

prompt = st.chat_input("Please provide your query…")


# ---------------------------------------------------------------------------
# Handle new user message
# ---------------------------------------------------------------------------

if prompt:
    user_message = {
        "role": "user",
        "content": prompt,
        "sources": [],
    }

    st.session_state.chat_messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating answer. Retrieving documents if needed…"):
                result: Dict[str, Any] = run_llm(
                    query=prompt,
                    thread_id=st.session_state.thread_id,
                )

                answer = str(result.get("answer", "")).strip()
                answer = answer or "(No answer returned.)"

                sources = _format_sources(result.get("context", []))

            st.markdown(answer)

            if sources:
                with st.expander("Sources Encountered by Agent During Execution"):
                    _render_sources(sources)

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }

            st.session_state.chat_messages.append(assistant_message)

        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)