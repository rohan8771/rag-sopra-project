from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

from typing import Any, Dict, List
from uuid import uuid4

import streamlit as st

from backend.core import run_llm


WELCOME_MESSAGE = (
    "Ask me anything about LangChain docs. "
    "I’ll retrieve relevant context and cite sources."
)


def _format_sources(context_docs: List[Any]) -> List[str]:
    """
    Extract unique source strings from retrieved LangChain Document objects.
    """
    sources = []

    for doc in context_docs or []:
        metadata = getattr(doc, "metadata", None) or {}
        source = metadata.get("source") or "Unknown"
        sources.append(str(source))

    return sorted(set(sources))


def _render_sources(sources: List[str]) -> None:
    """
    Display sources in Streamlit.

    If a source looks like a URL, render it as a clickable markdown link.
    Otherwise, render it as normal text.
    """
    for source in sources:
        if source.startswith(("http://", "https://")):
            st.markdown(f"- [{source}]({source})")
        else:
            st.markdown(f"- {source}")


def _initialize_session_state() -> None:
    """
    Initialize all Streamlit session-level state.

    chat_messages:
        Used only for clean UI display.

    thread_id:
        Used by LangGraph/LangChain checkpointer to maintain real agent memory.
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
    agent memory by thread_id. If we kept the same thread_id, the backend
    could still remember the old conversation even after the UI was cleared.
    """
    st.session_state.thread_id = str(uuid4())
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "sources": [],
        }
    ]


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

_initialize_session_state()

with st.sidebar:
    st.subheader("Session")

    if st.button("Clear chat", use_container_width=True):
        _reset_chat()
        st.rerun()

    with st.expander("Session details"):
        st.markdown(f"**Thread ID:** `{st.session_state.thread_id}`")

# Render existing chat messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("Sources"):
                _render_sources(msg["sources"])

prompt = st.chat_input("Ask a question about LangChain…")

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
                with st.expander("Sourced Encountered by Agent During Execution"):
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
def _get_clean_messages() -> List[Dict[str, str]]:
    """
    Convert Streamlit session messages into the clean format expected by LangChain.

    st.session_state.messages contains UI-only fields like "sources".
    The LLM only needs role + content.
    """
    return [
        {
            "role": msg["role"],
            "content": msg["content"],
        }
        for msg in st.session_state.messages
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "sources": [],
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating answer. Retrieving documents if needed…"):
                messages = _get_clean_messages()

                result: Dict[str, Any] = run_llm(messages)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)