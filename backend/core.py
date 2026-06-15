from typing import Any, Dict, List
import hashlib
import re

import chromadb
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langgraph.checkpoint.memory import InMemorySaver

from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERSIST_DIRECTORY = "chroma_db"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"

# For each user/tool query, retrieve this many docs from EACH Chroma collection.
#
# Example:
#   3 collections * k=5 = max 15 candidate docs before deduplication/BM25.
DOCS_PER_COLLECTION = 5

# Final number of docs passed to the LLM after BM25 reranking.
#
# Since we now have multiple collections, 6 gives the model a bit more room
# than the older top_k=4.
FINAL_DOCS_AFTER_RERANK = 6


# ---------------------------------------------------------------------------
# Model and embeddings
# ---------------------------------------------------------------------------

# Must match the embedding model used during ingestion.
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Main chat model used by the agent.
#
# Important:
#   In this simplified version, we do NOT use the model to generate extra
#   search queries. The model is only used by the agent for answering/tool use.
model = init_chat_model(
    CHAT_MODEL,
    model_provider="openai",
    temperature=0,
)


# ---------------------------------------------------------------------------
# Load all Chroma collections from chroma_db
# ---------------------------------------------------------------------------
# Your LangChain docs ingestion did not specify a collection name, so that data
# should be in the default Chroma/LangChain collection, usually "langchain".
#
# Your legacy backend ingestion created additional collections in the same
# chroma_db folder.
#
# Instead of hard-coding names, we discover all collections automatically.


def _get_collection_name(collection: Any) -> str:
    """
    Convert Chroma collection objects/names into a plain string.

    Different Chroma versions may return:
        - strings
        - collection objects with a .name attribute
    """
    if isinstance(collection, str):
        return collection

    name = getattr(collection, "name", None)

    if name:
        return str(name)

    return str(collection)


def _load_chroma_vectorstores() -> Dict[str, Chroma]:
    """
    Load one LangChain Chroma vectorstore for each collection in chroma_db.

    Returns:
        {
            "langchain": Chroma(...),
            "some_legacy_collection": Chroma(...),
            ...
        }
    """
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    raw_collections = client.list_collections()

    collection_names = [
        _get_collection_name(collection).strip()
        for collection in raw_collections
    ]

    collection_names = [
        name for name in collection_names
        if name
    ]

    if not collection_names:
        raise RuntimeError(
            f"No Chroma collections found inside '{PERSIST_DIRECTORY}'. "
            "Run ingestion first, then restart the app."
        )

    vectorstores: Dict[str, Chroma] = {}

    for collection_name in collection_names:
        vectorstores[collection_name] = Chroma(
            collection_name=collection_name,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

    return vectorstores


# Loaded once when backend/core.py is imported.
VECTORSTORES = _load_chroma_vectorstores()

# One retriever per collection.
RETRIEVERS = {
    collection_name: vectorstore.as_retriever(
        search_kwargs={"k": DOCS_PER_COLLECTION}
    )
    for collection_name, vectorstore in VECTORSTORES.items()
}

LOADED_COLLECTION_NAMES = sorted(RETRIEVERS.keys())


# ---------------------------------------------------------------------------
# BM25 reranking
# ---------------------------------------------------------------------------
# Flow:
#   1. User asks question.
#   2. Agent calls retrieve_context with ONE query.
#   3. We retrieve from every Chroma collection using that one query.
#   4. We deduplicate.
#   5. We BM25-rerank the candidates.
#   6. We pass only the top docs to the LLM.


STOPWORDS = {
    "the", "is", "are", "am", "a", "an", "and", "or", "of", "to", "in",
    "for", "on", "with", "as", "by", "at", "from", "this", "that", "it",
    "be", "can", "how", "what", "why", "when", "where", "which", "do",
    "does", "did", "was", "were", "has", "have", "had", "i", "you",
    "me", "my", "we", "our", "your"
}


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.

    Keeps letters and numbers.
    Removes punctuation.
    Removes common stopwords.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    return [
        token for token in tokens
        if token not in STOPWORDS
    ]


def bm25_rerank(
    query: str,
    docs: List[Document],
    top_k: int = FINAL_DOCS_AFTER_RERANK,
) -> List[Document]:
    """
    Rerank retrieved docs using BM25 lexical relevance.

    This helps especially when the user mentions exact things like:
        - class names
        - method names
        - file names
        - service names
        - framework names
    """
    if not docs:
        return []

    tokenized_corpus = [
        tokenize(doc.page_content)
        for doc in docs
    ]

    tokenized_query = tokenize(query)

    # If the query becomes empty after tokenization, keep original retrieval order.
    if not tokenized_query:
        return docs[:top_k]

    # If all docs become empty after tokenization, BM25 cannot help.
    if not any(tokenized_corpus):
        return docs[:top_k]

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda item: item[0], reverse=True)

    return [
        doc for score, doc in scored_docs[:top_k]
    ]


# ---------------------------------------------------------------------------
# Document metadata helpers
# ---------------------------------------------------------------------------


def _copy_doc_with_collection_metadata(
    doc: Document,
    collection_name: str,
) -> Document:
    """
    Copy a retrieved Document and add collection metadata.

    This is important because the app now searches multiple collections.
    The LLM and UI should know which collection a chunk came from.
    """
    metadata = dict(doc.metadata or {})
    metadata["collection"] = collection_name

    # main.py currently displays metadata["source"].
    # Some legacy chunks may use file_path instead of source.
    # So we create a reasonable fallback source if missing.
    if not metadata.get("source"):
        metadata["source"] = (
            metadata.get("source_path")
            or metadata.get("file_path")
            or metadata.get("path")
            or collection_name
    )

    return Document(
        page_content=doc.page_content,
        metadata=metadata,
    )


def _content_hash(text: str) -> str:
    """
    Create a stable hash for document content.
    Used as part of deduplication.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate docs.

    Duplicates can happen if similar/identical chunks exist across collections.
    """
    unique_docs = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata or {}

        key = (
            metadata.get("collection", ""),
            metadata.get("source", ""),
            metadata.get("page", ""),
            metadata.get("file_path", ""),
            metadata.get("class_name", ""),
            metadata.get("method_name", ""),
            metadata.get("chunk_id", ""),
            metadata.get("record_id", ""),
            _content_hash(doc.page_content),
        )

        if key not in seen:
            unique_docs.append(doc)
            seen.add(key)

    return unique_docs


def _format_metadata_for_context(metadata: Dict[str, Any]) -> str:
    """
    Format useful metadata for the LLM.

    We include selected keys only, so the context stays readable.
    """
    preferred_keys = [
        "collection",
        "source",
        "file_path",
        "page",
        "class_name",
        "method_name",
        "function_name",
        "record_type",
        "service",
        "module",
        "package",
    ]

    lines = []

    for key in preferred_keys:
        value = metadata.get(key)

        if value is not None and value != "":
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-query retrieval across all collections
# ---------------------------------------------------------------------------


def retrieve_from_all_collections(query: str) -> List[Document]:
    """
    Retrieve documents from every loaded Chroma collection

    If the agent calls:
        retrieve_context("billing invoice batch job")

    Then this function searches that exact query across:
        - the LangChain docs collection
        - the legacy backend collection(s)
        - any other collection inside chroma_db
    """
    all_docs: List[Document] = []

    for collection_name, retriever in RETRIEVERS.items():
        docs = retriever.invoke(query)

        docs_with_collection = [
            _copy_doc_with_collection_metadata(doc, collection_name)
            for doc in docs
        ]

        all_docs.extend(docs_with_collection)

    return deduplicate_docs(all_docs)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a grounded technical assistant for a local RAG system.\n\n"

    "The local knowledge base is stored in Chroma and may contain multiple collections, "
    "including LangChain/LangGraph/RAG documentation and legacy backend codebase knowledge. "
    f"The currently loaded Chroma collections are: {', '.join(LOADED_COLLECTION_NAMES)}.\n\n"

    "The user may ask about LangChain documentation, LangGraph, LangSmith, agents, RAG, retrieval, "
    "tools, memory, vector stores, embeddings, chains, prompts, or related technical topics. "
    "The user may also ask about the legacy backend codebase, Java classes, services, controllers, "
    "repositories, batch jobs, dependencies, business rules, database/table-like concepts, or "
    "modernization opportunities.\n\n"

    "For any question about the local documentation, legacy backend, codebase, modernization, "
    "RAG, LangChain, LangGraph, LangSmith, agents, retrieval, tools, memory, vector stores, "
    "embeddings, chains, prompts, or related technical topics, you must use the retrieve_context "
    "tool before answering.\n\n"

    "When using retrieve_context, pass one clear, specific search query. "

    "You may call retrieve_context again if the first retrieved context is insufficient or unrelated. "
    "When doing that, rewrite the query to be more specific while preserving the user's intent.\n\n"

    "Only answer after you have retrieved context that directly supports the answer, or after retrieval "
    "still fails to find sufficient support.\n\n"

    "Grounding rules:\n"
    "1. Answer only using information present in the retrieved context.\n"
    "2. Do not use outside knowledge, general knowledge, memory, assumptions, or guessed facts.\n"
    "3. Do not invent documentation URLs, file names, class names, method names, database tables, "
    "services, or architecture details.\n"
    "4. Do not cite or mention any source that was not returned by retrieve_context.\n"
    "5. If the retrieved context does not contain enough information to answer, say: "
    "'I could not find this in the retrieved knowledge base.'\n"
    "6. Do not fill gaps using your own knowledge.\n"
    "7. Do not provide extra examples unless they are present in the retrieved context.\n"
    "8. Do not imply that a source exists in the vectorstore unless it was retrieved.\n\n"

    "Modernization-analysis rule:\n"
    "You may make cautious modernization observations only when they are directly based on retrieved "
    "legacy backend context. Label such statements clearly as 'Inference'. Do not present inferred "
    "modernization suggestions as confirmed facts.\n\n"

    "Citation/source rule:\n"
    "When citing or referring to sources, use only the source, file_path, class_name, method_name, "
    "record_type, and collection metadata present in the retrieved context.\n"
    "Every factual claim about the documentation or codebase must be supported by retrieved context.\n\n"

    "If the user asks a non-technical or non-knowledge-base question that clearly does not require "
    "retrieval, you may answer directly.\n\n"

    "Do NOT offer follow-up help."
)


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant context from all local Chroma collections

    """
    retrieved_docs = retrieve_from_all_collections(query)

    reranked_docs = bm25_rerank(
        query=query,
        docs=retrieved_docs,
        top_k=FINAL_DOCS_AFTER_RERANK,
    )

    serialized = "\n\n---\n\n".join(
        (
            f"Metadata:\n"
            f"{_format_metadata_for_context(doc.metadata or {})}\n\n"
            f"Content:\n"
            f"{doc.page_content}"
        )
        for doc in reranked_docs
    )

    return serialized, reranked_docs


# ---------------------------------------------------------------------------
# Agent memory
# ---------------------------------------------------------------------------

# In-memory short-term memory.
#
# This persists only while the Python process is running.
# It is separated per chat session using thread_id.
checkpointer = InMemorySaver()


# Create the agent once globally.
agent = create_agent(
    model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


# ---------------------------------------------------------------------------
# Extract latest retrieved docs for Streamlit source display
# ---------------------------------------------------------------------------


def _extract_latest_context_docs(response_messages: List[Any]) -> List[Any]:
    """
    Extract retrieved Document artifacts only from the latest user turn.

    Because checkpointed memory can include the whole conversation history,
    we avoid showing sources from older turns.
    """
    latest_human_index = None

    for i in range(len(response_messages) - 1, -1, -1):
        if isinstance(response_messages[i], HumanMessage):
            latest_human_index = i
            break

    if latest_human_index is None:
        messages_to_inspect = response_messages
    else:
        messages_to_inspect = response_messages[latest_human_index + 1:]

    context_docs = []

    for message in messages_to_inspect:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return context_docs


# ---------------------------------------------------------------------------
# Public function called by Streamlit
# ---------------------------------------------------------------------------


def run_llm(query: str, thread_id: str) -> Dict[str, Any]:
    """
    Run the RAG agent.

    Args:
        query:
            Latest user message.

        thread_id:
            Session/conversation ID used by the checkpointer.

    Returns:
        {
            "answer": final assistant response,
            "context": retrieved documents from this turn
        }
    """
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )

    response_messages = response["messages"]

    answer = response_messages[-1].content
    context_docs = _extract_latest_context_docs(response_messages)

    return {
        "answer": answer,
        "context": context_docs,
    }


# ---------------------------------------------------------------------------
# Local smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_thread_id = "test-thread"

    print("Loaded Chroma collections:")
    for collection_name in LOADED_COLLECTION_NAMES:
        print(f"- {collection_name}")

    print("\nFIRST RESULT")
    first_result = run_llm(
        query="What are deep agents?",
        thread_id=test_thread_id,
    )
    print(first_result["answer"])

    print("\nSECOND RESULT")
    second_result = run_llm(
        query="Which classes or services are involved in billing in the legacy backend?",
        thread_id=test_thread_id,
    )
    print(second_result["answer"])