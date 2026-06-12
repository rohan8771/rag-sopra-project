from typing import Any, Dict, List
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

from rank_bm25 import BM25Okapi

load_dotenv()

# Initialize embeddings, same as ingestion.py
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# For multi-query retrieval:
# Original query + 2 generated queries = 3 total queries.
# 3 queries * 5 docs each = 15 candidate docs before deduplication.
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# Initialize chat model
model = init_chat_model(
    "gpt-5-mini",
    model_provider="openai",
    temperature=0,
)


STOPWORDS = {
    "the", "is", "are", "am", "a", "an", "and", "or", "of", "to", "in",
    "for", "on", "with", "as", "by", "at", "from", "this", "that", "it",
    "be", "can", "how", "what", "why", "when", "where", "which", "do",
    "does", "did", "was", "were", "has", "have", "had", "i", "you"
}


def tokenize(text: str) -> List[str]:
    """
    Convert text into simple lowercase tokens for BM25.
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
    top_k: int = 4,
) -> List[Document]:
    """
    Rerank retrieved documents using BM25.

    Multi-query retrieval first collects candidates.
    BM25 then reorders those candidates using lexical relevance.
    """
    if not docs:
        return []

    tokenized_corpus = [
        tokenize(doc.page_content)
        for doc in docs
    ]

    tokenized_query = tokenize(query)

    # If tokenization removes everything from the query,
    # fall back to the original retrieved order.
    if not tokenized_query:
        return docs[:top_k]

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda item: item[0], reverse=True)

    return [
        doc for score, doc in scored_docs[:top_k]
    ]


def clean_generated_query(line: str) -> str:
    """
    Clean one generated query line.

    Handles outputs like:
    - query
    1. query
    2) query
    """
    line = line.strip()
    line = line.lstrip("-•").strip()

    # Remove numbering like "1. query" or "2) query"
    if len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]:
        line = line[2:].strip()

    # Remove surrounding quotes if present
    line = line.strip('"').strip("'").strip()

    return line


def generate_search_queries(user_query: str) -> List[str]:
    """
    Generate multiple search queries for retrieval.

    The original query is always included first.

    Final output:
        original query + 2 generated queries = 3 total queries
    """
    prompt = (
        "You are generating search queries for a LangChain documentation RAG system.\n"
        "Given the user's question, generate 2 different search queries that may help retrieve "
        "directly useful documentation.\n"
        "The queries should be concise, keyword-rich, and useful for searching technical docs.\n"
        "For broad questions, include terms like overview, use cases, applications, core components, "
        "agents, retrieval, tools, models, or memory when relevant.\n"
        "Do not answer the question.\n"
        "Return only the queries, one per line."
    )

    response = model.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=user_query),
        ]
    )

    generated_queries = [
        clean_generated_query(line)
        for line in response.content.splitlines()
        if clean_generated_query(line)
    ]

    # Include original query first, then only 2 generated queries.
    all_queries = [user_query] + generated_queries[:2]

    # Deduplicate while preserving order.
    unique_queries = []
    seen = set()

    for query in all_queries:
        normalized_query = query.lower().strip()

        if normalized_query and normalized_query not in seen:
            unique_queries.append(query)
            seen.add(normalized_query)

    return unique_queries


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate documents retrieved across multiple search queries.
    """
    unique_docs = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        content_preview = doc.page_content[:300]

        key = (source, page, content_preview)

        if key not in seen:
            unique_docs.append(doc)
            seen.add(key)

    return unique_docs


def multi_query_retrieve(query: str) -> List[Document]:
    """
    Retrieve documents using the original query plus 2 LLM-generated search queries.
    """
    search_queries = generate_search_queries(query)

    all_docs = []

    for search_query in search_queries:
        docs = retriever.invoke(search_query)
        all_docs.extend(docs)

    unique_docs = deduplicate_docs(all_docs)

    return unique_docs


# System prompt for the global agent
SYSTEM_PROMPT = (
    "You are an assistant for LangChain documentation. "

    "For any question about LangChain, LangGraph, LangSmith, agents, RAG, retrieval, "
    "tools, memory, vector stores, embeddings, chains, prompts, or related technical topics, "
    "you must use the retrieve_context tool before answering. "

    "You may call retrieve_context more than once if the first retrieved context is insufficient. "
    "If the retrieved context is mostly navigation, index listings, table-of-contents content, "
    "installation instructions, deployment instructions, or generic page lists, and it does not "
    "directly answer the user's question, call retrieve_context again with a more specific rewritten query. "

    "When rewriting a query for a second retrieval, preserve the user's intent but make the query "
    "more retrieval-friendly using likely documentation terms. "
    "For example, if the user asks 'Tell how LangChain is used', a better retrieval query could be "
    "'LangChain overview use cases applications agents retrieval tools models'. "

    "Only answer after you have retrieved context that directly supports the answer, or after retrieval "
    "still fails to find sufficient support. "

    "When you use retrieved documentation, you must obey these grounding rules strictly: "
    "1. Answer only using information present in the retrieved context. "
    "2. Do not use outside knowledge, general knowledge, memory, assumptions, or guessed facts. "
    "3. Do not invent, guess, or add documentation URLs. "
    "4. Do not cite any source that was not returned by the retrieve_context tool. "
    "5. If the retrieved context does not contain enough information to answer, say: "
    "'I could not find this in the retrieved documentation.' "
    "6. Do not fill gaps using your own knowledge. "
    "7. Do not provide extra examples unless they are present in the retrieved context. "
    "8. Do not offer to fetch, show, or pull specific documentation pages, sections, URLs, "
    "or topics unless those exact pages, sections, URLs, or topics are present in the "
    "retrieved context. "
    "9. Do not imply that a source exists in the vectorstore unless it was retrieved. "
    "10. At the end of the answer, do not suggest specific next sections or pages unless "
    "they were retrieved. "

    "If the user asks a non-documentation question that clearly does not require retrieval, "
    "you may answer directly. "

    "When citing sources, cite only the sources present in the retrieved context. "
    "Every factual claim about the documentation must be supported by retrieved context. "

    "Do NOT offer follow-up help."
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = multi_query_retrieve(query)

    reranked_docs = bm25_rerank(
        query=query,
        docs=retrieved_docs,
        top_k=4,
    )

    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            f"Content: {doc.page_content}"
        )
        for doc in reranked_docs
    )

    return serialized, reranked_docs


# In-memory checkpointer for short-term agent memory.
#
# This stores the agent's conversation state while the Python process is running.
checkpointer = InMemorySaver()

# Create the agent once globally.
#
# This is the reusable agent/graph object.
# Conversation-specific memory is not stored merely because this object is global;
# it is stored through the checkpointer using the thread_id passed at invoke time.
agent = create_agent(
    model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


def _extract_latest_context_docs(response_messages: List[Any]) -> List[Any]:
    """
    Extract retrieved Document artifacts only from the latest user turn.

    With checkpointed memory, response["messages"] may contain the full thread history.
    So we first find the latest HumanMessage, then only inspect ToolMessages after it.
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


def run_llm(query: str, thread_id: str) -> Dict[str, Any]:
    """
    Run the RAG agent using LangGraph/LangChain short-term memory.

    Args:
        query:
            The latest user message only.

        thread_id:
            Conversation/session identifier.
            The checkpointer uses this to load and save the agent's state.

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents used during this run
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


if __name__ == "__main__":
    test_thread_id = "test-thread"

    first_result = run_llm(
        query="What are deep agents?",
        thread_id=test_thread_id,
    )

    print("FIRST RESULT")
    print(first_result["answer"])
    print()

    second_result = run_llm(
        query="How are they different from normal agents?",
        thread_id=test_thread_id,
    )

    print("SECOND RESULT")
    print(second_result["answer"])