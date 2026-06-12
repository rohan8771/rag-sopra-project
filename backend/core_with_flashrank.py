from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

# core.py is inside:
# project_folder/backend/core.py
#
# So:
# Path(__file__).resolve().parent        -> project_folder/backend
# Path(__file__).resolve().parent.parent -> project_folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FLASHRANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_CACHE_DIR = PROJECT_ROOT / ".flashrank_cache"
FLASHRANK_MODEL_DIR = FLASHRANK_CACHE_DIR / FLASHRANK_MODEL_NAME


# -------------------------------------------------------------------
# Initialize embeddings, vector store, retriever, reranker, and model
# -------------------------------------------------------------------

# Initialize embeddings, same as ingestion.py
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = Chroma(
    persist_directory=str(PROJECT_ROOT / "chroma_db"),
    embedding_function=embeddings,
)

# First-stage retriever.
#
# This gets a broader set of candidate documents from Chroma.
# The reranker will then choose the best few from these candidates.
candidate_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)

# Safety check:
#
# If this folder does not exist, FlashRank will try to download the model
# from Hugging Face again. We raise a clearer error instead.
if not FLASHRANK_MODEL_DIR.exists():
    raise FileNotFoundError(
        f"FlashRank model folder not found:\n"
        f"{FLASHRANK_MODEL_DIR}\n\n"
        f"Expected structure:\n"
        f"{FLASHRANK_CACHE_DIR}\\{FLASHRANK_MODEL_NAME}\\<model files>\n\n"
        f"Download and extract {FLASHRANK_MODEL_NAME}.zip into .flashrank_cache first."
    )

# Local FlashRank reranker.
#
# Because cache_dir points to project_folder/.flashrank_cache and the model
# folder already exists, FlashRank should load locally instead of downloading.
ranker = Ranker(
    model_name=FLASHRANK_MODEL_NAME,
    cache_dir=str(FLASHRANK_CACHE_DIR),
    max_length=512,
)

# Initialize chat model
model = init_chat_model("gpt-5-mini", model_provider="openai")


# -------------------------------------------------------------------
# System prompt
# -------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions about LangChain documentation. "
    "You have access to a tool that retrieves relevant documentation. "
    "Use the tool to find relevant information before answering documentation questions. "
    "If the user's question does not need retrieval, do not retrieve; answer directly. "
    "Use the conversation history to understand follow-up questions and references like "
    "'it', 'that', 'this', or 'they'. "
    "Always cite the sources you use in your answers, if any. "
    "If you cannot find the answer in the retrieved documentation, say so."
)


# -------------------------------------------------------------------
# Reranking helper
# -------------------------------------------------------------------

def rerank_documents(query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
    """
    Rerank candidate documents using FlashRank and return the top documents.

    Flow:
        1. Chroma gives us candidate docs.
        2. We convert those docs into FlashRank passages.
        3. FlashRank scores/ranks them against the query.
        4. We map the ranked results back to LangChain Document objects.
    """
    if not docs:
        return []

    passages = []

    for index, doc in enumerate(docs):
        passages.append(
            {
                "id": index,
                "text": doc.page_content,
                "meta": doc.metadata,
            }
        )

    rerank_request = RerankRequest(
        query=query,
        passages=passages,
    )

    ranked_results = ranker.rerank(rerank_request)

    reranked_docs = []

    for result in ranked_results[:top_n]:
        original_index = result["id"]
        original_doc = docs[original_index]

        reranked_doc = Document(
            page_content=original_doc.page_content,
            metadata={
                **original_doc.metadata,
                "rerank_score": result.get("score"),
            },
        )

        reranked_docs.append(reranked_doc)

    return reranked_docs


# -------------------------------------------------------------------
# Retrieval tool
# -------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve and rerank relevant documentation to help answer user queries about LangChain."""
    candidate_docs = candidate_retriever.invoke(query)
    retrieved_docs = rerank_documents(query, candidate_docs, top_n=4)

    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Response parsing
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Local test
# -------------------------------------------------------------------

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