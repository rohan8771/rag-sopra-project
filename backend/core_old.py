from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Initialize embeddings, same as ingestion.py
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# Initialize retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# Initialize chat model
model = init_chat_model("gpt-5-mini", model_provider="openai")

# System prompt for the global agent
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


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = retriever.invoke(query)

    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


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