"""
Agentic RAG: agent decides whether to query the vector DB or answer directly.
Industry-standard pattern: retrieve only when the question needs knowledge-base context.
"""
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from src.llms import get_chat_model

SYSTEM_PROMPT = """You are a helpful assistant with access to a search tool that queries a knowledge base. The knowledge base may contain information about IPL teams, uploaded documents (PDFs, graphs, reports), or other indexed content.

**When to use the search tool:**
- Questions about IPL teams, players, history, stats, or records
- Questions about content from uploaded documents (e.g., graphs, reports, specific topics in the docs)
- Anything that needs specific facts from the knowledge base

**When NOT to use the search tool:**
- Greetings (e.g., "Hi", "Hello")
- General conversation or thanks
- Meta-questions (e.g., "What can you do?", "Help")
- Questions you can answer from general knowledge without the knowledge base

If the question needs context from the knowledge base, call the search tool first, then answer using the retrieved context. Otherwise, answer directly without calling any tool.
"""


def _format_docs_for_tool(docs):
    return "\n\n".join(
        f"[{i}] {d.page_content.strip()}" for i, d in enumerate(docs, 1)
    )


def create_rag_agent(retriever, llm_provider="openai", llm_kwargs=None):
    """Build agent with conditional retrieval tool."""
    llm_kwargs = llm_kwargs or {}

    @tool
    def search_ipl_knowledge(query: str) -> str:
        """Search the knowledge base for information. Use this when the user's question needs
        specific facts from uploaded documents or the knowledge base (e.g. IPL, graphs, reports)."""
        docs = retriever.invoke(query)
        return _format_docs_for_tool(docs) if docs else "No relevant results found."

    llm = get_chat_model(llm_provider, **llm_kwargs)
    agent = create_react_agent(
        llm,
        tools=[search_ipl_knowledge],
        prompt=SYSTEM_PROMPT,
    )
    return agent


def invoke_agent(agent, question: str):
    """Invoke agent and return (answer, context_from_search or None)."""
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])
    answer = ""
    context_str = None

    for m in messages:
        if hasattr(m, "content") and m.content:
            if "ToolMessage" in type(m).__name__:
                context_str = m.content
            elif "AIMessage" in type(m).__name__:
                answer = m.content  # last AI message wins

    if not answer and messages:
        last = messages[-1]
        answer = getattr(last, "content", None) or str(last)
    return answer, context_str
