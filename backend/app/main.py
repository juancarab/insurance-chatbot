"""FastAPI application exposing the Insurance Chatbot API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a chat message."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Input payload for the /chat endpoint."""

    messages: List[Message] = Field(
        ..., description="Ordered list of chat messages, latest last."
    )
    top_k: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of policy snippets to retrieve for grounding.",
    )
    enable_web_search: bool = Field(
        False,
        description="Whether to allow the backend to use a web-search fallback.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional client-provided metadata."
    )


class Source(BaseModel):
    """A retrieved source used to craft the answer."""

    id: str
    title: str
    snippet: str
    url: Optional[str] = None


class ChatResponse(BaseModel):
    """Response structure for the /chat endpoint."""

    answer: str
    sources: List[Source]
    usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Implementation-specific diagnostics (latency, token counts, etc.).",
    )


class PolicyRetriever:
    """Stub retriever that mimics returning policy snippets."""

    def search(self, query: str, *, top_k: int) -> List[Source]:
        # This is a pure mock that returns deterministic placeholder data.
        return [
            Source(
                id=f"policy-{i+1}",
                title=f"Mock Policy Document {i+1}",
                snippet=f"Relevant excerpt {i+1} for query: '{query}'.",
                url=f"https://example.com/policies/{i+1}",
            )
            for i in range(top_k)
        ]


class WebSearchClient:
    """Stub client for optional web search fallback."""

    def search(self, query: str) -> List[Source]:
        return [
            Source(
                id="web-1",
                title="Mock Web Result",
                snippet=f"Web search result related to '{query}'.",
                url="https://search.example.com/mock",
            )
        ]


class AnswerFormatter:
    """Stub answer formatter that would normally invoke an LLM."""

    def format_answer(self, messages: List[Message], contexts: List[Source]) -> str:
        latest_user_message = next(
            (message for message in reversed(messages) if message.role == "user"),
            None,
        )
        question = latest_user_message.content if latest_user_message else "your question"
        bullet_points = "\n".join(
            f"- {source.title}: {source.snippet}" for source in contexts
        )
        return (
            "This is a placeholder answer. Once the LLM formatter is integrated, "
            "it will generate a detailed response grounded on the retrieved "
            "policy documents.\n\n"
            f"For now, here's what we found related to '{question}':\n{bullet_points}"
        )


app = FastAPI(title="Insurance Chatbot API", version="0.1.0")
retriever = PolicyRetriever()
formatter = AnswerFormatter()
web_search = WebSearchClient()


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    latest_message = request.messages[-1]
    if latest_message.role != "user":
        raise HTTPException(
            status_code=400,
            detail="The latest message in the conversation must come from the user.",
        )

    retrieved_sources = retriever.search(latest_message.content, top_k=request.top_k)

    if request.enable_web_search:
        retrieved_sources.extend(web_search.search(latest_message.content))

    answer = formatter.format_answer(request.messages, retrieved_sources)

    usage = {
        "retrieved_documents": len(retrieved_sources),
        "web_search_enabled": request.enable_web_search,
    }

    return ChatResponse(answer=answer, sources=retrieved_sources, usage=usage)


@app.get("/health", tags=["health"])
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}
