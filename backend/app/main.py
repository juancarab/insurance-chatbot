"""FastAPI application exposing the Insurance Chatbot API."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, get_settings


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


class AnswerFormatterProtocol(Protocol):
    """Interface implemented by all answer formatter strategies."""

    def format_answer(self, messages: List[Message], contexts: List[Source]) -> str:
        """Return the assistant message given the conversation and retrieved sources."""


class MockAnswerFormatter:
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


class LangChainAgentFormatter:
    """Adapter that delegates formatting to a LangChain-powered runner."""

    def __init__(self, runner: Callable[..., Any]):
        self._runner = runner

    @classmethod
    def from_settings(cls, settings: Settings) -> "LangChainAgentFormatter":
        """Resolve the runner configured through application settings."""

        target = settings.langchain_runner
        if not target:
            raise RuntimeError(
                "INSURANCE_CHATBOT_LANGCHAIN_RUNNER must be defined when "
                "INSURANCE_CHATBOT_FORMATTER=langchain. Use 'module:function'."
            )

        module_name, _, attr = target.partition(":")
        if not module_name or not attr:
            raise RuntimeError(
                "INSURANCE_CHATBOT_LANGCHAIN_RUNNER must follow the format "
                "'package.module:function_name'."
            )

        module = importlib.import_module(module_name)
        runner = getattr(module, attr)
        if not callable(runner):
            raise RuntimeError(
                f"The object '{attr}' loaded from '{module_name}' must be callable."
            )

        return cls(runner)

    def format_answer(self, messages: List[Message], contexts: List[Source]) -> str:
        try:
            result = self._runner(
                messages=[message.dict() for message in messages],
                contexts=[source.dict() for source in contexts],
            )
        except Exception as exc:  # pragma: no cover - defensive guard for integrations
            raise RuntimeError("LangChain runner failed to generate a response.") from exc

        if isinstance(result, dict):
            for key in ("output", "answer", "result", "content"):
                if key in result:
                    return str(result[key])
            return str(result)

        return str(result)


class GeminiAnswerFormatter:
    """Formatter that delegates response generation to Google Gemini."""

    def __init__(self, model: Any):
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiAnswerFormatter":
        """Configure the Gemini client using application settings."""

        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "google-generativeai must be installed to use the Gemini formatter."
            ) from exc

        api_key = settings.gemini_api_key
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY must be set when INSURANCE_CHATBOT_FORMATTER=gemini."
            )

        model_name = settings.gemini_model
        if not model_name:
            raise RuntimeError(
                "GEMINI_MODEL must be set when INSURANCE_CHATBOT_FORMATTER=gemini."
            )

        genai.configure(api_key=api_key)

        generation_config: Dict[str, Any] = {}
        if settings.gemini_temperature is not None:
            generation_config["temperature"] = settings.gemini_temperature

        if settings.gemini_top_p is not None:
            generation_config["top_p"] = settings.gemini_top_p

        if settings.gemini_max_output_tokens is not None:
            generation_config["max_output_tokens"] = settings.gemini_max_output_tokens

        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config or None,
        )

        return cls(model=model)

    def format_answer(self, messages: List[Message], contexts: List[Source]) -> str:
        conversation = "\n".join(
            f"{message.role.upper()}: {message.content}" for message in messages
        )

        if contexts:
            context_block = "\n".join(
                f"{source.title}: {source.snippet}" for source in contexts
            )
            context_prompt = (
                f"Relevant policy snippets:\n{context_block}\n\n"
            )
        else:
            context_prompt = "No policy snippets were retrieved for this turn.\n\n"

        system_prompt = (
            "You are an insurance assistant. Provide concise, accurate answers grounded in the"
            " provided policy snippets. If the context does not contain the answer, say you"
            " do not know. Respond in the same language as the user."
        )

        composed_prompt = (
            f"{system_prompt}\n\n"
            f"{context_prompt}Conversation history:\n{conversation}\n\n"
            "Assistant response:"
        )

        try:
            response = self._model.generate_content(composed_prompt)
        except Exception as exc:  # pragma: no cover - defensive guard for API failures
            raise RuntimeError("Gemini failed to generate a response.") from exc

        text_response = getattr(response, "text", None)

        if not text_response and getattr(response, "candidates", None):
            parts: List[str] = []
            for candidate in response.candidates:
                for part in getattr(candidate.content, "parts", []):
                    value = getattr(part, "text", None)
                    if value:
                        parts.append(value)
            text_response = "\n".join(parts)

        if not text_response:
            raise RuntimeError("Gemini returned an empty response.")

        return text_response.strip()


def build_formatter(settings: Settings) -> AnswerFormatterProtocol:
    """Return the formatter strategy indicated by configuration."""

    if settings.formatter == "langchain":
        return LangChainAgentFormatter.from_settings(settings)
    if settings.formatter == "gemini":
        return GeminiAnswerFormatter.from_settings(settings)

    return MockAnswerFormatter()


app = FastAPI(title="Insurance Chatbot API", version="0.1.0")
retriever = PolicyRetriever()
settings = get_settings()
formatter = build_formatter(settings)
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
        "formatter": settings.formatter,
    }

    return ChatResponse(answer=answer, sources=retrieved_sources, usage=usage)


@app.get("/chat", include_in_schema=False)
def chat_get_landing() -> Dict[str, Any]:
    """Helpful response for accidental GET requests to /chat."""

    return {
        "message": "Use POST /chat with a ChatRequest payload to talk to the bot.",
        "example_request": {
            "messages": [
                {"role": "user", "content": "What does the policy cover?"}
            ],
            "top_k": 3,
            "enable_web_search": False,
        },
        "schema_docs": "/docs#/default/chat_endpoint_chat__post",
    }


@app.get("/", include_in_schema=False)
def root() -> Dict[str, str]:
    """Simple landing endpoint for manual pokes at the API root."""

    return {
        "message": "Insurance Chatbot API is running. Use POST /chat or visit /docs.",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["health"])
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}
