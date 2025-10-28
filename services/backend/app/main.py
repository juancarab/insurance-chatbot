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
    id: Optional[str] = None
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

class AnswerFormatterProtocol(Protocol):
    """
    Interface implemented by all answer formatter strategies.
    A formatter can return a simple string or a dictionary for richer responses.
    """
    def format_answer(self, messages: List[Message], contexts: List[Source]) -> Dict[str, Any] | str:
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
        try:
            module = importlib.import_module(module_name)
            runner = getattr(module, attr)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(f"Failed to load LangChain runner '{target}'.") from exc
        if not callable(runner):
            raise RuntimeError(
                f"The object '{attr}' loaded from '{module_name}' must be callable."
            )
        return cls(runner)

    def format_answer(self, messages: List[Message], contexts: List[Source]) -> Dict[str, Any]:
        """
        Invokes the LangChain runner and expects a dictionary containing at least 'answer'.
        The runner can also override 'sources' if needed.
        """
        try:
            result = self._runner(
                messages=[message.dict() for message in messages],
                contexts=[source.dict() for source in contexts],
            )
        except Exception as exc: 
            raise RuntimeError("LangChain runner failed to generate a response.") from exc

        if isinstance(result, dict) and "answer" in result:
            return {
                "answer": str(result["answer"]),
                "sources": result.get("sources", [source.dict() for source in contexts]),
            }
        if isinstance(result, dict):
            for key in ("output", "answer", "result", "content"):
                if key in result:
                    return {"answer": str(result[key]), "sources": [source.dict() for source in contexts]}
            return {"answer": str(result), "sources": [source.dict() for source in contexts]}

        return {"answer": str(result), "sources": [source.dict() for source in contexts]}


class GeminiAnswerFormatter:
    """Formatter que delega a Google Gemini (SDK nuevo: google-genai)."""

    def __init__(self, client, model_name: str, generation_config: Dict[str, Any]):
        self._client = client
        self._model_name = model_name
        self._generation_config = generation_config or {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiAnswerFormatter":
        try:
            from google import genai  # SDK nuevo
        except ImportError as exc:
            raise RuntimeError(
                "Para usar el formatter Gemini instala el SDK nuevo: `pip install google-genai`."
            ) from exc

        api_key = settings.gemini_api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY debe estar configurado.")

        model_name = settings.gemini_model
        if not model_name:
            raise RuntimeError("GEMINI_MODEL debe estar configurado.")

        client = genai.Client(api_key=api_key)

        generation_config: Dict[str, Any] = {}
        if settings.gemini_temperature is not None:
            generation_config["temperature"] = settings.gemini_temperature
        if settings.gemini_top_p is not None:
            generation_config["top_p"] = settings.gemini_top_p
        if settings.gemini_max_output_tokens is not None:
            generation_config["max_output_tokens"] = settings.gemini_max_output_tokens

        return cls(client=client, model_name=model_name, generation_config=generation_config)

    def format_answer(self, messages: List[Message], contexts: List[Source]) -> str:
        conversation = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)

        if contexts:
            ctx = "\n".join(f"{s.title}: {s.snippet}" for s in contexts)
            context_prompt = f"Relevant policy snippets:\n{ctx}\n\n"
        else:
            context_prompt = "No policy snippets were retrieved for this turn.\n\n"

        system_prompt = (
            "You are an insurance assistant. Provide concise, accurate answers grounded in the "
            "provided policy snippets. If the context does not contain the answer, say you do not know. "
            "Respond in the same language as the user."
        )

        composed_prompt = (
            f"{system_prompt}\n\n"
            f"{context_prompt}Conversation history:\n{conversation}\n\n"
            "Assistant response:"
        )

        try:
            resp = self._client.models.generate_content(
                model=self._model_name,
                contents=composed_prompt,
                config=self._generation_config or None,
            )
        except Exception as exc:
            raise RuntimeError("Gemini falló al generar la respuesta.") from exc

        # SDK nuevo: texto directo en resp.text
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts = []
            for cand in resp.candidates:
                for part in getattr(cand.content, "parts", []):
                    v = getattr(part, "text", None)
                    if v:
                        parts.append(v)
            text = "\n".join(parts)

        if not text:
            raise RuntimeError("Gemini retornó una respuesta vacía.")

        return text.strip()

def build_formatter(settings: Settings) -> AnswerFormatterProtocol:
    """Return the formatter strategy indicated by configuration."""
    if settings.formatter == "langchain":
        return LangChainAgentFormatter.from_settings(settings)
    if settings.formatter == "gemini":
        return GeminiAnswerFormatter.from_settings(settings)
    return MockAnswerFormatter()


app = FastAPI(title="Insurance Chatbot API", version="0.1.0")
settings = get_settings()
formatter = build_formatter(settings)

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
    
    retrieved_sources = []

    formatted_result = formatter.format_answer(request.messages, retrieved_sources)

    if isinstance(formatted_result, dict):
        answer = formatted_result.get("answer", "Error: The model response did not contain 'answer'.")
        final_sources_data = formatted_result.get("sources", [])
        final_sources = [Source(**s) for s in final_sources_data]
    else:
        answer = str(formatted_result)
        final_sources = retrieved_sources

    usage = {
        "retrieved_documents": len(final_sources),
        "web_search_enabled": request.enable_web_search,
        "formatter": settings.formatter,
    }
    return ChatResponse(answer=answer, sources=final_sources, usage=usage)


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
