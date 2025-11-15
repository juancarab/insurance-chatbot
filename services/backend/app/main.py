"""FastAPI application exposing the Insurance Chatbot API."""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Represents a chat message."""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Input payload for the /chat endpoint."""
    messages: List[Message] = Field(
        ..., description="Ordered list of chat messages, latest last."
    )

    query: str = Field(
        "",
        description="Último mensaje del usuario (opcional si ya viene en messages)"
    )

    top_k: int = Field(
        4, ge=1, le=10,
        description="Número de fragmentos a recuperar para grounding."
    )
    enable_web_search: bool = Field(
        False,
        description="Permite o no el uso de la herramienta de búsqueda web."
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadatos opcionales enviados por el cliente."
    )

    debug: bool = False
    language: str = "es"


class Source(BaseModel):
    """A retrieved source used to craft the answer."""
    id: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    url: Optional[str] = None
    file_name: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str | int] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    """Response structure for the /chat endpoint."""
    answer: str
    sources: List[Source] = []
    usage: Optional[dict] = None
    debug: Optional[dict] = None


class AnswerFormatterProtocol(Protocol):
    """
    Interface implemented by all answer formatter strategies.
    A formatter can return a simple string or a dictionary for richer responses.
    """
    def format_answer(
        self,
        messages: List[Message],
        contexts: List[Source],
        *,
        top_k: int,
        enable_web_search: bool,
        debug: bool,
        language: str,
    ) -> Dict[str, Any] | str:
        """Return the assistant message given the conversation and retrieved sources."""


class MockAnswerFormatter:
    """Stub answer formatter that would normally invoke an LLM."""

    def format_answer(
        self,
        messages: List[Message],
        contexts: List[Source],
        *,
        top_k: int,
        enable_web_search: bool,
        debug: bool,
        language: str,
    ) -> Dict[str, Any] | str:
        latest_user_message = next(
            (message for message in reversed(messages) if message.role == "user"),
            None,
        )
        question = latest_user_message.content if latest_user_message else "your question"
        bullet_points = "\n".join(
            f"- {s.title or s.file_name or 'Fuente'}: {(s.snippet or '').strip()}"
            for s in contexts
        )
        answer = (
            f"(mock) Responding n {language}. "
            "Once the LLM is integrated, it will generate a response based on the sources.\n\n"
            f"Request: '{question}'\n{bullet_points}"
        )

        if debug:
            sources_payload = [s.dict() for s in contexts]
            debug_payload: Dict[str, Any] = {
                "formatter": "mock",
                "messages": [m.dict() for m in messages],
                "contexts": sources_payload,
                "top_k": top_k,
                "enable_web_search": enable_web_search,
                "language": language,
            }
            return {"answer": answer, "sources": sources_payload, "debug": debug_payload}

        return answer


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

    async def format_answer(
        self,
        messages: List[Message],
        contexts: List[Source],
        *,
        top_k: int,
        enable_web_search: bool,
        debug: bool,
        language: str,
    ) -> Dict[str, Any]:
        """
        Invokes the LangChain/LangGraph runner and expects a dict with 'answer'.
        The runner can also override 'sources' and include 'usage'/'debug'.
        """
        try:
            result = await self._runner(
                messages=[m.dict() for m in messages],
                contexts=[s.dict() for s in contexts],
                top_k=top_k,
                enable_web_search=enable_web_search,
                debug=debug,
                language=language,
            )
        except Exception as exc:
            logger.error("LangChain runner failed to generate a response", exc_info=True)
            raise RuntimeError("LangChain runner failed to generate a response.") from exc

        if isinstance(result, dict):
            answer = str(
                result.get(
                    "answer",
                    result.get("output", result.get("result", result.get("content", ""))),
                )
            )
            sources = result.get("sources", [s.dict() for s in contexts])
            out: Dict[str, Any] = {"answer": answer, "sources": sources}

            if "usage" in result:
                out["usage"] = result["usage"]
            if "debug" in result:
                out["debug"] = result["debug"]
            return out

        return {"answer": str(result), "sources": [s.dict() for s in contexts]}


class GeminiAnswerFormatter:
    """Formatter que delega a Google Gemini (SDK nuevo: google-genai)."""

    def __init__(self, client, model_name: str, generation_config: Dict[str, Any]):
        self._client = client
        self._model_name = model_name
        self._generation_config = generation_config or {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiAnswerFormatter":
        try:
            from google import genai  
        except ImportError as exc:
            raise RuntimeError(
                "To use the Gemini formatter, install the SDK: `pip install google-genai`."
            ) from exc

        api_key = settings.gemini_api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY shoudl be configured.")

        model_name = settings.gemini_model
        if not model_name:
            raise RuntimeError("GEMINI_MODEL should be configured.")

        client = genai.Client(api_key=api_key)

        generation_config: Dict[str, Any] = {}
        if settings.gemini_temperature is not None:
            generation_config["temperature"] = settings.gemini_temperature
        if settings.gemini_top_p is not None:
            generation_config["top_p"] = settings.gemini_top_p
        if settings.gemini_max_output_tokens is not None:
            generation_config["max_output_tokens"] = settings.gemini_max_output_tokens

        return cls(client=client, model_name=model_name, generation_config=generation_config)

    def format_answer(
        self,
        messages: List[Message],
        contexts: List[Source],
        *,
        top_k: int,
        enable_web_search: bool,
        debug: bool,
        language: str,
    ) -> str:
        conversation = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)

        if not contexts:
            no_ctx_answer = (
                "I couldn’t retrieve any policy fragments for this query. "
                "Please provide the exact plan name or the policy’s particular conditions so I can answer precisely."
            )

            if debug:
                return {
                    "answer": no_ctx_answer,
                    "sources": [],
                    "debug": {
                        "formatter": "gemini",
                        "reason": "no_contexts",
                    },
                }

            return no_ctx_answer

        ctx = "\n".join(
            f"{(s.title or s.file_name or 'Source')}: {(s.snippet or '').strip()}"
            for s in contexts
        )
        context_prompt = f"Relevant fragments from internal documents:\n{ctx}\n\n"

        system_prompt = (
            "You are an insurance assistant. You must answer ONLY using the information found in the fragments below. "
            "If the fragments do not explicitly contain the answer, say you don't know and ask the user for the exact plan "
            "or for the policy's particular conditions. "
            f"ALWAYS answer in {language}."
        )

        composed_prompt = (
            f"{system_prompt}\n\n"
            f"{context_prompt}"
            f"Conversation history:\n{conversation}\n\n"
            "Assistant answer:"
        )

        try:
            resp = self._client.models.generate_content(
                model=self._model_name,
                contents=composed_prompt,
                config=self._generation_config or None,
            )
        except Exception as exc:
            logger.error("Gemini failed to generate the answer", exc_info=True)
            raise RuntimeError("Gemini failed to generate the answer.") from exc

        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts: List[str] = []
            for cand in resp.candidates:
                for part in getattr(cand.content, "parts", []):
                    v = getattr(part, "text", None)
                    if v:
                        parts.append(v)
            text = "\n".join(parts)

        if not text:
            raise RuntimeError("Gemini returned an empty answer.")

        answer = text.strip()

        if debug:
            return {
                "answer": answer,
                "sources": [s.dict() for s in contexts],
                "debug": {
                    "formatter": "gemini",
                    "model": self._model_name,
                    "prompt": composed_prompt,
                    "generation_config": self._generation_config or {},
                },
            }

        return answer

def build_formatter(settings: Settings) -> AnswerFormatterProtocol:
    """Return the formatter strategy indicated by configuration."""
    if settings.formatter == "langchain":
        return LangChainAgentFormatter.from_settings(settings)
    if settings.formatter == "gemini":
        return GeminiAnswerFormatter.from_settings(settings)
    return MockAnswerFormatter()


app = FastAPI(title="Insurance Chatbot API", version="0.2.0")
settings = get_settings()
formatter = build_formatter(settings)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")
    latest_message = request.messages[-1]
    if latest_message.role != "user":
        raise HTTPException(
            status_code=400,
            detail="The latest message in the conversation must come from the user.",
        )

    retrieved_sources: List[Source] = []

    try:
        formatted_result = await formatter.format_answer(
            request.messages,
            retrieved_sources,
            top_k=request.top_k,
            enable_web_search=request.enable_web_search,
            debug=request.debug,
            language=request.language,
        )
    except RuntimeError as exc:
        logger.error("Formatter failed in /chat: %s", exc, exc_info=True)
        debug_payload = {"error": str(exc)} if request.debug else None
        return ChatResponse(
            answer="Error: The response could not be generated at this time.",
            sources=[],
            usage={
                "retrieved_documents": 0,
                "formatter": settings.formatter,
                "web_search_enabled": request.enable_web_search,
                "language": request.language,
                "top_k": request.top_k,
                "debug_enabled": request.debug,
            },
            debug=debug_payload,
        )
    except Exception as exc:
        logger.exception("Unexpected error in /chat")
        debug_payload = {"error": str(exc)} if request.debug else None
        return ChatResponse(
            answer="Error: An unexpected problem occurred on the server.",
            sources=[],
            usage={
                "retrieved_documents": 0,
                "formatter": settings.formatter,
                "web_search_enabled": request.enable_web_search,
                "language": request.language,
                "top_k": request.top_k,
                "debug_enabled": request.debug,
            },
            debug=debug_payload,
        )

    if isinstance(formatted_result, dict):
        answer = formatted_result.get(
            "answer", "Error: The model response did not contain 'answer'."
        )
        final_sources_data = formatted_result.get("sources", [])
        final_sources: List[Source] = [
            Source(**s) if isinstance(s, dict) else s for s in final_sources_data
        ]

        usage = formatted_result.get("usage", {}) or {}
        usage.setdefault("retrieved_documents", len(final_sources))
        usage.setdefault("web_search_enabled", request.enable_web_search)
        usage.setdefault("formatter", settings.formatter)
        usage.setdefault("language", request.language)
        usage.setdefault("top_k", request.top_k)
        usage.setdefault("debug_enabled", request.debug)

        debug_payload: Optional[dict] = None
        if request.debug:
            debug_payload = formatted_result.get("debug") or {}

        return ChatResponse(
            answer=answer,
            sources=final_sources,
            usage=usage,
            debug=debug_payload,
        )

    usage = {
        "retrieved_documents": len(retrieved_sources),
        "web_search_enabled": request.enable_web_search,
        "formatter": settings.formatter,
        "language": request.language,
        "top_k": request.top_k,
    }
    debug_payload = {} if request.debug else None
    return ChatResponse(
        answer=str(formatted_result),
        sources=retrieved_sources,
        usage=usage,
        debug=debug_payload,
    )


@app.get("/chat", include_in_schema=False)
def chat_get_landing() -> Dict[str, Any]:
    """Helpful response for accidental GET requests to /chat."""
    return {
        "message": "Use POST /chat with a ChatRequest payload to talk to the bot.",
        "example_request": {
            "messages": [{"role": "user", "content": "¿Qué cubre la póliza?"}],
            "top_k": 4,
            "enable_web_search": False,
            "debug": False,
            "language": "es",
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