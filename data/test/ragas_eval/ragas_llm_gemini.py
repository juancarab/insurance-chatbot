from __future__ import annotations

import asyncio
import json
from typing import List, Optional

import requests
from ragas.llms.base import BaseRagasLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue


class GeminiRagasLLM(BaseRagasLLM):
    """
    Wrapper mínimo para usar Gemini con RAGAS.

    - Fuerza salida JSON desde Gemini.
    - Limpia bloques ```...``` si el modelo los mete.
    - Intenta parsear el resultado como JSON.
    - Si está truncado o es inválido, devuelve un JSON válido de emergencia
      para que RAGAS no truene.
    """

    def __init__(self, model: str, api_key: str, debug: bool = False):
        super().__init__()

        # Gemini suele ir con "models/..."
        if not model.startswith("models/"):
            model = f"models/{model}"
        self.model_name = model
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:generateContent"
        )
        self.api_key = api_key
        self.debug = debug

        # Que RAGAS no paralelice mucho, porque Gemini lo estamos haciendo sync
        self.run_config.timeout = 180
        self.run_config.max_workers = 1
        self.run_config.max_retries = 1

    # ------------------------------------------------------------------
    # Normalización y robustez del JSON
    # ------------------------------------------------------------------
    def _normalize_llm_output(self, text: str) -> str:
        """
        Deja el texto en el mejor JSON posible.

        Pasos:
        1. strip
        2. quitar ```...``` si los hay
        3. si hay un { o [ más adelante, recortar desde ahí
        4. intentar json.loads(...)
           - si funciona, lo dejamos así
           - si NO, devolvemos un JSON seguro ({} o [])
        """
        text = text.strip()

        # 1) quitar bloques ```...``` típicos
        if text.startswith("```"):
            parts = text.split("```")
            # buscamos desde el final porque muchas veces el JSON está al final
            for part in reversed(parts):
                part = part.strip()
                if part.startswith("{") or part.startswith("["):
                    text = part
                    break

        # 2) si todavía no empieza con { o [, intentamos recortar desde el primer { o [
        if not (text.startswith("{") or text.startswith("[")):
            start_obj = text.find("{")
            start_arr = text.find("[")
            starts = [x for x in (start_obj, start_arr) if x != -1]
            if starts:
                text = text[min(starts):].strip()

        # 3) ahora probamos realmente si es JSON válido
        if text:
            try:
                # si esto no lanza, ya estamos
                json.loads(text)
                return text
            except Exception:
                # seguimos al fallback
                pass

        # 4) Fallbacks super defensivos
        # si parecía lista pero quedó cortada: devolvemos lista vacía
        if text.startswith("["):
            return "[]"

        # si parecía objeto pero quedó cortado: devolvemos un objeto válido
        if text.startswith("{"):
            return json.dumps(
                {
                    "reason": "LLM devolvió JSON inválido o truncado.",
                    "verdict": 0,
                }
            )

        # si no parecía nada, devolvemos un objeto válido básico
        return json.dumps(
            {
                "reason": text or "LLM devolvió salida vacía o no JSON.",
                "verdict": 0,
            }
        )

    # ------------------------------------------------------------------
    # Llamada a Gemini
    # ------------------------------------------------------------------
    def _call_gemini(self, prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json",
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        resp = requests.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            json=payload,
            timeout=120,
        )
        if not resp.ok:
            raise RuntimeError(f"Gemini falló: {resp.status_code} {resp.text}")

        data = resp.json()
        parts: List[str] = []

        for cand in data.get("candidates", []):
            content = cand.get("content") or {}
            for part in content.get("parts", []):
                t = part.get("text")
                if t:
                    parts.append(t)

        text_out = "\n".join(parts) if parts else ""
        if self.debug:
            print("\n--- LLM RAW OUTPUT (text) ---")
            print(text_out)
            print("--- END LLM RAW OUTPUT (text) ---\n")
        return text_out

    # ------------------------------------------------------------------
    # Métodos que RAGAS espera
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        raw = self._call_gemini(prompt.to_string(), temperature)
        cleaned = self._normalize_llm_output(raw)
        return LLMResult(generations=[[Generation(text=cleaned)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        raw = await asyncio.to_thread(self._call_gemini, prompt.to_string(), temperature)
        cleaned = self._normalize_llm_output(raw)
        return LLMResult(generations=[[Generation(text=cleaned)]])


def build_ragas_llm(model_name: str, api_key: str, debug_llm: bool = False) -> BaseRagasLLM:
    return GeminiRagasLLM(model=model_name, api_key=api_key, debug=debug_llm)