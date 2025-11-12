import os
import requests
from typing import Any, Dict, List

DEFAULT_API_URL = os.getenv("INSURANCE_CHATBOT_API_URL", "http://localhost:8000/chat")

def check_api_status(api_url: str) -> bool:
    try:
        response = requests.get(f"{api_url.replace('/chat', '')}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def call_backend_api(
    api_url: str,
    messages: List[Dict[str, str]],
    top_k: int,
    enable_web_search: bool,
    debug: bool,
    language: str,
) -> Dict[str, Any]:
    payload = {
        "messages": messages,
        "top_k": top_k,
        "enable_web_search": enable_web_search,
        "debug": debug,
        "language": language,
    }
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Timeout: El servidor tardó demasiado en responder"}
    except requests.exceptions.ConnectionError:
        return {"error": "No se pudo conectar al servidor backend"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Error HTTP: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}