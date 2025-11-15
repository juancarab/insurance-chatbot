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
        return {"error": "Timeout: The server took too long to respond."}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to the backend server"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}