# Insurance Chatbot

Interfaz demo y API para el proyecto *Insurance Chatbot*. Este repositorio incluye únicamente:

- Backend FastAPI con endpoint `/chat` y mocks para el recuperador, el formateador LLM y la búsqueda web.
- Frontend Streamlit con una UI de chat que consume la API.
- Dependencias mínimas y guía para ejecutar todo localmente.

## Requisitos previos

- Python 3.10+
- (Opcional) Docker y Docker Compose v2

## Instalación y ejecución manual

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Backend (FastAPI)

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- Documentación interactiva: <http://localhost:8000/docs>
- Healthcheck: <http://localhost:8000/health>

### Frontend (Streamlit)

En otra terminal con el entorno virtual activado:

```bash
streamlit run frontend/app.py --server.port 8501
```

La aplicación abrirá en <http://localhost:8501>. En la barra lateral podés ajustar el número de snippets a recuperar, habilitar la búsqueda web (mock) y configurar la URL del backend. Si querés definir una URL por defecto distinta antes de iniciar Streamlit, exportá `INSURANCE_CHATBOT_API_URL` (por ejemplo, `export INSURANCE_CHATBOT_API_URL=http://localhost:8000/chat`).

## Contrato del endpoint `/chat`

### Request

```json
{
  "messages": [
    {"role": "user", "content": "Último mensaje del usuario"},
    {"role": "assistant", "content": "Mensajes previos opcionales"}
  ],
  "top_k": 3,
  "enable_web_search": false,
  "metadata": {"client": "web"}
}
```

- `messages`: historial completo ordenado (último mensaje al final). El último **debe** ser del usuario.
- `top_k`: máximo 10. Indica cuántos fragmentos recuperar.
- `enable_web_search`: activa un stub de búsqueda web (sin implementación real).
- `metadata`: campo libre para futuras integraciones.

### Response

```json
{
  "answer": "Texto generado por el formateador (mock).",
  "sources": [
    {
      "id": "policy-1",
      "title": "Mock Policy Document 1",
      "snippet": "Relevant excerpt...",
      "url": "https://example.com/policies/1"
    }
  ],
  "usage": {
    "retrieved_documents": 3,
    "web_search_enabled": false
  }
}
```

- `answer`: respuesta del stub `AnswerFormatter`.
- `sources`: lista de objetos `Source`. Sirve de contrato para integrar el recuperador real.
- `usage`: métricas o diagnósticos (libre).

## Docker Compose (opcional)

```bash
docker compose up --build
```

El `docker-compose.yml` ya define `INSURANCE_CHATBOT_API_URL=http://backend:8000/chat` para que el frontend se comunique con el backend dentro de la red interna del stack. Externamente tendrás:

- Backend disponible en <http://localhost:8000>
- Frontend disponible en <http://localhost:8501>

## Próximos pasos (no incluidos)

- Reemplazar `PolicyRetriever.search` con integración real a los 13 PDFs.
- Conectar `AnswerFormatter.format_answer` a un LLM.
- Implementar búsqueda web real en `WebSearchClient`.
