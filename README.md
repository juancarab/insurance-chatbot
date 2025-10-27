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

## Configuración de variables de entorno

- El backend lee toda su configuración desde `backend/app/config.py`, un módulo basado en `pydantic.BaseSettings` que también carga automáticamente un archivo `.env` si está presente en la raíz del repositorio.
- Copiá `.env.example` a `.env` y completá las variables según la integración que quieras probar (`mock`, `langchain` o `gemini`).
- Los mismos valores funcionan tanto para ejecuciones locales como para Docker Compose, evitando tener que exportar manualmente cada variable en la terminal.

## Contrato del endpoint `/chat`

### Opensearch & Tavily

  ## APIKEY de tavily:
  https://app.tavily.com/home


  ## levantar los servicios de docker
  ```bash
  docker compose up -d --build
  ```

  ## Crear el indice hibrido
  ```bash
  docker compose exec backend bash -lc "python infra/opensearch/setup_opensearch.py"
  ```

  ## para ejecutar los test
  ```bash
  python -m pytest -q
  ```

  ## .env 
  OPENSEARCH_HOST=opensearch
  OPENSEARCH_PORT=9200
  OPENSEARCH_INDEX=policies
  OPENSEARCH_EMBED_DIM=384
  TAVILY_API_KEY=apikey
  WEB_SEARCH_MAX_RESULTS=5
  WEB_SEARCH_FRESHNESS_DAYS=30


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

- `answer`: respuesta del stub `MockAnswerFormatter` (o del agente si está configurado).
- `sources`: lista de objetos `Source`. Sirve de contrato para integrar el recuperador real.
- `usage`: métricas o diagnósticos (libre).

## Docker Compose (opcional)

```bash
docker compose up --build
```

El `docker-compose.yml` ya define `INSURANCE_CHATBOT_API_URL=http://backend:8000/chat` para que el frontend se comunique con el backend dentro de la red interna del stack. Externamente tendrás:

- Backend disponible en <http://localhost:8000>
- Frontend disponible en <http://localhost:8501>

## Integración directa con Google Gemini

El backend incluye un formateador dedicado para Gemini de modo que puedas hacer una
demo real solo con tu API key y el modelo que quieras utilizar.

1. **Instalá las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

2. **Exportá tus credenciales y modelo de Gemini**
   ```bash
   export INSURANCE_CHATBOT_FORMATTER=gemini
   export GEMINI_API_KEY="tu_api_key_de_gemini"
   export GEMINI_MODEL="gemini-1.5-flash-latest"  # o el modelo que prefieras

   # Opcionales para tunear la generación
   export GEMINI_TEMPERATURE=0.2
   export GEMINI_TOP_P=0.95
   export GEMINI_MAX_OUTPUT_TOKENS=1024
   ```

   Si alguna de estas variables falta o tiene un formato inválido, el backend emitirá
   un error claro al iniciar para evitar respuestas inesperadas.

3. **Levantá el backend** (`uvicorn backend.app.main:app ...`). Con esas variables, el
   formateador construirá un prompt con el historial del chat y los snippets
   recuperados, lo enviará a Gemini y devolverá la respuesta al frontend usando el
   mismo contrato JSON.

## Integración con un agente LangChain

El backend permite sustituir el formateador mock por cualquier agente o runnable de
LangChain sin modificar el endpoint. Solo hay que exponer una función que reciba el
historial y los snippets recuperados, y configurar dos variables de entorno.

1. **Instalá las dependencias del agente** (ejemplo):
   ```bash
   pip install langchain langchain-openai google-generativeai  # ajustá según tu stack
   ```

2. **Implementá un runner de LangChain**. Guardalo, por ejemplo, en
   `backend/app/langchain_runner.py`:

   ```python
   import os
   from typing import Any, Dict, List

   from langchain.agents import AgentType, initialize_agent
   from langchain.chat_models import ChatOpenAI


   def run_langchain_agent(*, messages: List[Dict[str, Any]], contexts: List[Dict[str, Any]]) -> Dict[str, Any] | str:
       """Construye el prompt con los snippets y delega la respuesta al agente."""

       question = messages[-1]["content"]
       context_block = "\n\n".join(
           f"{item['title']}: {item['snippet']}" for item in contexts
       ) or "No se recuperaron fragmentos relevantes."

       prompt = (
           "Sos un asesor de pólizas. Usa el contexto para responder con precisión.\n"
           f"Pregunta: {question}\n\nContexto:\n{context_block}"
       )

       llm = ChatOpenAI(
           model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
           temperature=0,
       )
       agent = initialize_agent(
           tools=[],
           llm=llm,
           agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
           verbose=False,
       )

       # Puede devolver un string o un diccionario con clave "output".
       return agent.invoke({"input": prompt})
   ```

   El adaptador `LangChainAgentFormatter` convierte `messages` y `contexts` en
   diccionarios serializables antes de invocar tu función. Si el agente devuelve un
   `dict`, extraerá automáticamente la clave `output`/`answer`/`result` para poblar el
   campo `answer` de la API.

3. **Configura las variables de entorno antes de iniciar el backend**:

   ```bash
   export INSURANCE_CHATBOT_FORMATTER=langchain
   export INSURANCE_CHATBOT_LANGCHAIN_RUNNER=backend.app.langchain_runner:run_langchain_agent
   ```

   Si falta alguna variable o la ruta no existe, el servicio no arrancará y mostrará
   un error claro en consola. Esto evita arrancar el servidor con un agente incompleto.

4. **Levanta el backend** (`uvicorn backend.app.main:app ...`). El endpoint `/chat` seguirá
   aceptando el mismo contrato; solo cambiará la lógica con la que se construye la
   respuesta.

## Próximos pasos (no incluidos)

- Reemplazar `PolicyRetriever.search` con integración real a los 13 PDFs.
- Ajustar `GeminiAnswerFormatter` o conectar tu formateador definitivo según el stack final.
- Implementar búsqueda web real en `WebSearchClient`.
