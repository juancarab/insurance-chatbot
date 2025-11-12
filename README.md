# Insurance Chatbot

Interfaz **demo** y **API** para el proyecto *Insurance Chatbot*.

Este repositorio incluye:

- **Backend (FastAPI)** con el endpoint `/chat`.
- **Frontend (Streamlit)** con una UI de chat que consume la API.
- **Agent** con herramientas de **búsqueda web (Tavily)** y **retrieval en OpenSearch** (vía Haystack).
- **Pipelines de datos** (EDA e **ingesta a OpenSearch**) y utilidades de **setup del índice**.

> Monorepo: `services/backend`, `services/frontend`, `services/agent`, y carpeta `data/` para PDFs, ingesta y setup.

---

## Requisitos

- **Python 3.10+**
- **Docker** y **Docker Compose v2** (recomendado para levantar todo el stack)
- (Opcional, para búsqueda web real) Cuenta y **API Key de Tavily**: <https://app.tavily.com/home>

---

## Estructura relevante

```
.
├── docker-compose.yml
├── services/
│   ├── backend/
│   │   ├── app/ (FastAPI: main.py, config.py)
│   │   └── Dockerfile
│   ├── frontend/
│   │   ├── app.py (Streamlit)
│   │   └── Dockerfile
│   └── agent/
│       └── app/
│           ├── langchain_runner.py
│           └── tools/
│               ├── retrieval/haystack_opensearch_tool.py
│               └── web_search/web_search.py
├── data/
│   ├── raw_policies/ (PDFs)
│   ├── pipeline/ (ingest.py, eda_policies.py)
│   ├── opensearch/setup_opensearch.py
│   └── test/test_opensearch_setup.py
├── tests/
└── web_search_cli.py
```

---

## Variables de entorno

Crea un archivo **`.env`** en la raíz (puedes partir de `.env.example` si existe). Ejemplo mínimo:

```env
# --- OpenSearch ---
# Docker Compose: host 'opensearch' | Ejecución local: 'http://localhost:9200'
OPENSEARCH_HOST=http://localhost:9200
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=policies
OPENSEARCH_EMBED_DIM=384

# --- Búsqueda web (Tavily) ---
TAVILY_API_KEY=tu_api_key
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_FRESHNESS_DAYS=30

# --- Selección del "formatter" del backend ---
# mock | gemini | langchain
INSURANCE_CHATBOT_FORMATTER=mock

# --- Gemini (si usas INSURANCE_CHATBOT_FORMATTER=gemini) ---
GEMINI_API_KEY=tu_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2
GEMINI_TOP_P=0.95
GEMINI_MAX_OUTPUT_TOKENS=1024

# --- LangChain (si usas INSURANCE_CHATBOT_FORMATTER=langchain) ---
# Con PYTHONPATH=services, el runner vive en services.agent.app.langchain_runner
INSURANCE_CHATBOT_LANGCHAIN_RUNNER=services.agent.app.langchain_runner:run_langchain_agent
```

> **Nota**: si levantas el stack con Docker Compose, cambia `OPENSEARCH_HOST` a `opensearch` (el nombre del servicio); para ejecución local directa, deja `http://localhost:9200`.

---

## Inicio rápido (Docker Compose — recomendado)

1) **Levanta el stack**:
    ```bash
    docker compose up -d --build
    ```

2) **Crea el índice híbrido** de OpenSearch:
    ```bash
    # Si el proyecto está montado dentro del contenedor backend (lo usual):
    docker compose exec backend bash -lc "python data/opensearch/setup_opensearch.py"
    
    # Alternativa (desde tu host, con deps de 'data/requirements.txt'):
    #   pip install -r data/requirements.txt
    #   python data/opensearch/setup_opensearch.py
    ```

3) **Ingesta de PDFs** a OpenSearch (opcional pero recomendado):
    ```bash
    docker compose exec backend bash -lc "python data/pipeline/ingest.py"
    ```

4) **Acceso**:
    - **Backend**: <http://localhost:8000>  
      - Docs: <http://localhost:8000/docs>  
      - Health: <http://localhost:8000/health>
    - **Frontend**: <http://localhost:8501>
    ---

## Ejecución local (sin Compose)

1) **Crear entorno**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    ```

2) **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```

3) **Arrancar OpenSearch** (usa Docker si no tienes un cluster local):
    ```bash
    docker compose up -d opensearch
    ```

4) **Crear índice e ingerir pólizas** (los PDFs deben estar en `data/raw_policies/`):
    ```bash
    python data/opensearch/setup_opensearch.py
    python data/pipeline/ingest.py
    ```

5) **Backend** (desde la raíz del repo):
    ```bash
    PYTHONPATH=services uvicorn services.backend.app.main:app --reload --host 0.0.0.0 --port 8001
    ```

6) **Frontend** (otra terminal):
    ```bash
    export INSURANCE_CHATBOT_API_URL=http://127.0.0.1:8001/chat
    streamlit run services/frontend/app.py --server.port 8501
    ```
    - Activa la casilla “Modo debug” en el sidebar para ver los pasos del agente (`debug.steps`) y las fuentes recuperadas en los expanders correspondientes.

---

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

- `messages`: historial ordenado; el **último** debe ser del **usuario**.
- `top_k`: máximo **10**, cantidad de fragmentos a recuperar.
- `enable_web_search`: si es `true` y hay `TAVILY_API_KEY`, habilita búsqueda web real (Tavily); de lo contrario, se usa stub.
- `metadata`: libre.

### Response
```json
{
  "answer": "Texto generado por el formatter (mock/gemini/langchain).",
  "sources": [
    {
      "id": "policy-1",
      "title": "Documento de Póliza",
      "snippet": "Extracto relevante...",
      "url": "https://example.com/policies/1"
    }
  ],
  "usage": {
    "retrieved_documents": 3,
    "web_search_enabled": false,
    "formatter": "langchain"
  }
}
```

- `answer`: texto final (mock/Gemini/tu agente LangChain).
- `sources`: contrato para trazabilidad/justificación.
- `usage`: métricas diagnósticas (puedes ampliarlo).

### Ejemplo con `debug=true`

```bash
curl -s -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [
          {"role": "user", "content": "¿Qué cubre la póliza de hogar básica?"}
        ],
        "top_k": 3,
        "enable_web_search": false,
        "debug": true,
        "language": "es"
      }'
```

Respuesta (ejemplo con formatter `langchain`/`mock`):

```json
{
  "answer": "(mock) Respondiendo en es. Cuando el LLM esté integrado...",
  "sources": [],
  "usage": {
    "retrieved_documents": 0,
    "web_search_enabled": false,
    "formatter": "langchain",
    "language": "es",
    "top_k": 3,
    "debug_enabled": true
  },
  "debug": {
    "formatter": "mock",
    "messages": [
      {"role": "user", "content": "¿Qué cubre la póliza de hogar básica?"}
    ],
    "contexts": [],
    "top_k": 3,
    "enable_web_search": false,
    "language": "es"
  }
}
```

> Con un agente LangChain real, el bloque `debug` incluye los pasos del grafo, chunks recuperados, herramientas invocadas, etc.

---

## Selección de formateador (mock / Gemini / LangChain)

El backend decide según `INSURANCE_CHATBOT_FORMATTER`:

### 1) Mock (por defecto)
```env
INSURANCE_CHATBOT_FORMATTER=mock
```
Responde sin llamar a modelos externos.

### 2) Gemini
```env
INSURANCE_CHATBOT_FORMATTER=gemini
GEMINI_API_KEY=tu_api_key
GEMINI_MODEL=gemini-2.5-flash
```

### 3) LangChain + Tools (retrieval + web search)
```env
INSURANCE_CHATBOT_FORMATTER=langchain
INSURANCE_CHATBOT_LANGCHAIN_RUNNER=services.agent.app.langchain_runner:run_langchain_agent
TAVILY_API_KEY=tu_api_key             # para web search real
OPENSEARCH_HOST=opensearch|localhost  # según tu modo
```

El runner vive en `services/agent/app/langchain_runner.py` y puede invocar:
- **Retrieval** híbrido (BM25 + embeddings) en OpenSearch:  
  `services/agent/app/tools/retrieval/haystack_opensearch_tool.py`
- **Web search** (Tavily):  
  `services/agent/app/tools/web_search/web_search.py`

---

## OpenSearch (setup e ingesta)

1) **Crear índice híbrido**:
    ```bash
    # Docker:
    docker compose exec backend bash -lc "python data/opensearch/setup_opensearch.py"
    
    # Local:
    python data/opensearch/setup_opensearch.py
    ```

2) **Ingestar PDFs** (`data/raw_policies/`):
    ```bash
    python data/pipeline/ingest.py
    ```

> La ingesta usa **sentence-transformers/all-MiniLM-L6-v2** (dim=**384**) ⇒ debe coincidir con `OPENSEARCH_EMBED_DIM=384`.

---

## Frontend (Streamlit)

- Archivo: `services/frontend/app.py`
- Ejecuta la UI de chat en <http://localhost:8501>.
- Configurable vía env: `INSURANCE_CHATBOT_API_URL` (por defecto apunta al backend local o al servicio `backend` en Docker).

---

## Utilidades y pruebas

- **CLI de búsqueda web** (útil para depurar Tavily):
  ```bash
  python web_search_cli.py --q "qué cubre hospitalización" --k 5
  ```
- **Pruebas** (setup de OpenSearch, web search, etc.):
  ```bash
  pytest -q
  # o:
  python -m pytest -q
  ```

### Golden Set (benchmark de respuestas)

1. Asegúrate de tener el backend levantado en `http://127.0.0.1:8001` (o ajusta `--base-url`).
2. Ejecuta el script de evaluación:
   ```bash
   python scripts/run_golden_set.py \
     --base-url http://127.0.0.1:8001/chat \
     --golden-set data/golden_set/golden_set.json \
     --output results/golden_before.jsonl
   ```
   - El archivo `data/golden_set/golden_set.json` incluye 35 escenarios distribuidos en cinco categorías: `simple`, `follow_up`, `web`, `combined`, `negative`.
   - El script crea una fila por pregunta con la respuesta del agente, tiempos y metadatos; por defecto guarda la corrida en `results/golden_<timestamp>.jsonl`.
3. Implementa tus cambios (por ejemplo, Tarea 1 o 2) y vuelve a ejecutar el script para generar `results/golden_after.jsonl`.
4. El script también produce un archivo `*.metrics.json` con métricas RAGAS (`faithfulness` y `context_precision`). Usa `--baseline-metrics` para comparar contra una corrida anterior:
   ```bash
   python scripts/run_golden_set.py \
     --base-url http://127.0.0.1:8001/chat \
     --baseline-metrics results/golden_before.metrics.json
   ```
5. Por defecto el script usa `gemini-2.5-flash`. Asegúrate de exportar `GEMINI_API_KEY` (o pasa `--gemini-api-key`):
   ```bash
   python scripts/run_golden_set.py \
     --base-url http://127.0.0.1:8001/chat \
     --ragas-model gemini-2.5-flash \
     --gemini-api-key "$GEMINI_API_KEY"
   ```
6. Completa el campo `reference_answer` en `data/golden_set/golden_set.json` para tener contexto adicional al revisar resultados (aunque actualmente solo se calculan `faithfulness` y `context_precision`).

---

## Solución de problemas

- **OpenSearch “unhealthy” o sin índice**  
  Verifica puertos `9200/9600`. Corre `setup_opensearch.py` y revisa logs:
  ```bash
  docker compose logs -f opensearch
  ```
- **El backend no encuentra módulos `backend.*`**  
  Ejecuta con `PYTHONPATH=services` (ver comandos arriba).
- **Web search no retorna resultados**  
  Asegura `TAVILY_API_KEY` y `enable_web_search=true` en la request; verifica que el formatter/runner invoque la tool.
- **Dimensión de embeddings inconsistente**  
  Si cambias el modelo de embeddings, actualiza `OPENSEARCH_EMBED_DIM` y reindexa.

---

## Roadmap breve

- Integrar retrieval real por defecto en `/chat`.
- Afinar prompts del formatter (Gemini/LangChain).
- Mejorar trazabilidad de `sources` y diagnósticos en `usage`.


## Router multi-índice

- **Activar venv 3.9 e instalar
& .\.venv\Scripts\Activate.ps1
python --version   # debe decir 3.9.x
pip install -r requirements.txt
pip install eval-type-backport==0.2.2 importlib-metadata==6.8.0 "zipp>=3.15"

- **OpenSearch
docker compose up -d opensearch
curl.exe -s http://localhost:9200 | Out-String

- **Ingesta PDFs → índice `policies`
$env:OPENSEARCH_HOST = "http://localhost:9200"
python .\data\pipeline\ingest.py
curl.exe -s http://localhost:9200/policies/_count | Out-String  # debe ser > 0

- **Probar localizador de pólizas (Etapa 1)
$env:OPENSEARCH_HOST = "http://localhost:9200"
python -m services.agent.app.tools.find_relevant_policies "coaseguro en el extranjero"

- **Backend (elige uno)
$env:PYTHONPATH = "services"

- **LangChain (usa el retriever real)
$env:INSURANCE_CHATBOT_FORMATTER = "langchain"
$env:INSURANCE_CHATBOT_LANGCHAIN_RUNNER = "services.agent.app.langchain_runner:run_langchain_agent"
uvicorn services.backend.app.main:app --reload --host 127.0.0.1 --port 8001

- **Consultas de prueba
Invoke-RestMethod http://127.0.0.1:8001/health
'{"messages":[{"role":"user","content":"¿Cuál es el deducible anual de hospitalización?"}],"top_k":3,"enable_web_search":false,"debug":true,"language":"es"}' |
  Set-Content -Path req1.json -Encoding utf8 -NoNewline
curl.exe -s -X POST "http://127.0.0.1:8001/chat" -H "Content-Type: application/json; charset=utf-8" --data-binary "@req1.json"

