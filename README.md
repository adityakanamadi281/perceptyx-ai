# Perplexity Agent

Agentic web search, reasoning, and fact-checking system powered by **LangGraph**, **LangChain**, and **Gemini**.

---

## 🚀 Features

* **🗺️ Multi-Agent Planning**: Decomposes complex user queries into optimal parallel search sub-queries.
* **🔍 Parallel Web Search & Scraping**: Performs lightning-fast concurrent searches via Serper API and scrapes full page contents using headless **Playwright** browsers.
* **🧠 Chain-of-Thought Reasoning**: Evaluates and cross-checks information across multiple sources to resolve contradictions and extract key facts.
* **📝 Source Synthesis & Citation**: Generates a unified response with structured, traceable, numbered citations.
* **📊 Deep Observability**: Logs structured telemetry (latency, token usage) for every agent step, with optional integration for **LangSmith** and **OpenTelemetry**.

---

## 🛠️ Tech Stack

* **Core Frameworks**: [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/) (Async API layer), [LangGraph](https://www.langchain.com/langgraph) & [LangChain](https://www.langchain.com/) (Agentic orchestration & chains)
* **LLM Engine**: Google [Gemini API](https://ai.google.dev/) (`gemini-1.5-flash` by default)
* **Web Scraping**: [Playwright](https://playwright.dev/python/) (Chromium backend), [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/), [Readability-lxml](https://github.com/buriy/python-readability)
* **Environment & Package Management**: [uv](https://github.com/astral-sh/uv) (Extremely fast Rust-based dependency resolver and workspace manager)
* **Quality Assurance**: [pytest](https://docs.pytest.org/), [ruff](https://github.com/astral-sh/ruff) (formatting & linting), [mypy](https://mypy-lang.org/) (type checking)
* **Observability**: [structlog](https://www.structlog.org/), [OpenTelemetry SDK](https://opentelemetry.io/), [LangSmith](https://smith.langchain.com/)

---

## 🏗️ Architecture

```
POST /api/v1/query
        │
        ▼
    Planner  ──► [sub-query 1, sub-query 2, …, sub-query N]
        │
        ▼  asyncio.gather (parallel)
┌───────────────────────────────────────┐
│  Search agent  →  Reason agent  × N  │
│  (Serper + Playwright scraper)        │
│  (Gemini chain-of-thought)            │
└───────────────────────────────────────┘
        │
        ▼
  Answer agent  ──► Synthesise + cite sources
        │
        ▼
  AnswerResponse (JSON with citations)
```

**Observability** is wired throughout via a custom `TelemetryCallback` that plugs into every LangChain LLM invocation, recording latency and token usage into a unified `PipelineTrace`.

---

## ⏱️ Setup and Run in 5 Minutes

### 1. Install `uv`
If you haven't installed Astral's `uv` yet:
```bash
# macOS / Linux
curl -Lsf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. Clone and Synchronize Workspace
Clone the repository and automatically prepare your virtual environment and dependencies using `uv sync`:
```bash
git clone https://github.com/adityakanamadi281/perplexity-agent.git 
cd perplexity_agent
uv sync --extra dev
uv run playwright install chromium
```

### 3. Configure Environment Variables
Copy the example environment file and configure your API keys:
```bash
cp .env.example .env
```
Open `.env` and set:
* `GEMINI_API_KEY`: Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
* `SERPER_API_KEY`: Get one from [Serper.dev](https://serper.dev) for web search results.

### 4. Run the Development Server
Launch the FastAPI application:
```bash
uv run uvicorn api.main:app --reload --port 8000
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to interact with the API explorer.

### 5. Run Verification Tasks
Ensure everything compiles, formats, and passes tests:
```bash
# Run pytest test suite
uv run pytest -v

# Format code with ruff
uv run ruff format
```

---

## 📂 Project Structure

```
perplexity_agent/
├── config/settings.py        # Pydantic-Settings, env vars
├── core/
│   ├── planner.py            # Query → sub-queries via Gemini
│   ├── orchestrator.py       # LangGraph state machine
│   └── observability.py      # structlog, TelemetryCallback, OTEL
├── agents/
│   ├── search.py             # Serper search + Playwright scraper
│   ├── reason.py             # Gemini chain-of-thought reasoning
│   └── answer.py             # Synthesis + citation
├── providers/gemini.py       # Cached ChatGoogleGenerativeAI
├── tools/
│   ├── serper.py             # Serper API wrapper
│   └── scraper.py            # Playwright / httpx / readability
├── models/schemas.py         # All Pydantic v2 models
├── api/
│   ├── main.py               # FastAPI app factory
│   └── query.py              # POST /query router
├── tests/test_pipeline.py    # Integration tests
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🔌 API Reference

### `POST /api/v1/query`

**Request:**
```json
{
  "query": "What are the latest advances in quantum computing?",
  "max_sources": 5,
  "locale": "en"
}
```

**Response:**
```json
{
  "run_id": "9d439c52-ad6c-480d-87bd-5dc241946622",
  "query": "...",
  "answer": "Prose answer here...",
  "citations": [
    {
      "index": 1,
      "title": "Source title",
      "url": "https://...",
      "relevant_snippet": "..."
    }
  ],
  "total_tokens": 2840,
  "latency_ms": 18500.3,
  "created_at": "2026-06-04T10:00:00Z"
}
```

### `GET /health`
Returns system status and API version:
```json
{"status": "ok", "version": "0.1.0"}
```

---

## 📊 Observability Details

Every request emits structured JSON logs via `structlog` containing:
* `run_id`: UUID correlating all logs for a given execution trace.
* `agent`: The active component (e.g., `planner`, `reason_agent`, `answer_agent`).
* `latency_ms`: Response duration.
* `tokens`: Detailed input/output token usage.

### LangSmith Tracing (Optional)
To enable real-time visual traces of execution graphs on [LangSmith](https://smith.langchain.com/), configure these environment variables in your `.env`:
```env
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=perplexity-agent
```

### OpenTelemetry (Optional)
Export tracing spans to any OTLP-compliant backend (Jaeger, Datadog, Grafana Tempo) by setting:
```env
ENABLE_OTEL=true
OTEL_ENDPOINT=http://your-otel-collector:4317
```

---

## ⚙️ Configuration Reference

| Environment Variable | Default Value | Description |
|---|---|---|
| `GEMINI_API_KEY` | *Required* | API key for Google Gemini |
| `SERPER_API_KEY` | *Required* | API key for Serper.dev |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model selection |
| `MAX_SUB_QUERIES` | `4` | Max concurrent sub-queries generated by Planner |
| `MAX_SEARCH_RESULTS` | `5` | Search results scraped per sub-query |
| `SCRAPE_TIMEOUT_S` | `15` | Scraping timeout per URL (in seconds) |
| `PIPELINE_TIMEOUT_S` | `120` | Hard timeout for the full reasoning pipeline |
| `LOG_LEVEL` | `INFO` | Structlog level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ENABLE_OTEL` | `false` | Enable/disable OpenTelemetry export |
| `LANGSMITH_API_KEY` | `None` | LangSmith API Token |
