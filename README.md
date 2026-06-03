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

## ⏱️ Setup and Run

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
Open [http://localhost:8000/](http://localhost:8000/) in your browser to interact with the search web client interface.

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
