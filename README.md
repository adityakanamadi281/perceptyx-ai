# PerceptyxAI

A production-grade, Perplexity-class AI search engine built for high-performance and deep multi-source research.

---

## 🚀 Features

- **Intelligent Query Complexity Routing**: Dynamically classifies user queries (SIMPLE, MEDIUM, COMPLEX, or RESEARCH) to optimize pathing, latency, and resource utilization.
- **Parallel Multi-Source Retrieval**: Searches simultaneously across Google (Serper), DuckDuckGo (fallback), local vector stores, news sources, and GitHub.
- **Advanced Hybrid RAG**: Uses a hybrid search engine (Dense Vector Embeddings + BM25 Lexical Search) with Reciprocal Rank Fusion (RRF) and BGE Cross-Encoder Reranking (mapping top-50 candidates to top-10).
- **Multi-Level Caching**: Decreases latency using a multi-tiered cache layout (L1 in-memory, L2 Redis with distributed locks/pubsub, L3 PostgreSQL persistence).
- **Agentic Multi-Step Reasoning**: Leverages dedicated planning and reasoning agents for complex multi-hop queries, with automated fallback from Gemini to Cloudflare Workers AI.
- **Asynchronous Deep Research**: Handles long-running research tasks in the background via ARQ queues and workers (scraping, crawling, embedding, and execution).
- **Multimodal Capabilities**: Integrates speech-to-text via Faster-Whisper and processes queries containing rich media inputs.
- **Built-in Quality Evaluation**: Evaluates research answers dynamically using an automated LLM-in-the-loop evaluator.
- **Production Observability**: Full metric integration with Prometheus, Grafana, OpenTelemetry, LangSmith, and structured logs.

---

## 🛠️ Tech Stack 

- **Framework**: FastAPI & Uvicorn (High-performance ASGI server)
- **Agent Framework**: LangChain 
- **Databases & Cache**: PostgreSQL (via SQLAlchemy & asyncpg), ChromaDB (Vector DB), and Redis (L2 caching & lock management)
- **LLM Providers**: Google Gemini (primary) and Cloudflare Workers AI (fallback/Llama models)
- **Information Retrieval**: Sentence-Transformers (Embeddings), BGE Cross-Encoder (Reranking), and Rank-BM25 (Lexical search)
- **Background Workers**: ARQ (Redis-based job queue)
- **Scraping & Crawling**: Playwright, BeautifulSoup4, and Readability
- **Observability**: Prometheus, Grafana, OpenTelemetry, and structlog
- **Package Manager / Build**: Python >= 3.11, Hatchling

---

## 📦 Project Structure

```text
├── agents/             # Core agent logic (answering, RAG, reasoning, router, search)
├── api/                # FastAPI application routes (main entrypoint, query handling, SSE)
├── config/             # Environment variables and configuration settings
├── core/               # Orchestration engine (complexity routing, caching, planning, context)
├── db/                 # DB schemas, models, and connection engine (PostgreSQL)
├── evaluation/         # Auto-evaluators to score generated answers
├── memory/             # Persistent chat/session memory store
├── migrations/         # Alembic database migration scripts
├── models/             # Pydantic request/response schemas
├── monitoring/         # Configuration files for Prometheus and Grafana dashboards
├── providers/          # Integrations for LLMs (Gemini, Cloudflare Workers AI fallback)
├── rag/                # Hybrid search, reranker, and vector store ingestion scripts
├── scripts/            # Shell and Python utility scripts
├── static/             # Static files and assets
├── tools/              # Agent tools (Playwright scraping, DuckDuckGo, GitHub, News, Serper, Whispers)
└── workers/            # Async ARQ task queue workers (crawl, embed, research, search)
```

---

## ⚙️ Setup and Run

### 1. Prerequisites
Ensure you have the following installed on your system:
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Fast Python package manager)
- Docker & Docker Compose
- API Keys: Google Gemini, Serper API

### 2. Environment Configuration
Copy the sample environment file and fill in your keys:
```bash
cp .env.example .env
```
Update the values in `.env` for your database configuration, `GEMINI_API_KEY`, `SERPER_API_KEY`, and optional Cloudflare Workers credentials.

### 3. Installation
Initialize the virtual environment and install all dependencies using `uv`:
```bash
uv sync
```
*Note: This automatically creates a `.venv` virtual environment and installs the project and its dependencies based on `pyproject.toml` and `uv.lock`.*

### 4. Run Services
You can run commands directly using `uv run` (which executes them within the virtual environment context), or manually activate the virtual environment beforehand:

**To activate manually:**
```bash
# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# On Linux/macOS
source .venv/bin/activate
```

#### Step A: Start Infrastructure (PostgreSQL & Redis)

**Using Docker (Recommended):**
```bash
docker-compose up -d
```

**Natively (Without Docker):**
Ensure that both PostgreSQL and Redis services are running on your local machine:
* **PostgreSQL**: Standard service active on port `5432` (ensure the credentials in [.env](file:///c:/Users/adity/DS_Projects/interview-project/perplexity_agent/.env#L19) match your local setup).
* **Redis**: Standard service active on port `6379`.
  * *On Windows (WSL)*: `sudo service redis-server start`
  * *On Windows (Native)*: Run `.\redis-server.exe` in your Redis directory.

#### Step B: Apply Database Migrations
```bash
uv run alembic upgrade head
```

#### Step C: Start the FastAPI API Server
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Step D: Run Async Workers
In a separate terminal, start the background queue:
```bash
uv run python -m arq workers.settings.WorkerSettings
```

### 5. Accessing the Applications
- **API Swagger Documentation**: http://localhost:8000/docs
- **Grafana Dashboards**: http://localhost:3000 (Default credentials: admin / admin)
- **Prometheus UI**: http://localhost:9090
