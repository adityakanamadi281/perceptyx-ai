# Perceptyx-AI

A production-grade, Perplexity-class AI search engine built for high-performance, real-time query answering, and deep multi-source research.

---

## 🚀 Features

- **Real-Time Token Streaming**
- **Intelligent Query Complexity Classifier**
- **Resilient Multi-Provider Search Aggregator**
- **Advanced Cross-Encoder Reranking**
- **High-Performance 2-Tier Scraper**
- **Redis Semantic Caching**
- **Structured Deep Research Worker**
- **Dynamic Frontend Dashboard**
- **Local Directory Document Ingester**
- **Production Observability**

---

## 🛠️ Tech Stack

- **Backend ASGI Framework**: FastAPI & Uvicorn
- **Agent Orchestration**: LangChain Core
- **Distributed Cache & Queue**: Redis (L2 semantic cache, distributed locks, and ARQ background job queue)
- **Primary Database**: PostgreSQL (via SQLAlchemy & asyncpg)
- **Vector Search Database**: ChromaDB (with local sentence-transformers embeddings)
- **AI Models & LLM Providers**: Google Gemini (primary `SafeChatGoogleGenerativeAI` wrapper) and Cloudflare Workers AI (failover tier)
- **Inference & Reranking**: Sentence-Transformers (`all-MiniLM-L6-v2` for cache/RAG embeddings, `ms-marco-MiniLM-L-6-v2` for cross-encoder reranking)
- **Background Workers**: ARQ (Redis-based background scheduler)
- **Scraping Tools**: HTTPX, Jina Reader API, Readability-lxml, and BeautifulSoup4
- **Observability**: Prometheus, Grafana, OpenTelemetry, and structlog
- **Package Management**: [uv](https://github.com/astral-sh/uv) (Fast Python packager)

---

## 📦 Project Structure

```text
├── agents/      # Core agent logic (answering, RAG, reasoning, router, search)
├── api/         # FastAPI application routes (SSE channels, query entrypoints)
├── config/      # Environment variables and Pydantic configuration settings
├── core/        # Orchestration engine (complexity routing, caching, planning)
├── db/          # DB schemas, models, and connection engine (PostgreSQL)
├── evaluation/  # Auto-evaluators to score generated answers
├── memory/      # Persistent chat/session memory store
├── migrations/  # Alembic database migration scripts
├── models/      # Pydantic request/response schemas
├── monitoring/  # Configuration files for Prometheus and Grafana dashboards
├── providers/   # Integrations for LLMs (Gemini, Cloudflare Workers AI fallback)
├── rag/         # Hybrid search, reranker, and vector store ingestion scripts
├── scripts/     # CLI utilities and startup scripts
├── static/      # Static files and assets (single-page web dashboard)
├── tools/       # Agent tools (DDG, GitHub, News, Serper, Whispers)
└── workers/     # Async ARQ task queue workers (crawl, embed, research, search)
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
Update the values in `.env` for your database configuration, `GEMINI_API_KEY`, `SERPER_API_KEY`. To enable aggregate providers, fill in `TAVILY_API_KEY` and `FIRECRAWL_API_KEY`.

### 3. Installation
Initialize the virtual environment and install all dependencies using `uv`:
```bash
uv sync
```

### 4. Run Services

You can activate the virtual environment manually:
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
* **PostgreSQL**: Standard service active on port `5432`.
* **Redis**: Standard service active on port `6379`.

#### Step B: Apply Database Migrations
```bash
uv run alembic upgrade head
```

#### Step C: Start the FastAPI Server
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Step D: Run Async Workers
In a separate terminal, start the background queue:
```bash
uv run python -m arq workers.settings.WorkerSettings
```

---

## 📂 Local Knowledge Base Ingestion

To populate your local RAG search index from a folder of documents (PDFs, Markdown, plain text), use the built-in CLI ingester:
```bash
uv run python scripts/ingest_directory.py --dir ./path/to/your/documents
```
*By default, the script reads from `./data/raw_docs` and writes directly to ChromaDB.*

---

## 🔗 Accessing the Application

- **Web Dashboard Interface**: [http://localhost:8000](http://localhost:8000)
- **API Swagger Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Grafana Metrics Dashboard**: [http://localhost:3000](http://localhost:3000) (admin / admin)
- **Prometheus Metrics Collector**: [http://localhost:9090](http://localhost:9090)
