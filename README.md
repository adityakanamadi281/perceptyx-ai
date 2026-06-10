# Perceptyx-AI

A production-grade, Perplexity-class AI search engine built for high-performance, real-time query answering, and deep multi-source research.

---

## 🚀 Features

- **Real-Time Token Streaming**: Real-time token-by-token answer streaming on the frontend using Server-Sent Events (SSE) with progressive markdown parsing, auto-scrolling, and inline citation rendering.
- **Intelligent Query Complexity Classifier**: Dynamically classifies user queries (`SIMPLE`, `MEDIUM`, `COMPLEX`, or `RESEARCH`) using rules and LLM fallback checks to select the fastest, most cost-effective execution path.
- **Resilient Multi-Provider Search Aggregator**: Runs Serper, Tavily, and Firecrawl searches in parallel. Automatically deduplicates results by domain, filters URL duplicates, boosts authoritative sources, and falls back to DuckDuckGo in case of complete API failure.
- **Advanced Cross-Encoder Reranking**: Re-orders scraped snippets and search results using a local Sentence-Transformers cross-encoder (defaulting to the highly efficient `ms-marco-MiniLM-L-6-v2`), executing inference on a thread pool to avoid blocking the event loop.
- **High-Performance 2-Tier Scraper**: Completely replaces native Playwright rendering with a fast HTTPX crawler and a Jina Reader (`r.jina.ai`) fallback tier, bypassing paywalls and JS-heavy sites without container bloat. Includes duplicate content fingerprinting and sentence-boundary truncation.
- **Redis Semantic Caching**: Employs sentence-transformer embeddings to cache query-answer pairs. Uses dot-product cosine similarity over Redis key-scans with a configurable match threshold (`0.92`) to prevent duplicate LLM calls and search costs.
- **Structured Deep Research Worker**: Redesigns asynchronous reports into a 4-phase worker pipeline: outline planning, semaphore-limited parallel section crawling/reasoning, evidence-based drafting, and executive summary assembly.
- **Dynamic Frontend Dashboard**: An interactive UI presenting real-time stage tracking badges, a dedicated sliding right sidebar for source citations, interactive follow-up question chips, and localStorage data pruning.
- **Local Directory Document Ingester**: Features a command-line script to parse and chunk local folder contents (PDFs, Markdown, text) and ingest them directly into ChromaDB.
- **Production Observability**: Full metric integration with Prometheus, Grafana, OpenTelemetry, LangSmith, and structured logs.

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
├── agents/             # Core agent logic (answering, RAG, reasoning, router, search)
│   ├── answer.py       # Streaming and citation-verified answer synthesis
│   ├── router.py       # Concurrent sub-query routing classifier
│   └── search.py       # Aggregator search agent with cross-encoder rerank trigger
├── api/                # FastAPI application routes (SSE channels, query entrypoints)
│   ├── main.py         # Application lifespan, mounts, and startup pre-warming
│   └── query.py        # Research task queueing endpoints
├── config/             # Environment variables and Pydantic configuration schemas
│   └── settings.py     # Application and provider setting parameters
├── core/               # Orchestration engine (complexity routing, caching, planning)
│   ├── cache.py        # Redis connection pools and resilient ARQ settings
│   ├── complexity.py   # Heuristic and LLM fallback query classifiers
│   ├── orchestrator.py # Fast-path and full-path SSE streaming pipelines
│   └── semantic_cache.py # Redis-backed embedding cosine similarity caching
├── db/                 # DB schemas, models, and connection engine (PostgreSQL)
├── evaluation/         # Auto-evaluators to score generated answers
├── memory/             # Persistent chat/session memory store
├── migrations/         # Alembic database migration scripts
├── models/             # Pydantic request/response schemas
│   └── schemas.py      # Unified API definitions and event types
├── monitoring/         # Configuration files for Prometheus and Grafana dashboards
├── providers/          # Integrations for LLMs (Gemini, Cloudflare Workers AI fallback)
│   └── llm.py          # Unified provider fallback chain, rate limits, and async stream generators
├── rag/                # Hybrid search, reranker, and vector store ingestion scripts
│   ├── ingester.py     # Document parser and chunker logic
│   └── reranker.py     # Thread-safe cross-encoder search reranking
├── scripts/            # CLI utilities and startup scripts
│   └── ingest_directory.py # local folder data ingest tool for ChromaDB
├── static/             # Static files and assets
│   └── index.html      # Stream-capable single-page web dashboard
├── tools/              # Agent tools (DDG, GitHub, News, Serper, Whispers)
│   ├── search_aggregator.py # Multi-source aggregator, de-duplicator, and scorer
│   ├── tavily_search.py     # Tavily Search API client
│   └── firecrawl_search.py  # Firecrawl Search API client and deep scraper
└── workers/            # Async ARQ task queue workers (crawl, embed, research, search)
    ├── settings.py     # Resilient Worker connection settings
    └── research_worker.py  # Redesigned 4-phase deep research worker pipeline
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
