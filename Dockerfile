FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl git && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dep resolution
RUN pip install uv

COPY pyproject.toml ./
RUN uv pip install --system -e ".[prod]"

# Install Playwright browsers for JS-heavy scraping
RUN playwright install --with-deps chromium 2>/dev/null || true

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
