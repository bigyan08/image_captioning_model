FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip
RUN python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/

# Create models directory but don't populate it
RUN mkdir -p notebook/models

ENV PYTHONPATH=/app
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]