# AgentAudit API -- deploy target for Fly.io / Railway / any container host.
# Builds the FastAPI service only (api/main.py). The Streamlit dashboard is
# deployed separately -- see Dockerfile.streamlit and the README's deployment
# section for why these are split.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Real-mode extras (HF models) are NOT installed here by default -- the
# deployed API defaults to mode="synthetic"/"ci" so the container stays
# small and doesn't need a GPU. If you want the deployed API to serve
# mode="real" audits directly, uncomment the next two lines (adds a few GB
# and a slow first request while models download):
# COPY requirements-real.txt .
# RUN pip install --no-cache-dir -r requirements-real.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
