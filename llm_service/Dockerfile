# Agent API Service
# Lightweight container for the command interpreter API

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    openai \
    pydantic

COPY agent_api.py .

EXPOSE 8080

CMD ["python", "agent_api.py", "--host", "0.0.0.0", "--port", "8080"]
