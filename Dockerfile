FROM python:3.12-slim

# soundfile decodes audio through libsndfile (FLAC/WAV/MP3/OGG).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies before copying code: Docker caches this layer,
# so source edits don't re-download every package on each build.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only the files the service imports, plus the front-end.
# separation.py imports cleanly without torch (it's imported lazily
# there), so the slim image can still serve -- with separation
# disabled via /capabilities.
COPY analyze.py separation.py server.py ./
COPY static/ ./static/

# Run as an unprivileged user instead of root.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# 0.0.0.0 means "all interfaces" -- required for the port to be
# reachable from outside the container (127.0.0.1 inside a container
# is not the host's loopback).
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
