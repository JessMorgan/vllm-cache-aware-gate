# syntax=docker/dockerfile:1

# ---- builder: install the project + deps into a venv -----------------------
FROM python:3.12-slim AS builder

WORKDIR /app
# Note: README.md and Makefile are excluded from the build context by
# .dockerignore, so only the files needed to build the wheel are copied.
COPY pyproject.toml ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir .

# ---- runtime: minimal image with the venv ---------------------------------
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --uid 10001 gate
USER gate

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status==200 else 1)"

ENTRYPOINT ["python", "-m", "gate.main"]
