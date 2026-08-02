# syntax=docker/dockerfile:1
#
# Sehat inference service image.
#
# For reproducible deploys, pin the base image by digest, e.g.:
#   FROM python:3.11-slim@sha256:<digest> AS builder
# (look up the current digest with `docker buildx imagetools inspect
# python:3.11-slim`). Left unpinned here so local builds always track the
# latest 3.11 patch release.
#
# The model artifact is NOT baked into the image; mount it at runtime:
#   docker build -t sehat-serve .
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/artifacts:/models:ro" \
#     -e SEHAT_MODEL_PATH=/models/tb.int8.onnx \
#     sehat-serve

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package itself, then the serving runtime. Explicit serving
# deps keep the image correct even if pyproject's extras drift; pip dedups
# anything already declared there. torch is intentionally excluded — the
# image serves ONNX artifacts via onnxruntime (the torch fallback is for
# development environments only).
RUN pip install --upgrade pip \
    && pip install . \
    && pip install \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "onnxruntime>=1.17" \
        "python-multipart>=0.0.9" \
        "pillow>=10.0" \
        "numpy>=1.26"

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SEHAT_HOST=0.0.0.0 \
    SEHAT_PORT=8000 \
    SEHAT_CONFIG=/app/configs/serve/default.yaml

RUN groupadd --system sehat && useradd --system --gid sehat --no-create-home sehat

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
RUN chown -R sehat:sehat /app

USER sehat

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('SEHAT_PORT', '8000') + '/healthz', timeout=4)"]

CMD ["python", "-m", "sehat.serving"]
